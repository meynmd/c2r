import argparse, sys, math, os, glob
import numpy as np

import torch
import torch.nn.functional as F

import pr_dataset
# import baseline.rnn_baseline as auto
# import v5.cnn as auto
import v5.crnnv6 as auto


def crop_and_batch(orig_tensor, window_len, stride, max_batch):
    assert(orig_tensor.shape[1] >= window_len)
    cropped_tensors, start, stop = [], 0, window_len
    while stop < orig_tensor.shape[1]:
        cropped_tensors.append(orig_tensor[:, start : stop].unsqueeze(0))
        start += stride
        stop += stride

    batches = []
    if len(cropped_tensors) > max_batch:
        num_batches = int(math.ceil(len(cropped_tensors) / max_batch))
        for i in range(num_batches):
            b = cropped_tensors[i*max_batch : min((i+1)*max_batch, len(cropped_tensors))]
            batches.append(torch.stack(b))
    else:
        batches = [torch.stack(cropped_tensors)]

    return batches


'''
find PRF measures only on boundary points
'''
def count_on_off_performance(target, predicted):
    y = target.cpu().unsqueeze(1).type(torch.FloatTensor)
    yh = predicted.cpu().unsqueeze(1).type(torch.FloatTensor)

    f = torch.FloatTensor([[[[-1., 1., 0.]]]])
    gt_mask = F.conv2d(y, f, padding=(0, 1)).squeeze(1)
    p_mask = F.conv2d(yh, f, padding=(0, 1)).squeeze(1)

    # mask:  1 means note-on, -1 means note-off
    gt_onsets, gt_offsets = (gt_mask > 0).type(torch.LongTensor), (gt_mask < 0).type(torch.LongTensor)
    pred_onsets, pred_offsets = (p_mask > 0).type(torch.LongTensor), (p_mask < 0).type(torch.LongTensor)

    tp = gt_onsets*pred_onsets
    tn = gt_offsets*pred_offsets
    fp = pred_onsets - tp
    fn = pred_offsets - tn
    return tuple((t.sum().item() for t in (tp, fp, fn)))


'''
find (true pos, false pos, false negative), P&E counts for all frames and pitches
'''
def count_performance(target, predicted):
    y, yh = target.cpu(), predicted.cpu()
    pred_minus_targ = yh - y
    fp = (pred_minus_targ == 1).type(torch.LongTensor)
    fn = (pred_minus_targ == -1).type(torch.LongTensor)
    tp = yh * y
    counts = tuple((t.sum().item() for t in (tp, fp, fn)))

    # P&E error
    n_ref, n_sys = torch.sum(y, dim=1), torch.sum(yh, dim=1)
    err_count = torch.sum(torch.max(n_ref, n_sys) - torch.sum(tp, dim=1)).item()
    subst_count = torch.sum(torch.min(n_ref, n_sys) - torch.sum(tp, dim=1)).item()

    zero = torch.zeros(n_ref.shape).type(torch.LongTensor)
    miss_count = torch.sum(torch.max(zero, n_ref - n_sys)).item()
    fa_count = torch.sum(torch.max(zero, n_sys - n_ref)).item()

    n_ref_count = torch.sum(y).item()

    return counts, (err_count, subst_count, miss_count, fa_count, n_ref_count)

'''
find precision, recall, f-measure

with this batching method, stride has to be set to max_w, otherwise calculations will not be quite correct
'''
def test(net, data, max_w, batch_size=32, cuda_dev=None, labels=None, threshold=0.5, num_per_class=None):
    net.eval()
    stride = max_w
    if labels is None:
        labels = data.idx2name.items()

    p_total, r_total, f_total, err_t_total, err_s_total, err_m_total, err_f_total, num_pr_total = \
        0., 0., 0., 0., 0., 0., 0., 0
    nb_p_total, nb_r_total, nb_f_total = 0., 0., 0.

    for i, label in labels:
        class_datapoints = list(data.get_from_class(label))

        if num_per_class is None:
            num_per_class = len(class_datapoints)
        else:
            num_per_class = min(num_per_class, len(class_datapoints))

        class_datapoints = class_datapoints[:num_per_class]
        num_pr_total += num_per_class

        for (x, y), (composer, x_path) in class_datapoints:
            net.eval()
            tf_per_batch = batch_size*stride
            if x.shape[1] > max_w:
                x_batches = crop_and_batch(x, max_w, stride, batch_size)
            else:
                if x.shape[1] < max_w:
                    x = F.pad(x, (max_w - x.shape[1], 0))
                x_batches = [x.unsqueeze(0).unsqueeze(0)]

            x_tp, x_fp, x_fn, err_sum, subst_sum, miss_sum, fa_sum, n_ref_sum = 0., 0., 0., 0., 0., 0., 0., 0.
            x_true_onsets, x_false_offsets, x_false_onsets = 0., 0., 0.
            for j, batch in enumerate(x_batches):
                target = batch.clone().type(torch.LongTensor).squeeze(1)
                pr_size = target.shape[1] * target.shape[2]
                if cuda_dev is not None:
                    batch, target = batch.cuda(cuda_dev), target.cuda(cuda_dev)
                z = net(batch)
                p = torch.sigmoid(z).squeeze(1)

                y_hat = (p >= threshold).type(torch.LongTensor)

                # if cuda_dev is not None:
                #     y_hat = y_hat.cuda(cuda_dev)

                # pred_minus_targ = y_hat - target
                # fp = (pred_minus_targ == 1).type(torch.LongTensor)
                # fn = (pred_minus_targ == -1).type(torch.LongTensor)
                # tp = y_hat*target

                # true_onsets, false_onsets, false_offsets = find_boundary_tp_tn(target, y_hat)
                # batch_tp, batch_fp, batch_fn = (x.sum().item() for x in (tp, fp, fn))
                # batch_true_onsets, batch_false_onsets, batch_false_offsets = (x.sum().item() for x in (true_onsets, false_onsets, false_offsets))

                (batch_tp, batch_fp, batch_fn), (err_count, subst_count, miss_count, fa_count, n_ref_count) = \
                    count_performance(target, y_hat)

                batch_true_onsets, batch_false_onsets, batch_false_offsets = count_on_off_performance(target, y_hat)

                # num_tn = pr_size - num_tp - num_fp - num_fn

                x_tp += batch_tp
                x_fp += batch_fp
                x_fn += batch_fn
                # x_tn += num_tn

                x_true_onsets += batch_true_onsets
                x_false_onsets += batch_false_onsets
                x_false_offsets += batch_false_offsets

                # calculate Poliner & Ellis error for batch
                # n_ref, n_sys = torch.sum(target, dim=1), torch.sum(y_hat, dim=1)
                # err_sum += torch.sum(torch.max(n_ref, n_sys) - torch.sum(tp, dim=1)).item()
                # subst_sum += torch.sum(torch.min(n_ref, n_sys) - torch.sum(tp, dim=1)).item()
                err_sum += err_count
                subst_sum += subst_count

                # zero = torch.zeros(n_ref.shape).type(torch.LongTensor)
                # if cuda_dev is not None:
                #     zero = zero.cuda(cuda_dev)

                # miss_sum += torch.sum(torch.max(zero, n_ref - n_sys)).item()
                # fa_sum += torch.sum(torch.max(zero, n_sys - n_ref)).item()
                miss_sum += miss_count
                fa_sum += fa_count

                # n_ref_sum += torch.sum(target).item()
                n_ref_sum += n_ref_count

            # x_tp, x_fp, x_fn = (m/float(num_samples) for m in (x_tp, x_fp, x_fn))

            # print('{}\n\ttp: {}\tfp: {}\tfn: {}'.format(os.path.basename(x_path), x_tp, x_fp, x_fn))

            # calculate P, R, F measure
            precision = x_tp/(x_tp + x_fp)
            recall = x_tp/(x_tp + x_fn)
            f_score = 2*precision*recall/(precision + recall)

            # calculate measures for note onsets, offsets
            nb_precision = x_true_onsets/(x_true_onsets + x_false_onsets)
            nb_recall = x_true_onsets/(x_true_onsets + x_false_offsets)
            nb_f_score = 2*nb_precision*nb_recall/(nb_precision + nb_recall)

            err_total = err_sum/float(n_ref_sum)
            err_subs = subst_sum/float(n_ref_sum)
            err_miss = miss_sum/float(n_ref_sum)
            err_fa = fa_sum/float(n_ref_sum)

            # print('\tP: {:.4}\tR: {:.4}\tF: {:.4}'.format(precision, recall, f_score))
            # print(
            #     'e_tot (P&E): {:.4}\te_subs: {:.4}\te_miss: {:.4}\te_fa: {:.4}'.format(
            #         err_total, err_subs, err_miss, err_fa
            # ))
            sys.stdout.flush()

            p_total += precision
            r_total += recall
            f_total += f_score

            nb_p_total += nb_precision
            nb_r_total += nb_recall
            nb_f_total += nb_f_score

            err_t_total += err_total
            err_s_total += err_subs
            err_m_total += err_miss
            err_f_total += err_fa

        # print('\navg measures for class {}'.format(label))
        # print('P: {:.4}\tR: {:.4}\tF: {:.4}'.format(p_sum/num_per_class, r_sum/num_per_class, f_sum/num_per_class))
        # print(80*'*')
        # sys.stdout.flush()

    print('\noverall stats:\nP: {:.4}\tR: {:.4}\tF: {:.4}'.format(
        p_total/num_pr_total, r_total/num_pr_total, f_total/num_pr_total)
    )
    print('err_tot: {:.4}\terr_subs: {:.4}\terr_miss: {:.4}\terr_fa: {:.4}'.format(
        err_t_total/num_pr_total, err_s_total/num_pr_total, err_m_total/num_pr_total, err_f_total/num_pr_total
    ))

    print('nb_precision: {:.4}\tnb_recall: {:.4}\tnb_f_measure: {:.4}'.format(
        nb_p_total/num_pr_total, nb_r_total/num_pr_total, nb_f_total/num_pr_total
    ))



def roc(net, data, max_w, batch_size=32, cuda_dev=None, increment=0.05):
    net.eval()
    stride = max_w

    p_sum, r_sum, f_sum = (torch.zeros(int(1./increment)) for _ in range(3))
    for i in range(len(data)):
        x, _ = data.__getitem__(i)
        net.eval()
        tf_per_batch = batch_size*stride
        if x.shape[1] > max_w:
            x_batches = crop_and_batch(x, max_w, stride, batch_size)
        else:
            if x.shape[1] < max_w:
                x = F.pad(x, (max_w - x.shape[1], 0))
            x_batches = [x.unsqueeze(0).unsqueeze(0)]

        x_tp, x_fp, x_fn, n_ref_sum = (torch.zeros(int(1./increment)) for _ in range(4))

        for j, batch in enumerate(x_batches):
            target = batch.clone().squeeze(1)
            if cuda_dev is not None:
                batch, target = batch.cuda(cuda_dev), target.cuda(cuda_dev)
            z = net(batch)
            p = torch.sigmoid(z).squeeze(1)

            y_hat = torch.stack([(p >= th).type(torch.FloatTensor) for th in np.arange(0., 1., increment)], dim=0)
            target = torch.stack([target for _ in range(int(1./increment))], dim=0)

            if cuda_dev is not None:
                y_hat = y_hat.cuda(cuda_dev)

            pred_minus_targ = y_hat - target
            fp = (pred_minus_targ == 1).type(torch.FloatTensor)
            fn = (pred_minus_targ == -1).type(torch.FloatTensor)
            tp = y_hat*target
            batch_tp, batch_fp, batch_fn = (x.sum(dim=3).sum(dim=2).sum(dim=1) for x in (tp, fp, fn))

            x_tp += batch_tp.cpu()
            x_fp += batch_fp.cpu()
            x_fn += batch_fn.cpu()

        # calculate P, R, F measure
        precision = x_tp/(x_tp + x_fp + 1e-10)
        recall = x_tp/(x_tp + x_fn + 1e-10)
        f_score = 2*precision*recall/(precision + recall + 1e-10)

        p_sum += precision
        r_sum += recall
        f_sum += f_score

    num_pr = len(data)
    print('Threshold: Precision,Recall,F-measure')
    for i in range(p_sum.shape[0]):
        print('{:.3}: {:.4},{:.4},{:.4}'.format(i*increment, p_sum[i]/num_pr, r_sum[i]/num_pr, f_sum[i]/num_pr))

    best = torch.argmax(f_sum, dim=0).item()
    print('choosing threshold {:.4}'.format(best*increment))
    return best*increment


def main(opts):
    if opts.use_cuda is not None:
        cuda_dev = int(opts.use_cuda)
    else:
        cuda_dev = None

    # load the data
    val_set = pr_dataset.PianoRollDataset(os.getcwd() + "/" + opts.data_dir, "labels.csv", "val")
    dataset = pr_dataset.PianoRollDataset(os.getcwd() + "/" + opts.data_dir, "labels.csv", "test")

    # crnn
    net = auto.AutoEncoder(batch_size=opts.batch_size, rnn_size=opts.rnn_size, rnn_layers=1, use_cuda=opts.use_cuda, max_w=opts.max_w)

    # rnn
    # net = auto.AutoEncoder(rnn_size=opts.rnn_size, rnn_layers=1, batch_size=opts.batch_size, use_cuda=opts.use_cuda)

    # cnn
    # net = auto.AutoEncoder(batch_size=opts.batch_size, use_cuda=opts.use_cuda, max_w=opts.max_w)

    for param in net.parameters():
        param.requires_grad = False

    print('evaluating model architecture {}'.format(net.name), file=sys.stderr)
    sys.stderr.flush()

    saved_state = torch.load(opts.weights, map_location='cpu')
    net.load_state_dict(saved_state)
    if cuda_dev is not None:
        net = net.cuda(cuda_dev)

    for param in net.parameters():
        param.requires_grad = False

    labels = dataset.idx2name.items()
    if opts.classes is not None:
        selected_labels = [s.replace('-', ' ') for s in opts.classes]
        labels = [(i, l) for i, l in labels if l in selected_labels]
        print ('evaluating on classes: {}'.format([l for i, l in labels]))

    label2idx = {l : i for (i, l) in labels}

    with torch.no_grad():
        if opts.threshold is None:
            threshold = roc(net, val_set, opts.max_w, batch_size=opts.batch_size, cuda_dev=opts.use_cuda, increment=0.05)
        else:
            threshold = opts.threshold

        print('predicting using threshold {}'.format(threshold))
        sys.stdout.flush()

        test(net, dataset, opts.max_w, batch_size=opts.batch_size, cuda_dev=opts.use_cuda, labels=labels, threshold=threshold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../../preprocessed")
    parser.add_argument("--max_w", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("-c", "--use_cuda", type=int, default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--classes", nargs='*')
    parser.add_argument("--rnn_size", type=int, default=1024)
    parser.add_argument("--threshold", type=float, default=None)
    # implement left padding?

    args = parser.parse_args()
    main(args)

