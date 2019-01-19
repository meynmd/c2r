import sys
import numpy as np

# find_min_right
# find the minimum value at or to the right of i
def find_min_right(series, peak_idx, first=None, end=None):
    if first == None:
        first = 0
    if end == None:
        end = peak_idx + 1

    m = end*[series[end - 1]]
    for i in range(end - 2, -1, first - 1):
        m[i] = min(m[i + 1], series[i])

    # m = peak_idx*[0] + [series[peak_idx]]
    # for i in range(end - 1, -1, first - 1):
    #     m[i] = min(m[i + 1], series[i])

    return m


# find_max_left
# find the maximum value strictly to the left of i
def find_max_left(series, peak_idx, first=None, end=None):
    if first == None:
        first = 0
    if end == None:
        end = peak_idx + 1

    m = end*[float('-inf')]
    for i in range(first + 1, end):
        m[i] = max(m[i - 1], series[i - 1])

    # m = [float('-inf')] + peak_idx*[0]
    # for i in range(first + 1, end + 1):
    #     m[i] = max(m[i - 1], series[i - 1])

    return m


# find_min_left
# find the minimum value to the left of or at i
def find_min_left(series, peak_idx, first=None, end=None):
    if first == None:
        first = peak_idx
    if end == None:
        end = series.shape[0]

    m = len(series)*[series[max(0, first - 1)]]
    for i in range(first, end):
        m[i] = min(m[i - 1], series[i])

    return m

    '''
    m = series.shape[0]*[0]

    m = series.shape[0]*[series[0]]
    for i in range(1, series.shape[0]):
        m[i] = min(m[i - 1], series[i])
    '''

    # if peak_idx > 0:
    #     m[peak_idx - 1] = float('-inf')

    # m[peak_idx] = series[peak_idx]
    # for i in range(first + 1, end):
    #     m[i] = min(m[i - 1], series[i])

    # return m


# find_max_right
# find the maximum value of all values strictly to the right of i
def find_max_right(series, peak_idx, first=None, end=None):
    if first == None:
        first = peak_idx
    if end == None:
        end = series.shape[0]

    m = len(series)*[float('-inf')]
    for i in range(end - 2, first - 1, -1):
        m[i] = max(m[i + 1], series[i + 1])
    return m


def find_left_plateau(series, l_max, r_min, peak_idx, first, end, threshold_min):
    val_max = threshold_min
    for i in range(first, end):
        val_max = max(val_max, l_max[i])
        if r_min[i] >= l_max[i] and r_min[i] >= threshold_min:
            return i, val_max
    return None, None


def find_right_plateau(series, r_max, l_min, peak_idx, first, end, threshold_min):
    val_max = threshold_min
    for i in range(end - 1, first - 1, -1):
        val_max = max(val_max, r_max[i])
        if l_min[i] >= r_max[i] and l_min[i] >= threshold_min:
            return i, val_max
    return None, None


def find_plateau(series, tolerance, k=1):
    # vals_peak, idxs_peak = torch.topk(series, k, dim=0)     # find top-k time position; for now let's do 1
    # for peak, t_peak in zip(vals_peak, idxs_peak):
    # peak, t_peak = vals_peak[0].item(), idxs_peak[0].item()

    # t_peak = max(range(len(series)), key=lambda x : series[x])
    # peak = series[t_peak]

    # series = series.numpy()
    peak, t_peak = np.max(series), np.argmax(series)

    while True:

        thr_min = peak - tolerance

        r_min, l_max, l_min, r_max = find_min_right(series, t_peak), find_max_left(series, t_peak), \
                                     find_min_left(series, t_peak), find_max_right(series, t_peak)

        plateau_lb, thr_l = find_left_plateau(series, l_max, r_min, t_peak, first=0, end=min(len(series), t_peak + 1), threshold_min=thr_min)
        plateau_rb, thr_r = find_right_plateau(series, r_max, l_min, t_peak, first=t_peak, end=series.shape[0], threshold_min=thr_min)

        while abs(thr_l - thr_r) > 0.1*tolerance:
            r_min, l_max, l_min, r_max = find_min_right(series, t_peak, first=plateau_lb), find_max_left(series, t_peak, first=plateau_lb), \
                                         find_min_left(series, t_peak, end=plateau_rb), find_max_right(series, t_peak, end=plateau_rb)
            if thr_l > thr_r:
                plateau_rb, thr_r = find_right_plateau(series, r_max, l_min, t_peak, first=t_peak, end=plateau_rb, threshold_min=thr_l)
            elif thr_l < thr_r:
                plateau_lb, thr_l = find_left_plateau(series, l_max, r_min, t_peak, first=plateau_lb, end=t_peak, threshold_min=thr_r)

            if thr_l is None or thr_r is None:
                tolerance *= 2
                break

        if plateau_lb is not None and plateau_rb is not None:
            return (plateau_lb, plateau_rb), thr_l

if __name__ == '__main__':
    series = np.load(sys.argv[1])
    thr = float(np.std(series))

    (left, right), _ = find_plateau(series, thr)
    print(left, right)