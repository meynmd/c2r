import sys
import argparse

def train(model, phase, batch_size, num_epochs, train_data, val_data, model_dir,
          steps_per_checkpoint, num_batches_val, beam_size, visualize, output_dir,
          trie, learning_rate_init, lr_decay, start_decay_at):

    loss, num_seen, num_samples, num_nonzero, accuracy = 0, 0, 0, 0, 0
    learning_rate = learning_rate_init
    model.optim_state.learning_rate = learning_rate
    prev_loss = None
    val_losses = {}
    if phase == "train":
        forward_only = False
    elif phase == "test":
        forward_only = True
        num_epochs = 1
        model.global_step = 0
    else:
        raise NameError("phase must be either train or test")

    print("Lr: {}".format(learning_rate), file=sys.stderr)

    for epoch in range(1, num_epochs):
        if not forward_only:
            pass


def main():


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()