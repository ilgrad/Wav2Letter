from __future__ import print_function
from iyo.core.nn.utils.data.decoders.wav2letter_greedy_decoder import Wav2LetterGreedyDecoder
import math
import torch.nn.functional as F
import torch.optim as optim
import argparse
import torch
from torch import nn
from iyo.core.__utils__ import Printer
import os
import numpy as np
import builtins
import pickle
import random
from tqdm import tqdm

class Wav2Letter(nn.Module):
    def __init__(self, 
                 index2char=None,
                 checkpoint=None,
                 trace_file=None):
        """

        :param num_classes:
        :param index2char:
        :param checkpoint:
        """
        super(Wav2Letter, self).__init__()
        self._index2char = pickle.load(open(index2char, "rb"))["index2char"]
        self._num_classes = len(self._index2char)
        self._checkpoint = checkpoint
        self.trace_file = trace_file
        self.printer = Printer(self.trace_file) if self.trace_file is not None else builtins.print

        
        # Conv1d(in_channels, out_channels, kernel_size, stride)
        layers = nn.Sequential(
            nn.Conv1d(13, 376, 48, 2),
            torch.nn.ReLU(),
            nn.Conv1d(376, 376, 7),
            torch.nn.ReLU(),
            nn.Conv1d(376, 376, 7),
            torch.nn.ReLU(),
            nn.Conv1d(376, 376, 7),
            torch.nn.ReLU(),
            nn.Conv1d(376, 376, 7),
            torch.nn.ReLU(),
            nn.Conv1d(376, 376, 7),
            torch.nn.ReLU(),
            nn.Conv1d(376, 376, 7),
            torch.nn.ReLU(),
            nn.Conv1d(376, 376, 7),
            torch.nn.ReLU(),
            nn.Conv1d(376, 2500, 32),
            torch.nn.ReLU(),
            nn.Conv1d(2500, 2500, 1),
            torch.nn.ReLU(),
            nn.Conv1d(2500, self._num_classes, 1),
        )
        # self._layers = DataParallelModel(layers)
        self._layers = layers.cuda()
        print(self._layers)
        self._ctc_loss = nn.CTCLoss()
        self._optimizer = optim.Adam(self._layers.parameters(), lr=1e-4)


    def fit(self,
            x_train,
            y_train,
            x_dev,
            y_dev,
            batch_size=56,
            epochs=1,
            checkpoint=None,
            print_every=0.1):
        """

        :param x_train:
        :param y_train:
        :param x_dev:
        :param y_dev:
        :param batch_size:
        :param epochs:
        :param checkpoint:
        :param print_every:
        :return:
        """
        print = self.printer
        # torch tensors
        x_train = torch.Tensor(x_train).cuda()
        y_train = torch.IntTensor(y_train).cuda()

        x_dev = torch.Tensor(x_dev).cuda()
        y_dev = torch.IntTensor(y_dev).cuda()

        x_train = x_train.transpose(1, 2)
        x_dev = x_dev.transpose(1, 2)

        # Verbose
        print("transposed x_train {}".format(x_train.shape))
        print("training google speech dataset")
        print("data size {}".format(len(x_train)))
        print("batch_size {}".format(batch_size))
        print("epochs {}".format(epochs))
        print("grapheme_count {}".format(self._num_classes))
        print("input shape {}".format(x_train.shape))
        print("y_train shape {}".format(y_train.shape))

        # Load the checkpoint
        self.load_state_dict(torch.load(checkpoint)) if os.path.isfile(checkpoint) else None

        # Get the total number of steps
        total_steps = math.ceil(len(x_train) / batch_size)
        for epoch in range(epochs):
            samples_processed = 0
            loss_train = 0
            # random.shuffle(inputs)
            pg = tqdm
            for step in pg(range(total_steps)):

                #  Fit the batch
                self._fit_batch_(x=x_train[samples_processed:batch_size + samples_processed],
                                 y=y_train[samples_processed: batch_size + samples_processed])

                # Add the batch loss
                loss_train += self._loss.item()

                # Update the number samples processed
                samples_processed += batch_size

                # Eval the model
                if (step % int(total_steps*print_every))==0:
                    # Eval one batch
                    ind = [random.randint(0, len(x_dev)-1)]
                    _, _, loss_eval_avg = self.evaluate(x_dev[ind],
                                                        y_dev[ind],
                                                        batch_size=1,
                                                        decode=True,
                                                        verbose=True)
                    loss_train_avg = loss_train / (step + 1)
                    print("{}/{} >> step {}/{}, train_loss {}, eval_loss {} ".format(epoch,
                                                                                     epochs,
                                                                                     step,
                                                                                     total_steps,
                                                                                     loss_train_avg,
                                                                                     loss_eval_avg))
            # Eval all
            _, _, loss_eval_avg = self.evaluate(x_dev,
                                                y_dev,
                                                batch_size=1)

            print("{}/{} >> step {}/{}, loss {}, avg loss {} ".format(epoch,
                                                                      epochs,
                                                                      step,
                                                                      total_steps,
                                                                      loss_train_avg,
                                                                      loss_eval_avg))
            print("---------------------------------------------------------------------------------------")
            # Save the model
            print(">> Saving the dictionary {} ".format(checkpoint))
            torch.save(self.state_dict(), checkpoint)

    def evaluate(self, x_dev, y_dev, batch_size=32, decode=True, verbose=False):
        def _decode_(sample_target, index2char):
            return "".join([index2char[i] for i in sample_target if i > 1])
        """

        :param x_dev:
        :param y_dev:
        :param decode:
        :return:
        """
        loss_eval = 0
        total_steps = math.ceil(len(x_dev) / batch_size)
        sample_targets, samples_predicted = [], []
        for step in range(total_steps):
            # Batch evaluation
            y = y_dev[(step)*batch_size:(step+1)*batch_size]
            loss, y_pred = self._loss_batch_(x_dev[(step)*batch_size:(step+1)*batch_size],
                                             y_dev[(step)*batch_size:(step+1)*batch_size])
            loss_eval += loss.item()

            #Pick up one randomly
            ind = random.randint(0,len(y)-1)
            y_pred, y = y_pred[ind], y[ind]

            # Values
            y_pred = y_pred.reshape(1, y_pred.shape[0], y_pred.shape[1])
            y_pred = Wav2LetterGreedyDecoder(y_pred)

            # Cpu
            y = np.array(y.cpu())
            y_pred = np.array(y_pred.cpu(), dtype=np.int32)

            # Get the string
            y = _decode_(y, self._index2char)  if decode else y
            y_pred = _decode_(y_pred, self._index2char) if decode else y_pred

            # Verbose
            print("\n{}/{} >> R:{} - T:{}".format(step, total_steps, y, y_pred)) if verbose else None

            # Append
            sample_targets.append(y)
            samples_predicted.append(y_pred)

        # Average loss
        loss_avg = loss/total_steps

        # Return
        return sample_targets, samples_predicted, loss_avg

    def _loss_batch_(self, x, y):
        """

        :param x:
        :param y:
        :return:
        """
        # Forward
        y_pred = self.__call__(x)
        y_pred = y_pred.transpose(1, 2).transpose(0, 1)
        x_lengths = torch.full((x.shape[0],), y_pred.shape[0], dtype=torch.long)
        y_lengths = torch.IntTensor([_y.shape[0] for _y in y])

        loss = self._ctc_loss(y_pred, y, x_lengths, y_lengths)
        y_pred = y_pred.transpose(0,1)
        y_pred = y_pred.transpose(1,2)

        # Backpropagate
        return loss, y_pred

    def _fit_batch_(self, x, y):
        """

        :param x:
        :param y:
        :return:
        """
        self._optimizer.zero_grad()
        self._loss, _ = self._loss_batch_(x, y)
        self._loss.backward()
        self._optimizer.step()

    def __call__(self, x, *args, **kwargs):
        """

        :param batch:
        :return:
        """
        # y_pred shape (batch_size, num_classes, output_len)
        y_pred = self._layers(x)

        # compute log softmax probability on graphemes
        y_pred = F.log_softmax(y_pred, dim=1)

        return y_pred



if __name__ == "__main__":
    # Load the args
    parser = argparse.ArgumentParser(description='Wav2Letter')
    parser.add_argument('--batch-size', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument("--npy_dir", type=str, default="data/npy")
    parser.add_argument("--pkl_dir", type=str, default="pkl/pkl")
    parser.add_argument("--checkpoint", type=str, default="checkpoint/wav2letter.pth")
    args = parser.parse_args()

    # Create the model
    model = Wav2Letter(index2char="{}/int_encoder.pkl".format(args.pkl_dir))

    # Train the model
    model.fit(x_train=np.load("{}/x_train.npy".format(args.npy_dir)),
              y_train=np.load("{}/y_train.npy".format(args.npy_dir)),
              x_dev=np.load("{}/x_dev.npy".format(args.npy_dir)),
              y_dev=np.load("{}/y_dev.npy".format(args.npy_dir)),
              batch_size=args.batch_size,
              epochs=args.epochs,
              checkpoint=args.checkpoint)

