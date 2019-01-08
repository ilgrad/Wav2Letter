import argparse
from iyo.core.nn.modules.wav2letter import Wav2Letter
import numpy as np
import os

if __name__ == "__main__":
    # Load the args
    parser = argparse.ArgumentParser(description='Wav2Letter')
    parser.add_argument('--batch-size', type=int, default=56)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument("--npy_dir", type=str, default="npy")
    parser.add_argument("--checkpoint", type=str, default="checkpoint/wav2letter.pth")
    args = parser.parse_args()

    # Create the model
    model = Wav2Letter(index2char="{}/index2char.npy".format(args.npy_dir),
                       trace_file="/tmp/Wav2Letter_{}.txt".format(os.getpid()))

    # Fit the model
    model.fit(x_train=np.load("{}/x_train.npy".format(args.npy_dir)),
              y_train=np.load("{}/y_train.npy".format(args.npy_dir)),
              x_dev=np.load("{}/x_dev.npy".format(args.npy_dir)),
              y_dev=np.load("{}/y_dev.npy".format(args.npy_dir)),
              batch_size=args.batch_size,
              epochs=args.epochs,
              checkpoint=args.checkpoint)


