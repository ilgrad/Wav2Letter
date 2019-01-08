from __future__ import print_function
from iyo.core.nn.utils.integer_encode import IntegerEncode
import os
from sonopy import mfcc_spec
from scipy.io.wavfile import read
from tqdm import tqdm
import random
from iyo.core.nn.utils.wav2letter import *
from iyo.core.__utils__ import read_csv, read_txt
import argparse

class DataPreparator:
    def __init__(self, manifest_prefixe, sr=16000):
        """

        :param manifest:
        :param sr:
        """
        self.manifest_prefixe = manifest_prefixe
        self.labels = [[np.unique([read_txt(txt_path)[0].replace(" ", "") for (_, txt_path) in read_csv(manifest)])]
                       for manifest in ["{}-train.csv".format(args.manifest_prefixe),
                                        "{}-dev.csv".format(args.manifest_prefixe),
                                        "{}-test.csv".format(args.manifest_prefixe)]]
        self.labels = np.concatenate([labels[0] for labels in self.labels], axis=0)
        self.intencode = IntegerEncode(self.labels)

        self.sr = sr
        self.max_frame_len = 1250
        # Get the data
        (self.x_train, self.y_train), (self.x_dev, self.y_dev), (self.x_test, self.y_test) = self.get_data()

    def get_data(self, progress_bar=True):
        def __get_data__( manifest, sr,max_frame_len, intencode):
            """

            :param progress_bar:
            :return:
            """
            pg = tqdm if progress_bar else lambda x: x
            random.shuffle(manifest)
            inputs, targets = [], []
            for md in pg(manifest):
                audio_path = md[0]
                labels_path = md[1]
                _, audio = read(audio_path)
                # audio = load_audio(audio_path)
                labels = read_txt(labels_path)[0].replace(" ", "")
                mfccs = mfcc_spec(
                    audio, sr, window_stride=(160, 80),
                    fft_size=512, num_filt=20, num_coeffs=13
                )
                mfccs = normalize(mfccs)
                diff = max_frame_len - mfccs.shape[0]
                if diff>=0:
                    mfccs = np.pad(mfccs, ((0, diff), (0, 0)), "constant")
                    target = intencode.convert_to_ints(labels)
                    if target is not None:
                        inputs.append(mfccs)
                        targets.append(target)
            return (inputs, targets)
        return [__get_data__(read_csv(manifest), sr=self.sr, max_frame_len=self.max_frame_len, intencode=self.intencode)
                for manifest in ["{}-train.csv".format(self.manifest_prefixe),
                                 "{}-dev.csv".format(self.manifest_prefixe),
                                 "{}-test.csv".format(self.manifest_prefixe)]]

    def export(self, npy_dir, pkl_dir):
        """

        :param file_path:
        :param x:
        :param y:
        :return:
        """
        # X
        np.save("{}/x_train.npy".format(npy_dir), np.asarray(self.x_train))
        np.save("{}/x_dev.npy".format(npy_dir), np.asarray(self.x_dev))
        np.save("{}/x_test.npy".format(npy_dir), np.asarray(self.x_test))
        # Y
        np.save("{}/y_train.npy".format(npy_dir), np.asarray(self.y_train))
        np.save("{}/y_dev.npy".format(npy_dir), np.asarray(self.y_dev))
        np.save("{}/y_test.npy".format(npy_dir), np.asarray(self.y_test))
        #IntEncoder
        self.intencode.save("{}/int_encoder.pkl".format(pkl_dir))


if __name__ == "__main__":
    # Load the args
    parser = argparse.ArgumentParser(description='Wav2Letter')
    parser.add_argument('--manifest_prefixe',
                        type=str,
                        default="data/csv/jsut_16000-0_inf")
    parser.add_argument('--npy_dir',
                        type=str,
                        default="data/npy")
    parser.add_argument('--pkl_dir',
                        type=str,
                        default="data/pkl")
    args = parser.parse_args()

    # Create the preparator
    data = DataPreparator(manifest_prefixe=args.manifest_prefixe)
    
    # Export
    data.export(args.npy_dir, args.pkl_dir)

    print("preprocessed and saved")
