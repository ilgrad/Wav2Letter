'''
The structure should be for each Youtube video id
[id].csv
[id]_audio/
[id]_audio/
    [id_file].wav
[id]_[code]_txt/
    [id_file].txt


id_file respect the following sequence
[id]_[start_time]_[stop_time]_[length_audio]
'''
import random
import sys

import pandas as pd

from iyo.audio.__utils__ import get_audio_length
from iyo.core.__utils__ import *
from iyo.core.nn.utils.data.splitters.splitter import Splitter
from tqdm import tqdm
import argparse

class Manifest:
    def __init__(self, root, partitions={"train":0.75, "dev":0.10, "test":0.20}, range_duration=None, mini=False):
        """

        :param dir_data:
        :param code:
        """
        self.root = root
        self.partitions = partitions
        self.mini = mini
        self.range_duration = range_duration
        self.dataset_name = name(root).lower()


    def export_csv(self, csv_dir):
        """
        Export the csv file in the csv directory

        :param root:
        :return:
        """

        # Scan all the pairs of files
        print(">> Scaning audio and text files...")
        audio_files = dict([(name(file), file) for file in tqdm(listfiles_pattern(self.root, patterns=[".wav"]))
                            if (os.stat(file).st_size > 0)])
        text_files = dict([(name(file), file) for file in tqdm(listfiles_pattern(self.root, patterns=[".txt"]))
                           if (os.stat(file).st_size > 0)])

        # Get the pairs
        pairs = [(audio_files[key], text_files[key]) for key in tqdm(audio_files.keys())
                 if ((key in audio_files) & (key in text_files))]

        # Filter the duration
        pairs = [(k, v) for (k, v) in tqdm(pairs) if int(name(k).split("_")[0]) in self.range_duration] \
            if self.range_duration is not None else pairs

        # Mini
        if self.mini:
            random.shuffle(pairs)
            pairs = pairs[:100]

        # Get the splits
        self.manifest_splits = Splitter(pairs, self.partitions)

        # Export for each phase
        for phase in ["train", "dev", "test"]:
            manifest = self.manifest_splits.__getattribute__("x_{}".format(phase))
            prefixe = "{}_mini".format(self.dataset_name) if self.mini else self.dataset_name
            durations = "{}_{}".format(self.range_duration.start, self.range_duration.stop)\
                if self.range_duration is not None else "0_inf"
            csv_name = "{}-{}-{}".format(prefixe, durations, phase)
            csv_filename = "{}/{}.csv".format(csv_dir, csv_name)
            pd.DataFrame.from_dict({"0": manifest[:,0], "1": manifest[:,1]}).to_csv(csv_filename,
                                                                                    header=False,
                                                                                    index=False)
            print(">> {} exported".format(csv_filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Wav2Letter')
    parser.add_argument('--root', type=str, default="dataset")
    parser.add_argument('--partitions', type=dict, default={"train":0.8, "dev":0.1, "test":0.1})
    parser.add_argument('--csv_dir', type=str, default="data/csv")
    args = parser.parse_args()

    manifest = Manifest(root=args.root,
                        partitions=args.partitions,
                        range_duration=range(2,10))
    manifest.export_csv(csv_dir=args.csv_dir)
