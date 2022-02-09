import argparse
import os

import numpy as np

from shallow2deep.helpers import utils


def main():
    parser = argparse.ArgumentParser(
        description="Extract single axis data from h5 volume and save it in the same volume")
    parser.add_argument("--source_path", type=str, required=True, help="H5 file or folder with source data")
    parser.add_argument("--src_internal_path", type=str, required=True,
                        help="Internal path for ilastik predictions")
    parser.add_argument("--tgt_internal_path", type=str, required=True,
                        help="Internal path to save fg Ilastik prediction")
    parser.add_argument("--axis", type=int, required=True, help="The axis where the probability channels are stored")
    parser.add_argument("--channel_num", type=int, required=True,
                        help="The channel num of the data. ")
    args = parser.parse_args()

    source_path = args.source_path
    src_internal_path = args.src_internal_path
    tgt_internal_path = args.tgt_internal_path
    axis = args.axis
    channel_num = args.channel_num

    if src_internal_path == tgt_internal_path:
        raise NameError(
            f"Source internal path '{src_internal_path}' can not be equal to "
            f"target internal path '{tgt_internal_path}'")

    if not os.path.isdir(source_path):
        source_path = [source_path]

    for file in source_path:
        prediction = np.take(utils.load_h5_volume(file, src_internal_path), channel_num, axis=axis)
        utils.write_h5_volume(file, tgt_internal_path, prediction)
