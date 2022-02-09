import argparse
import os
from shallow2deep.helpers import utils


def main():
    parser = argparse.ArgumentParser(description="Add volume to filters")
    parser.add_argument("--source_file", type=str, required=True, help="H5 file with source data")
    parser.add_argument("--save_path", type=str, required=True, help="h5 File or folder where "
                                                                     "the volume should be stored")
    parser.add_argument("--internal_path", type=str, default="raw", help="Internal path for raw data")
    args = parser.parse_args()

    source_file = args.source_file
    save_path = args.save_path
    internal_path = args.internal_path

    if not os.path.isdir(save_path):
        save_files = [save_path]
    else:
        save_files = [os.path.join(save_path, file) for file in os.listdir(save_path)]

    volume = utils.load_h5_volume(source_file, internal_path)

    for file in save_files:
        utils.write_h5_volume(file, internal_path, volume)
