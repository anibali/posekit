import argparse

import torch

from posekit.gui.video_player import VideoPlayer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True,
                        help='path to video file')
    return parser.parse_args()


def main():
    torch.set_grad_enabled(False)
    opts = parse_args()
    VideoPlayer(opts.video).run()


if __name__ == '__main__':
    main()
