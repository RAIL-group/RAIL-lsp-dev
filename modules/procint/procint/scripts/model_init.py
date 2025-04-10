import os
import torch
import argparse

import procint


def save_model(args=None):
    # Set up the learning
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = procint.learning.models.fcnn.IntvFcnn(args).to(device)
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'fcnn.pt'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create a randomly initialized network for intervention learning project.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--save_dir', type=str, required=True)
    args = parser.parse_args()
    save_model(args)
