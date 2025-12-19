import argparse
import sys
import torch
import random
import numpy as np
from test_quantize import SimpleTrainer2d 
from pathlib import Path


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Fit Gaussian to images")
    parser.add_argument(
        "-d", "--data_path", type=str, default="$DATA/afhq", help="Path to the image data"
    )
    parser.add_argument(
        "--model_path", type=str, default="./checkpoints_quant/GaussianImage_Cholesky.pth", help="Path to the pre-trained model"
    )
    parser.add_argument(
        "--data_name", type=str, default="afhq", help="Name of the image file")
    parser.add_argument(
        "--model_name", type=str, default="GaussianImage_Cholesky", help= "Name of the model"
    )
    parser.add_argument(
        "--output_path", type=str, default="./result", help="Path to save output"
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--num_points", type=int, default=2048, help="Number of points to sample from the Gaussian"
    )

def fit(imgs, models, num_pts):
    comp = []
    for i in range(len(imgs)):
        trainer = SimpleTrainer2d(image_path=imgs[i], num_points=num_pts, 
            iterations=50000, model_path=models[i])
        _, comp_dict = trainer.test()
        comp.append(comp_dict)
    return comp
    
def main(argv):
    args = parse_args(argv)
    data_path = args.data_path
    model_path = args.model_path
    output_path = Path(f"{args.output_path}/{args.model_name}_{args.data_name}") 

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)

    images = []
    models = []
    for i in range(24):
        image_path = Path(data_path) / f"{args.data_name}{i+1:02}.png"
        model_path = Path(model_path) / f"{args.model_name}{i+1:02}.pth"
        images.append(image_path)
        models.append(model_path)
    
    comps = fit(images, models, args.num_points)

    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    for i in range(len(images)):
        comp_dict = comps[i]
        comp_output_path = output_path / f"comp_{args.data_name}{i+1:02}_pts{args.num_points}.pth"
        torch.save(comp_dict, comp_output_path)
        print(f"Saved compression results to {comp_output_path}")

if __name__ == "__main__":
    main(sys.argv[1:])