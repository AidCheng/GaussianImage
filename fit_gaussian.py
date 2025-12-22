import argparse
import sys
import torch
import random
import numpy as np
from test_quantize import SimpleTrainer2d 
from pathlib import Path


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-d", "--dataset", type=str, default='./dataset/kodak/', help="Training dataset"
    )
    parser.add_argument(
        "--data_name", type=str, default='kodak', help="Training dataset"
    )
    parser.add_argument(
        "--iterations", type=int, default=50000, help="number of training epochs (default: %(default)s)"
    )
    parser.add_argument(
        "--model_name", type=str, default="GaussianImage_Cholesky", help="model selection: GaussianImage_Cholesky, GaussianImage_RS, 3DGS"
    )
    parser.add_argument(
        "--sh_degree", type=int, default=3, help="SH degree (default: %(default)s)"
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=50000,
        help="2D GS points (default: %(default)s)",
    )
    parser.add_argument("--model_path", type=str, default=None, help="Path to a checkpoint")
    parser.add_argument("--seed", type=float, default=1, help="Set random seed for reproducibility")
    parser.add_argument("--quantize", action="store_true", help="Quantize")
    parser.add_argument("--save_imgs", action="store_true", help="Save image")
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument("--pretrained", type=str, help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args

def fit(imgs, models, num_pts, args):
    comp = []
    for i in range(len(imgs)):
        trainer = SimpleTrainer2d(image_path=imgs[i], num_points=num_pts, 
            iterations=50000, model_path=models[i], args=args)
        _, comp_dict = trainer.test()
        comp.append(comp_dict)
    return comp
    
def main(argv):
    args = parse_args(argv)
    data_path = args.dataset
    model_path = args.model_path
    output_path = Path(f"./result/{args.model_name}_{args.data_name}_{args.num_points}/") 

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)

    images = []
    models = []

    if args.data_name == "afhq":
        num_images = 416 

    for i in range(num_images):
        image = Path(data_path) / f"{args.data_name}{i:02}.jpg"
        model = Path(model_path) / f"{args.data_name}{i:02}/gaussian_model.best.pth.tar"
        images.append(image)
        models.append(model)
    
    comps = fit(images, models, args.num_points, args)

    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    for i in range(len(images)):
        comp_dict = comps[i]
        comp_output_path = output_path / f"comp_{args.data_name}{i:02}_pts{args.num_points}.pth"
        torch.save(comp_dict, comp_output_path)
        print(f"Saved compression results to {comp_output_path}")

if __name__ == "__main__":
    main(sys.argv[1:])