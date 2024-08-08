import argparse  # Module for parsing command-line arguments

import torch  # PyTorch library
import torch.backends.cudnn as cudnn  # CuDNN library for GPU acceleration
from torch.utils.data.dataloader import DataLoader  # DataLoader for batching and data loading

from models import DRRN  # Importing the DRRN model
from datasets import EvalDataset  # Importing the custom evaluation dataset class
from utils import AverageMeter, denormalize, PSNR, load_weights  # Utility functions

if __name__ == '__main__':
    # Command-line argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)  # Path to the pretrained weights file
    parser.add_argument('--eval-file', type=str, required=True)  # Path to the evaluation dataset file
    parser.add_argument('--eval-scale', type=int, required=True)  # Scale factor for evaluation
    parser.add_argument('--B', type=int, default=1)  # Number of recursive blocks in DRRN
    parser.add_argument('--U', type=int, default=9)  # Number of residual units per recursive block
    parser.add_argument('--num-features', type=int, default=128)  # Number of feature maps in DRRN
    args = parser.parse_args()

    cudnn.benchmark = True  # Enable CuDNN benchmark mode for improved performance
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Use GPU if available

    # Initialize the DRRN model with specified parameters and move it to the device
    model = DRRN(B=args.B, U=args.U, num_features=args.num_features).to(device)
    model = load_weights(model, args.weights_file)  # Load the pretrained weights into the model

    # Create evaluation dataset and dataloader
    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)  # DataLoader for evaluation

    # Evaluation phase
    if args.eval_file is not None:
        model.eval()  # Set model to evaluation mode
        epoch_psnr = AverageMeter()  # AverageMeter to track PSNR during evaluation

        # Iterate over evaluation batches
        for data in eval_dataloader:
            inputs, labels = data  # Get inputs and labels from the dataloader

            inputs = inputs.to(device)  # Move inputs to device (GPU)
            labels = labels.to(device)  # Move labels to device (GPU)

            # Forward pass without gradient computation
            with torch.no_grad():
                preds = model(inputs)

            # Denormalize predicted output and ground truth
            preds = denormalize(preds.squeeze(0).squeeze(0))
            labels = denormalize(labels.squeeze(0).squeeze(0))

            # Calculate PSNR and update the PSNR tracker
            epoch_psnr.update(PSNR(preds, labels, shave_border=args.eval_scale), len(inputs))

        # Print the average PSNR for the evaluation dataset
        print('eval psnr: {:.2f}'.format(epoch_psnr.avg))
