import argparse  # Module for parsing command-line arguments
import os  # Operating system module for file operations
import copy  # Module for shallow and deep copy operations

import torch  # PyTorch library
from torch import nn  # Neural network module
import torch.optim as optim  # Optimization algorithms in PyTorch
import torch.backends.cudnn as cudnn  # CuDNN library for GPU acceleration
from torch.utils.data.dataloader import DataLoader  # DataLoader for batching and data loading
from tqdm import tqdm  # Progress bar library

from models import DRRN  # Importing the DRRN model
from datasets import TrainDataset, EvalDataset  # Importing custom dataset classes
from utils import AverageMeter, denormalize, PSNR, load_weights  # Utility functions


if __name__ == '__main__':
    # Command-line argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, required=True)  # Training dataset file path
    parser.add_argument('--outputs-dir', type=str, required=True)  # Output directory for model checkpoints
    parser.add_argument('--eval-file', type=str)  # Optional evaluation dataset file path
    parser.add_argument('--eval-scale', type=int)  # Optional scale factor for evaluation
    parser.add_argument('--weights-file', type=str)  # Optional pretrained weights file
    parser.add_argument('--B', type=int, default=1)  # Number of recursive blocks in DRRN
    parser.add_argument('--U', type=int, default=9)  # Number of residual units per recursive block
    parser.add_argument('--num-features', type=int, default=128)  # Number of feature maps in DRRN
    parser.add_argument('--lr', type=float, default=0.1)  # Initial learning rate
    parser.add_argument('--clip-grad', type=float, default=0.01)  # Gradient clipping value
    parser.add_argument('--batch-size', type=int, default=128)  # Batch size for training
    parser.add_argument('--num-epochs', type=int, default=50)  # Number of training epochs
    parser.add_argument('--num-workers', type=int, default=8)  # Number of workers for data loading
    parser.add_argument('--seed', type=int, default=123)  # Random seed for reproducibility
    args = parser.parse_args()

    # Create an output directory for saving model checkpoints
    args.outputs_dir = os.path.join(args.outputs_dir, 'x234')
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    # Enable CuDNN benchmark mode for improved performance if GPU is available
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Use GPU if available
    if torch.cuda.is_available():
        print("using gpu")
    torch.manual_seed(args.seed)  # Set random seed for reproducibility

    # Initialize the DRRN model with specified parameters and move it to the device
    model = DRRN(B=args.B, U=args.U, num_features=args.num_features).to(device)

    # Load pretrained weights if provided
    if args.weights_file is not None:
        model = load_weights(model, args.weights_file)

    criterion = nn.MSELoss(reduction='sum')  # Mean squared error loss criterion
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)  # SGD optimizer

    # Create training dataset and dataloader
    train_dataset = TrainDataset(args.train_file)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True)  # DataLoader for training

    # Create evaluation dataset and dataloader if evaluation file is provided
    if args.eval_file is not None:
        eval_dataset = EvalDataset(args.eval_file)
        eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)  # DataLoader for evaluation

    best_weights = copy.deepcopy(model.state_dict())  # Initialize best weights with current model state
    best_epoch = 0  # Initialize best epoch counter
    best_psnr = 0.0  # Initialize best PSNR value

    # Training loop over epochs
    for epoch in range(args.num_epochs):
        lr = args.lr * (0.5 ** ((epoch + 1) // 10))  # Learning rate schedule

        # Update learning rate in optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        model.train()  # Set model to training mode
        epoch_losses = AverageMeter()  # AverageMeter to track epoch losses

        # Progress bar for training batches
        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size), ncols=80) as t:
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

            # Iterate over training batches
            for data in train_dataloader:
                inputs, labels = data  # Get inputs and labels

                inputs = inputs.to(device)  # Move inputs to device (GPU)
                labels = labels.to(device)  # Move labels to device (GPU)

                preds = model(inputs)  # Forward pass

                # Calculate loss and update epoch losses
                loss = criterion(preds, labels) / (2 * len(inputs))
                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()  # Zero gradients
                loss.backward()  # Backward pass

                # Clip gradients to prevent explosion
                nn.utils.clip_grad.clip_grad_norm_(model.parameters(), args.clip_grad / lr)

                optimizer.step()  # Optimizer step

                # Update progress bar with current loss and learning rate
                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg), lr=lr)
                t.update(len(inputs))  # Update progress bar

        # Save model checkpoint after each epoch
        torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))

        # Evaluation phase if evaluation file is provided
        if args.eval_file is not None:
            model.eval()  # Set model to evaluation mode
            epoch_psnr = AverageMeter()  # AverageMeter to track epoch PSNR

            # Iterate over evaluation batches
            for data in eval_dataloader:
                inputs, labels = data  # Get inputs and labels

                inputs = inputs.to(device)  # Move inputs to device (GPU)
                labels = labels.to(device)  # Move labels to device (GPU)

                with torch.no_grad():
                    preds = model(inputs)  # Forward pass without gradient computation

                preds = denormalize(preds.squeeze(0).squeeze(0))  # Denormalize predicted output
                labels = denormalize(labels.squeeze(0).squeeze(0))  # Denormalize ground truth

                # Calculate PSNR and update epoch PSNR
                epoch_psnr.update(PSNR(preds, labels, shave_border=args.eval_scale), len(inputs))

            print('eval psnr: {:.2f}'.format(epoch_psnr.avg))  # Print average PSNR for evaluation

            # Update best epoch and best PSNR if current epoch's PSNR is higher
            if epoch_psnr.avg > best_psnr:
                best_epoch = epoch
                best_psnr = epoch_psnr.avg
                best_weights = copy.deepcopy(model.state_dict())

    # Final evaluation results if evaluation file is provided
    if args.eval_file is not None:
        print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))  # Print best epoch and PSNR
        torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))  # Save best weights

