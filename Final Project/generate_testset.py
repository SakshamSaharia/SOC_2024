import argparse  # Import argparse for command-line argument parsing
import glob  # Import glob for file path pattern matching
import h5py  # Import h5py for handling HDF5 file operations
import numpy as np  # Import numpy for numerical operations
from utils import load_image, modcrop, generate_lr, image_to_array, rgb_to_y, normalize
# Import utility functions from a separate module (utils) for image processing tasks

if __name__ == '__main__':
    # Argument parser setup for command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', type=str, required=True)
    # Argument for the directory containing the input images
    parser.add_argument('--output-path', type=str, required=True)
    # Argument for the output HDF5 file path
    parser.add_argument('--scale', type=int, default=2)
    # Argument for the scaling factor (default is 2)
    args = parser.parse_args()
    # Parse the arguments provided via the command line

    # Create an HDF5 file for storing the low-resolution and high-resolution images
    h5_file = h5py.File(args.output_path, 'w')

    # Create groups in the HDF5 file for low-resolution (lr) and high-resolution (hr) images
    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')

    # Iterate over all images in the specified directory
    for i, image_path in enumerate(sorted(glob.glob('{}/*'.format(args.images_dir)))):
        hr = load_image(image_path)  # Load the high-resolution image
        hr = modcrop(hr, args.scale)  # Crop the image to be divisible by the scale factor
        lr = generate_lr(hr, args.scale)  # Generate a low-resolution image from the high-resolution image

        hr = image_to_array(hr)  # Convert the high-resolution image to a numpy array
        lr = image_to_array(lr)  # Convert the low-resolution image to a numpy array

        # Convert the RGB images to Y-channel (luminance) and normalize them
        hr = np.expand_dims(normalize(rgb_to_y(hr.astype(np.float32), 'chw')), 0)
        lr = np.expand_dims(normalize(rgb_to_y(lr.astype(np.float32), 'chw')), 0)

        # Store the processed high-resolution and low-resolution images in the HDF5 file
        hr_group.create_dataset(str(i), data=hr)
        lr_group.create_dataset(str(i), data=lr)

    # Close the HDF5 file after saving all the data
    h5_file.close()
