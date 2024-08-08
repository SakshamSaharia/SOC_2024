import argparse  # Import argparse for command-line argument parsing
import glob  # Import glob for file path pattern matching
import h5py  # Import h5py for handling HDF5 file operations
import numpy as np  # Import numpy for numerical operations
from utils import load_image, modcrop, generate_lr, generate_patch, image_to_array, rgb_to_y, normalize
# Import utility functions from a separate module (utils) for image processing tasks

if __name__ == '__main__':
    # Argument parser setup for command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', type=str, required=True)
    # Argument for the directory containing the input images
    parser.add_argument('--output-path', type=str, required=True)
    # Argument for the output HDF5 file path
    parser.add_argument('--patch-size', type=int, default=31)
    # Argument for the patch size used for generating image patches (default is 31x31)
    parser.add_argument('--stride', type=int, default=21)
    # Argument for the stride used for generating image patches (default is 21)
    args = parser.parse_args()
    # Parse the arguments provided via the command line

    hr_patches = []  # List to store high-resolution image patches
    lr_patches = []  # List to store low-resolution image patches

    # Iterate over all images in the specified directory
    for i, image_path in enumerate(sorted(glob.glob('{}/*'.format(args.images_dir)))):
        # Scale 2
        hr = load_image(image_path)  # Load the high-resolution image
        hr = modcrop(hr, 2)  # Crop the image to be divisible by the scale factor of 2
        lr = generate_lr(hr, 2)  # Generate a low-resolution image from the high-resolution image

        # Generate patches from the high-resolution image
        for patch in generate_patch(hr, args.patch_size, args.stride):
            patch = image_to_array(patch)  # Convert the patch to a numpy array
            patch = np.expand_dims(normalize(rgb_to_y(patch.astype(np.float32), 'chw')), 0)
            # Convert RGB image to Y-channel (luminance) and normalize it
            hr_patches.append(patch)  # Add the processed patch to the list of high-resolution patches

        # Generate patches from the low-resolution image
        for patch in generate_patch(lr, args.patch_size, args.stride):
            patch = image_to_array(patch)  # Convert the patch to a numpy array
            patch = np.expand_dims(normalize(rgb_to_y(patch.astype(np.float32), 'chw')), 0)
            # Convert RGB image to Y-channel (luminance) and normalize it
            lr_patches.append(patch)  # Add the processed patch to the list of low-resolution patches

        # Scale 3
        hr = load_image(image_path)  # Load the high-resolution image again
        hr = modcrop(hr, 3)  # Crop the image to be divisible by the scale factor of 3
        lr = generate_lr(hr, 3)  # Generate a low-resolution image from the high-resolution image

        # Generate patches from the high-resolution image at scale 3
        for patch in generate_patch(hr, args.patch_size, args.stride):
            patch = image_to_array(patch)  # Convert the patch to a numpy array
            patch = np.expand_dims(normalize(rgb_to_y(patch.astype(np.float32), 'chw')), 0)
            # Convert RGB image to Y-channel (luminance) and normalize it
            hr_patches.append(patch)  # Add the processed patch to the list of high-resolution patches

        # Generate patches from the low-resolution image at scale 3
        for patch in generate_patch(lr, args.patch_size, args.stride):
            patch = image_to_array(patch)  # Convert the patch to a numpy array
            patch = np.expand_dims(normalize(rgb_to_y(patch.astype(np.float32), 'chw')), 0)
            # Convert RGB image to Y-channel (luminance) and normalize it
            lr_patches.append(patch)  # Add the processed patch to the list of low-resolution patches

        # Scale 4
        hr = load_image(image_path)  # Load the high-resolution image again
        hr = modcrop(hr, 4)  # Crop the image to be divisible by the scale factor of 4
        lr = generate_lr(hr, 4)  # Generate a low-resolution image from the high-resolution image

        # Generate patches from the high-resolution image at scale 4
        for patch in generate_patch(hr, args.patch_size, args.stride):
            patch = image_to_array(patch)  # Convert the patch to a numpy array
            patch = np.expand_dims(normalize(rgb_to_y(patch.astype(np.float32), 'chw')), 0)
            # Convert RGB image to Y-channel (luminance) and normalize it
            hr_patches.append(patch)  # Add the processed patch to the list of high-resolution patches

        # Generate patches from the low-resolution image at scale 4
        for patch in generate_patch(lr, args.patch_size, args.stride):
            patch = image_to_array(patch)  # Convert the patch to a numpy array
            patch = np.expand_dims(normalize(rgb_to_y(patch.astype(np.float32), 'chw')), 0)
            # Convert RGB image to Y-channel (luminance) and normalize it
            lr_patches.append(patch)  # Add the processed patch to the list of low-resolution patches

        # Print the progress of processing images and patches
        print('Images: {}, Patches: {}'.format(i + 1, len(hr_patches)))

        # Uncomment the following lines to limit the number of processed images
        # if i > 100:
        #     break

    # Create an HDF5 file for storing the patches
    h5_file = h5py.File(args.output_path, 'w')

    # Create datasets in the HDF5 file for storing high-resolution and low-resolution patches
    h5_file.create_dataset('hr', data=np.array(hr_patches))
    h5_file.create_dataset('lr', data=np.array(lr_patches))

    # Close the HDF5 file after saving all the patches
    h5_file.close()
