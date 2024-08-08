import PIL.Image as pil_image  # Import PIL for image handling
import numpy as np  # Import numpy for numerical operations
import torch  # Import torch for tensor operations


def load_image(path):
    # Load an image from the given path and convert it to RGB format
    return pil_image.open(path).convert('RGB')


def generate_lr(image, scale):
    # Generate a low-resolution image by downscaling and then upscaling the original image
    # Downscale the image by the given scale factor
    image = image.resize((image.width // scale, image.height // scale), resample=pil_image.BICUBIC)
    # Upscale the image back to the original size
    image = image.resize((image.width * scale, image.height * scale), resample=pil_image.BICUBIC)
    return image


def modcrop(image, modulo):
    # Crop the image so that its dimensions are divisible by the given modulo
    w = image.width - image.width % modulo  # Calculate the new width
    h = image.height - image.height % modulo  # Calculate the new height
    return image.crop((0, 0, w, h))  # Crop the image and return it


def generate_patch(image, patch_size, stride):
    # Generate patches of a given size from the image with a specified stride
    for i in range(0, image.height - patch_size + 1, stride):  # Iterate over the height
        for j in range(0, image.width - patch_size + 1, stride):  # Iterate over the width
            # Yield a patch cropped from the image
            yield image.crop((j, i, j + patch_size, i + patch_size))


def image_to_array(image):
    # Convert an image to a numpy array and transpose it to match the (C, H, W) format
    return np.array(image).transpose((2, 0, 1))


def normalize(x):
    # Normalize the pixel values to a range of 0 to 1
    return x / 255.0


def denormalize(x):
    # Convert the pixel values back to a range of 0 to 255
    if type(x) == torch.Tensor:
        return (x * 255.0).clamp(0.0, 255.0)  # Ensure values are within the range [0, 255]
    elif type(x) == np.ndarray:
        return (x * 255.0).clip(0.0, 255.0)  # Clip values to ensure they stay within [0, 255]
    else:
        # Raise an exception if the input is neither a tensor nor a numpy array
        raise Exception('The denormalize function supports torch.Tensor or np.ndarray types.', type(x))


def rgb_to_y(img, dim_order='hwc'):
    # Convert an RGB image to its Y (luminance) channel
    if dim_order == 'hwc':
        # Handle the image if it has height, width, and channel ordering
        return 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.
    else:
        # Handle the image if it has channel, height, and width ordering
        return 16. + (64.738 * img[0] + 129.057 * img[1] + 25.064 * img[2]) / 256.


def PSNR(a, b, max=255.0, shave_border=0):
    # Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images
    assert type(a) == type(b)  # Ensure both images are of the same type
    assert (type(a) == torch.Tensor) or (type(a) == np.ndarray)  # Ensure they are either tensors or numpy arrays

    # Optionally shave the border of the images before comparison
    a = a[shave_border:a.shape[0]-shave_border, shave_border:a.shape[1]-shave_border]
    b = b[shave_border:b.shape[0]-shave_border, shave_border:b.shape[1]-shave_border]

    if type(a) == torch.Tensor:
        # Calculate PSNR for torch tensors
        return 10. * ((max ** 2) / ((a - b) ** 2).mean()).log10()
    elif type(a) == np.ndarray:
        # Calculate PSNR for numpy arrays
        return 10. * np.log10((max ** 2) / np.mean(((a - b) ** 2)))
    else:
        # Raise an exception if the input is neither a tensor nor a numpy array
        raise Exception('The PSNR function supports torch.Tensor or np.ndarray types.', type(a))


def load_weights(model, path):
    # Load weights from a file and assign them to the model
    state_dict = model.state_dict()  # Get the model's current state dictionary
    # Load the saved weights and map them to the model's device
    for n, p in torch.load(path, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)  # Copy the parameter if it exists in the model
        else:
            raise KeyError(n)  # Raise an error if the parameter doesn't exist in the model
    return model  # Return the model with loaded weights


class AverageMeter(object):
    # A utility class to keep track of average values (e.g., loss, accuracy)
    def __init__(self):
        self.reset()  # Initialize the meter by resetting all values

    def reset(self):
        # Reset the meter's values to their initial states
        self.val = 0  # Current value
        self.avg = 0  # Average value
        self.sum = 0  # Sum of all values
        self.count = 0  # Count of values

    def update(self, val, n=1):
        # Update the meter with a new value and optionally a weight (n)
        self.val = val  # Update the current value
        self.sum += val * n  # Add the weighted value to the sum
        self.count += n  # Increment the count by the weight
        self.avg = self.sum / self.count  # Calculate the new average
