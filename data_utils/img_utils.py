"""
Image Processing Utilities for SceneCrafter

This module provides a comprehensive set of image processing utilities for 3D scene rendering,
including image loading/saving, color space conversions, video creation, visualization tools,
and 3D bounding box rendering.
"""

import os
import re
import torch
import imageio
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def process_image(file_name):
    """
    Load and process an image file, converting it to RGB numpy array format.
    
    This function handles PNG images specifically and converts them to RGB format
    for consistent processing across the pipeline.
    
    Args:
        file_name (str): Path to the image file to process
        
    Returns:
        np.ndarray: RGB image as numpy array with shape (H, W, 3)
        
    Raises:
        FileNotFoundError: If the specified file does not exist
        ValueError: If the file format is not supported
        
    Example:
        >>> img = process_image("path/to/image.png")
        >>> print(img.shape)  # (H, W, 3)
    """
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"Image file not found: {file_name}")
        
    if file_name.endswith(".png"):
        image = Image.open(file_name)
    else:
        raise ValueError(f"Unsupported image format: {file_name}")
        
    return np.array(image.convert("RGB"))


def create_video_from_images(image_folder, video_name, fps=30):
    """
    Create a video from PNG images in a specified folder.
    
    This function searches for PNG files containing 'concat' in their filename,
    sorts them numerically based on extracted digits, and creates a video with
    specified frame rate.
    
    Args:
        image_folder (str): Path to folder containing PNG images
        video_name (str): Output video file name (e.g., "output.mp4")
        fps (int, optional): Frames per second for the output video. Defaults to 30.
        
    Returns:
        bool: True if video creation successful, False otherwise (e.g., insufficient images)
        
    Example:
        >>> success = create_video_from_images("frames/", "output.mp4", fps=30)
        >>> if success:
        ...     print("Video created successfully")
    """
    # Find all PNG files containing 'concat' in filename
    image_files = [img for img in os.listdir(image_folder) 
                   if img.endswith(".png") and 'concat' in img]
    
    # Require at least 2 seconds of frames
    if len(image_files) < fps * 2:
        print(f"Insufficient images: found {len(image_files)}, need at least {fps * 2}")
        return False
    
    # Sort images numerically based on extracted digits
    image_files = sorted(image_files, key=lambda x: int(re.findall(r'\d+', x)[0]))
    
    # Process all images
    images = [process_image(os.path.join(image_folder, image)) for image in image_files]
    
    # Create video using imageio
    with imageio.get_writer(video_name, fps=fps) as video:
        for image in images:
            video.append_data(image)
    
    return True


def save_img_torch(x, name='out.png'):
    """
    Save a PyTorch tensor as an image file.
    
    Handles tensor dimension ordering and normalization automatically.
    Supports grayscale and RGB images.
    
    Args:
        x (torch.Tensor): Input tensor with shape [C, H, W] or [1, H, W] or [H, W]
        name (str, optional): Output filename. Defaults to 'out.png'.
        
    Example:
        >>> tensor = torch.randn(3, 256, 256)
        >>> save_img_torch(tensor, "output.png")
    """
    # Clamp values to [0, 1] range and convert to numpy
    x = (x.clamp(0., 1.).detach().cpu().numpy() * 255).astype(np.uint8)
    
    # Handle channel dimension
    if x.shape[0] == 1 or x.shape[0] == 3:
        x = x.transpose(1, 2, 0)  # Convert from [C, H, W] to [H, W, C]
    
    # Handle single channel
    if x.shape[-1] == 1:
        x = x.squeeze(-1)
    
    # Save image
    img = Image.fromarray(x)
    img.save(name)


def save_img_numpy(x, name='out.png'):
    """
    Save a numpy array as an image file.
    
    Handles array dimension ordering and normalization automatically.
    Supports grayscale and RGB images.
    
    Args:
        x (np.ndarray): Input array with shape [C, H, W] or [H, W, C] or [H, W]
        name (str, optional): Output filename. Defaults to 'out.png'.
        
    Example:
        >>> array = np.random.rand(3, 256, 256)
        >>> save_img_numpy(array, "output.png")
    """
    # Clip values to [0, 1] range and convert to uint8
    x = (x.clip(0., 1.) * 255).astype(np.uint8)
    
    # Handle channel dimension
    if x.shape[0] == 1 or x.shape[0] == 3:
        x = x.transpose(1, 2, 0)  # Convert from [C, H, W] to [H, W, C]
    
    # Handle single channel
    if x.shape[-1] == 1:
        x = x.squeeze(-1)
    
    # Save image
    img = Image.fromarray(x)
    img.save(name)


def unnormalize_img(img, mean, std):
    """
    Unnormalize an image tensor using provided mean and standard deviation.
    
    This function reverses the normalization process by multiplying by std
    and adding mean, then normalizes to [0, 1] range.
    
    Args:
        img (torch.Tensor): Normalized image tensor with shape [3, H, W]
        mean (list or tuple): Mean values for each channel [R, G, B]
        std (list or tuple): Standard deviation values for each channel [R, G, B]
        
    Returns:
        torch.Tensor: Unnormalized image tensor in [0, 1] range
        
    Example:
        >>> img_norm = torch.randn(3, 256, 256)
        >>> img_unnorm = unnormalize_img(img_norm, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    """
    img = img.detach().cpu().clone()
    
    # Reverse normalization: multiply by std and add mean
    img *= torch.tensor(std).view(3, 1, 1)
    img += torch.tensor(mean).view(3, 1, 1)
    
    # Normalize to [0, 1] range
    min_v = torch.min(img)
    img = (img - min_v) / (torch.max(img) - min_v)
    
    return img


def bgr_to_rgb(img):
    """
    Convert BGR image format to RGB format.
    
    Supports both 3-channel color images and single-channel grayscale images.
    
    Args:
        img (np.ndarray): Input image array with shape [H, W, 3] for BGR or [H, W, 1] for grayscale
        
    Returns:
        np.ndarray: Converted RGB image
        
    Raises:
        NotImplementedError: For unsupported channel configurations
        
    Example:
        >>> bgr_img = cv2.imread("image.jpg")
        >>> rgb_img = bgr_to_rgb(bgr_img)
    """
    if img.shape[-1] == 3:
        # BGR to RGB conversion
        return img[..., [2, 1, 0]]
    elif img.shape[-1] == 1:
        # Grayscale to RGB (replicate single channel)
        return np.repeat(img, 3, axis=-1)
    else:
        raise NotImplementedError(f"Unsupported channel count: {img.shape[-1]}")


def rgb_to_bgr(img):
    """
    Convert RGB image format to BGR format.
    
    Supports both 3-channel color images and single-channel grayscale images.
    
    Args:
        img (np.ndarray): Input image array with shape [H, W, 3] for RGB or [H, W, 1] for grayscale
        
    Returns:
        np.ndarray: Converted BGR image
        
    Raises:
        NotImplementedError: For unsupported channel configurations
        
    Example:
        >>> rgb_img = plt.imread("image.png")
        >>> bgr_img = rgb_to_bgr(rgb_img)
    """
    if img.shape[-1] == 3:
        # RGB to BGR conversion
        return img[..., [2, 1, 0]]
    elif img.shape[-1] == 1:
        # Grayscale to BGR (replicate single channel)
        return np.repeat(img, 3, axis=-1)
    else:
        raise NotImplementedError(f"Unsupported channel count: {img.shape[-1]}")


# Utility lambda function to convert float [0,1] to uint8 [0,255]
to8b = lambda x: (255*np.clip(x, 0, 1)).astype(np.uint8)


def horizon_concate(inp0, inp1):
    """
    Concatenate two images horizontally (side by side).
    
    Handles both color and grayscale images, padding with zeros if dimensions differ.
    
    Args:
        inp0 (np.ndarray): First input image
        inp1 (np.ndarray): Second input image
        
    Returns:
        np.ndarray: Horizontally concatenated image
        
    Example:
        >>> img1 = np.random.rand(100, 100, 3)
        >>> img2 = np.random.rand(100, 150, 3)
        >>> combined = horizon_concate(img1, img2)  # Shape: (100, 250, 3)
    """
    h0, w0 = inp0.shape[:2]
    h1, w1 = inp1.shape[:2]
    
    if inp0.ndim == 3:
        # Color image concatenation
        inp = np.zeros((max(h0, h1), w0 + w1, 3), dtype=inp0.dtype)
        inp[:h0, :w0, :] = inp0
        inp[:h1, w0:(w0 + w1), :] = inp1
    else:
        # Grayscale image concatenation
        inp = np.zeros((max(h0, h1), w0 + w1), dtype=inp0.dtype)
        inp[:h0, :w0] = inp0
        inp[:h1, w0:(w0 + w1)] = inp1
    
    return inp


def recover_shape(pixel_value, H, W, mask_at_box=None):
    """
    Recover the original spatial shape of pixel values from flattened format.
    
    Handles various input tensor/array shapes and applies masking if provided.
    
    Args:
        pixel_value (torch.Tensor or np.ndarray): Input pixel values
        H (int): Target height
        W (int): Target width
        mask_at_box (np.ndarray, optional): Boolean mask for valid pixels. Defaults to None.
        
    Returns:
        np.ndarray: Reshaped pixel values with shape [H, W, C] or [H, W]
        
    Example:
        >>> flat_pixels = torch.randn(1000, 3)
        >>> img = recover_shape(flat_pixels, 32, 32)  # Shape: (32, 32, 3)
    """
    from lib.config import cfg
    
    # Handle batch dimension
    if pixel_value.shape[0] == 1:
        pixel_value = pixel_value[0]
    
    # Convert torch tensor to numpy if needed
    if len(pixel_value.shape) == 3:
        if pixel_value.shape[0] in [1, 3]:
            # Convert from [C, H, W] to [H, W, C]
            pixel_value = pixel_value.permute(1, 2, 0).detach().cpu().numpy()
        else:
            pixel_value = pixel_value.detach().cpu().numpy()
    elif len(pixel_value.shape) == 2:
        pixel_value = pixel_value.detach().cpu().numpy()
        pixel_value = pixel_value.reshape(H, W, -1)
    elif len(pixel_value.shape) == 1:
        pixel_value = pixel_value.detach().cpu().numpy()[..., None]
        pixel_value = pixel_value.reshape(H, W, -1)
    else:
        raise ValueError(f'Invalid shape of pixel_value: {pixel_value.shape}')
    
    # Apply mask if provided
    if mask_at_box is not None:
        mask_at_box = mask_at_box.reshape(H, W)
        
        # Create background based on configuration
        if cfg.white_bkgd:
            full_pixel_value = np.ones((H, W, 3)).astype(np.float32)
        else:
            full_pixel_value = np.zeros((H, W, 3)).astype(np.float32)
        
        # Fill valid pixels
        full_pixel_value[mask_at_box] = pixel_value
        return full_pixel_value
    else:
        return pixel_value


def vertical_concate(inp0, inp1):
    """
    Concatenate two images vertically (one above another).
    
    Handles both color and grayscale images, padding with zeros if dimensions differ.
    
    Args:
        inp0 (np.ndarray): First input image
        inp1 (np.ndarray): Second input image
        
    Returns:
        np.ndarray: Vertically concatenated image
        
    Example:
        >>> img1 = np.random.rand(100, 100, 3)
        >>> img2 = np.random.rand(120, 100, 3)
        >>> combined = vertical_concate(img1, img2)  # Shape: (220, 100, 3)
    """
    h0, w0 = inp0.shape[:2]
    h1, w1 = inp1.shape[:2]
    
    if inp0.ndim == 3:
        # Color image concatenation
        inp = np.zeros((h0 + h1, max(w0, w1), 3), dtype=inp0.dtype)
        inp[:h0, :w0, :] = inp0
        inp[h0:(h0 + h1), :w1, :] = inp1
    else:
        # Grayscale image concatenation
        inp = np.zeros((h0 + h1, max(w0, w1)), dtype=inp0.dtype)
        inp[:h0, :w0] = inp0
        inp[h0:(h0 + h1), :w1] = inp1
    
    return inp


def save_image(pred, gt, save_dir, save_name, concat=False):
    """
    Save prediction and ground truth images with flexible saving options.
    
    Can save images separately or concatenated horizontally.
    
    Args:
        pred (np.ndarray): Predicted image
        gt (np.ndarray, optional): Ground truth image. If None, only saves prediction
        save_dir (str): Directory to save images
        save_name (str): Base filename for saved images
        concat (bool, optional): Whether to concatenate pred and gt horizontally. Defaults to False.
        
    Example:
        >>> pred_img = np.random.rand(256, 256, 3)
        >>> gt_img = np.random.rand(256, 256, 3)
        >>> save_image(pred_img, gt_img, "results/", "test", concat=True)
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    if gt is None:
        # Save only prediction
        cv2.imwrite(f'{save_dir}/{save_name}.png', to8b(rgb_to_bgr(pred)))
    else:
        if concat:
            # Save concatenated image
            img = horizon_concate(pred, gt)
            cv2.imwrite(f'{save_dir}/{save_name}.png', to8b(rgb_to_bgr(img)))
        else:
            # Save separate images
            cv2.imwrite(f'{save_dir}/{save_name}.png', to8b(rgb_to_bgr(pred)))
            cv2.imwrite(f'{save_dir}/{save_name}_gt.png', to8b(rgb_to_bgr(gt)))


def transparent_cmap(cmap):
    """
    Create a transparent version of a matplotlib colormap.
    
    Sets alpha values to 0.3 for all colors in the colormap.
    
    Args:
        cmap: Matplotlib colormap object
        
    Returns:
        Modified colormap with transparency
    """
    mycmap = cmap
    mycmap._init()
    mycmap._lut[:, -1] = 0.3  # Set alpha to 0.3
    return mycmap


# Create transparent jet colormap for visualization
cmap = transparent_cmap(plt.get_cmap('jet'))


def set_grid(ax, h, w, interval=8):
    """
    Configure grid lines for matplotlib axis.
    
    Sets up grid lines at specified intervals and removes tick labels.
    
    Args:
        ax: Matplotlib axis object
        h (int): Image height
        w (int): Image width
        interval (int, optional): Grid line spacing. Defaults to 8.
    """
    ax.set_xticks(np.arange(0, w, interval))
    ax.set_yticks(np.arange(0, h, interval))
    ax.grid()
    ax.set_yticklabels([])
    ax.set_xticklabels([])


# Comprehensive color palette for visualization
color_list = np.array([
    0.000, 0.447, 0.741,
    0.850, 0.325, 0.098,
    0.929, 0.694, 0.125,
    0.494, 0.184, 0.556,
    0.466, 0.674, 0.188,
    0.301, 0.745, 0.933,
    0.635, 0.078, 0.184,
    0.300, 0.300, 0.300,
    0.600, 0.600, 0.600,
    1.000, 0.000, 0.000,
    1.000, 0.500, 0.000,
    0.749, 0.749, 0.000,
    0.000, 1.000, 0.000,
    0.000, 0.000, 1.000,
    0.667, 0.000, 1.000,
    0.333, 0.333, 0.000,
    0.333, 0.667, 0.000,
    0.333, 1.000, 0.000,
    0.667, 0.333, 0.000,
    0.667, 0.667, 0.000,
    0.667, 1.000, 0.000,
    1.000, 0.333, 0.000,
    1.000, 0.667, 0.000,
    1.000, 1.000, 0.000,
    0.000, 0.333, 0.500,
    0.000, 0.667, 0.500,
    0.000, 1.000, 0.500,
    0.333, 0.000, 0.500,
    0.333, 0.333, 0.500,
    0.333, 0.667, 0.500,
    0.333, 1.000, 0.500,
    0.667, 0.000, 0.500,
    0.667, 0.333, 0.500,
    0.667, 0.667, 0.500,
    0.667, 1.000, 0.500,
    1.000, 0.000, 0.500,
    1.000, 0.333, 0.500,
    1.000, 0.667, 0.500,
    1.000, 1.000, 0.500,
    0.000, 0.333, 1.000,
    0.000, 0.667, 1.000,
    0.000, 1.000, 1.000,
    0.333, 0.000, 1.000,
    0.333, 0.333, 1.000,
    0.333, 0.667, 1.000,
    0.333, 1.000, 1.000,
    0.667, 0.000, 1.000,
    0.667, 0.333, 1.000,
    0.667, 0.667, 1.000,
    0.667, 1.000, 1.000,
    1.000, 0.000, 1.000,
    1.000, 0.333, 1.000,
    1.000, 0.667, 1.000,
    0.167, 0.000, 0.000,
    0.333, 0.000, 0.000,
    0.500, 0.000, 0.000,
    0.667, 0.000, 0.000,
    0.833, 0.000, 0.000,
    1.000, 0.000, 0.000,
    0.000, 0.167, 0.000,
    0.000, 0.333, 0.000,
    0.000, 0.500, 0.000,
    0.000, 0.667, 0.000,
    0.000, 0.833, 0.000,
    0.000, 1.000, 0.000,
    0.000, 0.000, 0.167,
    0.000, 0.000, 0.333,
    0.000, 0.000, 0.500,
    0.000, 0.000, 0.667,
    0.000, 0.000, 0.833,
    0.000, 0.000, 1.000,
    0.000, 0.000, 0.000,
    0.143, 0.143, 0.143,
    0.286, 0.286, 0.286,
    0.429, 0.429, 0.429,
    0.571, 0.571, 0.571,
    0.714, 0.714, 0.714,
    0.857, 0.857, 0.857,
    1.000, 1.000, 1.000,
    0.50, 0.5, 0
]).astype(np.float32)

# Reshape color list for visualization purposes
colors = color_list.reshape((-1, 3)) * 255
colors = np.array(colors, dtype=np.uint8).reshape(len(colors), 1, 1, 3)


def visualize_depth_numpy(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    Visualize depth map as a colored image using jet colormap.
    
    Handles NaN values and provides depth range information.
    
    Args:
        depth (np.ndarray): Depth map with shape (H, W)
        minmax (tuple, optional): (min, max) depth values for normalization. 
                                 If None, uses min/max from positive depths.
        cmap (int, optional): OpenCV colormap. Defaults to cv2.COLORMAP_JET.
        
    Returns:
        tuple: (colored_depth_image, depth_range) where depth_range is [min_depth, max_depth]
        
    Example:
        >>> depth_map = np.random.rand(480, 640)
        >>> vis_depth, depth_range = visualize_depth_numpy(depth_map)
        >>> print(f"Depth range: {depth_range}")
    """
    # Replace NaN values with 0
    x = np.nan_to_num(depth)
    
    # Determine depth range for normalization
    if minmax is None:
        mi = np.min(x[x > 0])  # Minimum positive depth
        ma = np.max(x)
    else:
        mi, ma = minmax
    
    # Normalize to [0, 1] range
    x = (x - mi) / (ma - mi + 1e-8)
    
    # Convert to uint8 and apply colormap
    x = (255 * x).astype(np.uint8)
    x_ = cv2.applyColorMap(x, cmap)
    
    return x_, [mi, ma]


def normalize_img(img):
    """
    Normalize image values to [0, 255] range.
    
    Performs min-max normalization and converts to uint8.
    
    Args:
        img (np.ndarray): Input image
        
    Returns:
        np.ndarray: Normalized uint8 image
        
    Example:
        >>> img = np.random.rand(100, 100) * 100
        >>> norm_img = normalize_img(img)
    """
    min_val, max_val = img.min(), img.max()
    img = (img - min_val) / (max_val - min_val)
    img = (img * 255).astype(np.uint8)
    return img


def linear_to_srgb(linear, eps=None):
    """
    Convert linear RGB values to sRGB color space.
    
    Implements the standard sRGB gamma correction formula.
    Reference: https://en.wikipedia.org/wiki/SRGB
    
    Args:
        linear (torch.Tensor): Linear RGB values in [0, 1] range
        eps (float, optional): Epsilon value for numerical stability. 
                              Defaults to torch.finfo(linear.dtype).eps
        
    Returns:
        torch.Tensor: sRGB values in [0, 1] range
        
    Example:
        >>> linear = torch.rand(3, 256, 256)
        >>> srgb = linear_to_srgb(linear)
    """
    if eps is None:
        eps = torch.finfo(linear.dtype).eps
    
    # sRGB conversion formula
    srgb0 = 323 / 25 * linear
    srgb1 = (211 * linear.clamp_min(eps) ** (5 / 12) - 11) / 200
    
    # Apply piecewise conversion
    return torch.where(linear <= 0.0031308, srgb0, srgb1)


def srgb_to_linear(srgb, eps=None):
    """
    Convert sRGB values to linear RGB color space.
    
    Implements the inverse sRGB gamma correction formula.
    Reference: https://en.wikipedia.org/wiki/SRGB
    
    Args:
        srgb (np.ndarray): sRGB values in [0, 1] range
        eps (float, optional): Epsilon value for numerical stability. 
                              Defaults to np.finfo(srgb.dtype).eps
        
    Returns:
        np.ndarray: Linear RGB values in [0, 1] range
        
    Example:
        >>> srgb = np.random.rand(3, 256, 256)
        >>> linear = srgb_to_linear(srgb)
    """
    if eps is None:
        eps = np.finfo(srgb.dtype).eps
    
    # Inverse sRGB conversion formula
    linear0 = 25 / 323 * srgb
    linear1 = np.maximum(eps, ((200 * srgb + 11) / (211))) ** (12 / 5)
    
    # Apply piecewise conversion
    return np.where(srgb <= 0.04045, linear0, linear1)


def draw_3d_box_on_img(vertices, img, color=(255, 128, 128), thickness=1):
    """
    Draw a 3D bounding box on a 2D image.
    
    Draws the 12 edges of a 3D box and adds diagonal lines on the front face
    to indicate orientation.
    
    Args:
        vertices (dict or array): 8 vertices of the 3D box in 2D image coordinates
        img (np.ndarray): Input image to draw on (will be modified in-place)
        color (tuple, optional): BGR color tuple. Defaults to (255, 128, 128) - light blue
        thickness (int, optional): Line thickness. Defaults to 1
        
    Example:
        >>> img = np.zeros((480, 640, 3), dtype=np.uint8)
        >>> vertices = {(i,j,k): (x,y) for ...}  # 8 vertices mapping
        >>> draw_3d_box_on_img(vertices, img)
    """
    # Draw the 12 edges of the 3D bounding box
    for k in [0, 1]:
        for l in [0, 1]:
            # Edges along x-axis
            cv2.line(img, tuple(vertices[(0, k, l)]), tuple(vertices[(1, k, l)]), color, thickness)
            # Edges along y-axis
            cv2.line(img, tuple(vertices[(k, 0, l)]), tuple(vertices[(k, 1, l)]), color, thickness)
            # Edges along z-axis
            cv2.line(img, tuple(vertices[(k, l, 0)]), tuple(vertices[(k, l, 1)]), color, thickness)
    
    # Draw diagonal lines on front face to indicate orientation
    for idx1, idx2 in [((1, 0, 0), (1, 1, 1)), ((1, 1, 0), (1, 0, 1))]:
        cv2.line(img, tuple(vertices[idx1]), tuple(vertices[idx2]), color, thickness)