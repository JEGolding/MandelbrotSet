import numpy as np
from matplotlib.colors import LogNorm, Normalize
import matplotlib as mpl
from PIL import Image
import math

def save_large_array_as_image(array, output_path, tile_size=1024, format='PNG'):
    """
    Save a large numpy array as an image by processing it in tiles to avoid memory issues.
    
    Parameters:
    -----------
    array : numpy.ndarray
        The input array to save as an image. Should be 2D or 3D (for RGB/RGBA).
    output_path : str
        The path where the output image should be saved.
    tile_size : int
        The size of tiles to process at once. Adjust based on available memory.
    format : str
        The output image format (e.g., 'PNG', 'TIFF', 'JPEG').
    """
    # Ensure array values are in valid range for images
    if array.dtype != np.uint8:
        # Normalize to 0-255 if not already in that range
        array = ((array - array.min()) * (255.0 / (array.max() - array.min()))).astype(np.uint8)
    
    height, width = array.shape[:2]
    
    # Create an empty image with the full size
    if len(array.shape) == 2:
        mode = 'L'  # Grayscale
    elif array.shape[2] == 3:
        mode = 'RGB'
    elif array.shape[2] == 4:
        mode = 'RGBA'
    else:
        raise ValueError("Unsupported array shape")
        
    # Create the output image
    output_image = Image.new(mode, (width, height))
    
    # Process the array in tiles
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            # Calculate the tile boundaries
            y_end = min(y + tile_size, height)
            x_end = min(x + tile_size, width)
            
            # Extract the tile
            tile = array[y:y_end, x:x_end]
            
            # Convert tile to PIL Image
            tile_image = Image.fromarray(tile)
            
            # Paste the tile into the output image
            output_image.paste(tile_image, (x, y))
            
            # Clear memory
            del tile
            del tile_image
    
    # Save the final image
    output_image.save(output_path, format=format, optimize=True)
    
    # Clear memory
    del output_image

def save_large_array_as_tiff(array, output_path, compression='lzw'):
    """
    Save a large numpy array as a TIFF file with compression.
    This method is more memory-efficient for TIFF format specifically.
    
    Parameters:
    -----------
    array : numpy.ndarray
        The input array to save as an image.
    output_path : str
        The path where the output TIFF should be saved.
    compression : str
        The compression method to use ('lzw', 'zip', etc.)
    """    
    # Ensure array values are in valid range
    if array.dtype != np.uint8:
        array = ((array - array.min()) * (255.0 / (array.max() - array.min()))).astype(np.uint8)
    
    # Convert to PIL Image
    if len(array.shape) == 2:
        mode = 'L'
    elif array.shape[2] == 3:
        mode = 'RGB'
    elif array.shape[2] == 4:
        mode = 'RGBA'
    else:
        raise ValueError("Unsupported array shape")
    
    image = Image.fromarray(array)
    image.save(output_path, format='TIFF', compression=compression)

if __name__=='__main__':
    # c_name = 'magma'
    c_name = 'RdYlGn'
    cmap = mpl.colormaps[c_name]

    name = 'Test5'

    arr = np.load(f'./Arrays/{name}.npy')
    vmin, vmax = np.nanquantile(arr, [0, 0.90])
    print(vmin, vmax)

    # arr = Normalize(vmin=np.nanmin(arr), vmax=np.nanmax(arr), clip=True)(arr)
    arr = LogNorm(vmin=np.nanmin(arr), vmax=np.nanmax(arr), clip=True)(arr)

    arr[np.isnan(arr)] = 1
    arr = np.uint8(cmap(arr)*255)
    # print(arr)

    # For general image formats (PNG, JPEG, etc.)
    save_large_array_as_image(arr, f'Images/{name}_{c_name}.png', tile_size=1024)

    # For TIFF format specifically
    # save_large_array_as_tiff(arr, f'Images/{name}.tiff', compression='lzw')