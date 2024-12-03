from PIL import Image, ImageDraw, ImageFont
import numpy as np
from matplotlib.colors import ListedColormap, LogNorm, Normalize, LinearSegmentedColormap
import matplotlib as mpl
import os


#https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.ListedColormap.html
#https://matplotlib.org/stable/users/explain/colors/colormap-manipulation.html#sphx-glr-users-explain-colors-colormap-manipulation-py
#https://scikit-image.org/docs/stable/api/skimage.color.html#skimage.color.hsv2rgb

def mandelbrot(z, c, limit):
    i = 0
    while i<limit:
        if abs(z) >= 2:
            # We know the function diverges.
            #
            return i

        z = z ** 2 + c

        i += 1

    # Did not diverge in time.
    #
    return np.nan

def make_img(fname, cmap, log, show=False):

    iters = np.load(f'Arrays/{fname}.npy')

    if log:
        iters = LogNorm()(iters)
    else:
        iters = Normalize()(iters)

    # Make it pretty.
    #
    colors = cmap(iters)*255

    img = Image.fromarray(np.uint8(colors))
    img.save(f'Images/{fname}_{cmap.name}{"_log"*log}.png')

    if show:
        img.show()

def make_location(x, y, rnge, name, cmap, psize=8, color=(255, 255, 255), show=False):
    iters = np.load(f'Arrays/00_p1000_l1000.npy')
    iters = LogNorm()(iters)
    colors = cmap(iters)*255
    img = Image.fromarray(np.uint8(colors))

    image_size = iters.shape[0]
    
    # Calculate the center of the image
    center = image_size // 2
    
    # Scale factor: half the image size corresponds to 2 units
    scale = (image_size / 2) / 2
    
    # Convert Cartesian coordinates to image coordinates
    # Scaling and centering the point
    image_x = center + x * scale
    image_y = center - y * scale  # Flip y to match image coordinate system
    
    draw = ImageDraw.Draw(img)

    # Draw horizontal line
    draw.line([
        0, image_y,
        image_size, image_y
    ], fill=color, width=5)
    
    # Draw vertical line
    draw.line([
        image_x, 0,
        image_x, image_size
    ], fill=color, width=5)

    # # Draw the point
    # draw.ellipse([
    #     image_x - psize // 2, 
    #     image_y - psize // 2, 
    #     image_x + psize // 2, 
    #     image_y + psize // 2
    # ], fill=color)

    # # font = ImageFont.load_default(24)
    # font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 96)
    # text = f'({x},{y})\n±{rnge}'

    # # Calculate text size
    # text_bbox = draw.textbbox((0, 0), text, font=font)
    # text_width = text_bbox[2] - text_bbox[0]
    # text_height = text_bbox[3] - text_bbox[1]
    
    # # Calculate position (bottom right corner with margin)
    # text_x = image_size - text_width - 12
    # text_y = image_size - text_height - 12
    
    # # Draw the text
    # draw.text((text_x, text_y), text, fill=color, font=font)

    img.save(f'Images/{name}_location.png')
    if show:
        img.show()


def make_arr(x, y, range, pixels, limit, name, verbose=True, dtype=np.float64, overwrite=False):

    fname = f'{name}_p{pixels}_l{limit}'

    if os.path.isfile(f'Arrays/{fname}.npy') and not overwrite:
        print('Skipping array generation, already exists')
        return fname
    
    # Make a coordinate matrix based on the parameters.
    #
    x_coord = np.linspace(x-range/2., x+range/2., pixels, dtype=dtype)
    y_coord = np.linspace(y-range/2., y+range/2., pixels, dtype=dtype)
    re, im = np.meshgrid(x_coord, y_coord, copy=False)

    coord = re + im * 1j

    # Each element of coord is now an imaginary coordinate.
    # We can apply our mandelbrot function.
    #
    iters = np.array([
        mandelbrot(0., c, limit) for c in coord.flatten()
    ]).reshape(
        (pixels, pixels)
    )

    print(f'Iterations min:{np.nanmin(iters)} max:{np.nanmax(iters)} std:{np.nanstd(iters)}')

    np.save(f'Arrays/{fname}.npy', iters)

    return fname
    

def truncate_colormap(cmap, name, minval=0.0, maxval=1.0, n=100):
    new_cmap = LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n))
    )
    new_cmap.name = name
    return new_cmap

if __name__=='__main__':
    from datetime import datetime
    startTime = datetime.now()

    # Magma
    #
    cmap = mpl.colormaps['magma']

    # Christmas
    #
    # cmap = truncate_colormap(mpl.colormaps['brg'], 'Christmas', 0.45, 1.0)

    # x, y, rnge, pixels, limit, name = 0, 0, 4, 1000, 1000, '00'
    # x, y, rnge, pixels, limit, name = -1.768620774, 0.002428273, 9e-06, 2000, 1000, 'SpiralWeb'
    x, y, rnge, pixels, limit, name = -1.74876455, 0.0, 4.8e-05, 2000, 2000, 'Bloom'
    # x, y, rnge, pixels, limit, name = -0.7454265, 0.1130105, 7.7e-05, 2000, 4000, 'TwinSeahorse'
    # x, y, rnge, pixels, limit, name = -0.7454210, 0.1130490, 1.5e-05, 2000, 4000, 'SpiralsAllTheWayDown'

    print(name)
    fname = make_arr(x, y, rnge, pixels, limit, name)

    # BIG
    # fnme = make_arr(-0.7454265, 0.1130105, 7.7e-05, 24000, 4000, 'TwinSeahorse')

    # Precision issues
    # fname = make_arr(-0.744539860355905, 0.121723773894425, 1.5e-11, 1000, 2000, 'Test')
    # make_image(-0.7445398603559084, 0.12172377389442482, 1.199040866595169e-14, 250, 4000, cmap, True, 'Test4', dtype=np.longdouble)

    print(datetime.now() - startTime)

    make_img(fname, cmap, True)
    # make_img(fname, cmap, False)
    make_location(x, y, rnge, name, cmap, show=True)
    

