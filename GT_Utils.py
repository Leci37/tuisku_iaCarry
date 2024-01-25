# Importing Required Modules
import cv2
from rembg import remove
from PIL import Image
import numpy as np



def remove_bg_from_png(input_path, output_path):
    # Processing the image
    input = Image.open(input_path)
    # Removing the background from the given Image
    output = remove(input)
    # Saving the image in the given path
    output.save(output_path)

def remove_bg_from_png_optimice(input_path, output_path):
    # Processing the image
    input = Image.open(input_path)
    # Removing the background from the given Image
    output = remove(input, alpha_matting =True,alpha_matting_foreground_threshold= 248,alpha_matting_background_threshold = 252 ,alpha_matting_erode_size =0 )
    # Saving the image in the given path
    output.save(output_path)


def reduce_size_for_transparency_size_png(path_witout_bg, path_exit = None):
    # Read input image, and convert to NumPy array.
    img = np.array(Image.open(path_witout_bg))  # img is 1080 rows by 1920 cols and 4 color channels, the 4'th channel is alpha.
    # Find indices of non-transparent pixels (indices where alpha channel value is above zero).
    idx = np.where(img[:, :, 3] > 0)
    # Get minimum and maximum index in both axes (top left corner and bottom right corner)
    x0, y0, x1, y1 = idx[1].min(), idx[0].min(), idx[1].max(), idx[0].max()
    # Crop rectangle and convert to Image
    out = Image.fromarray(img[y0:y1 + 1, x0:x1 + 1, :])
    # Save the result (RGBA color format).
    if path_exit  is not  None:
        out.save(path_exit)
    return img.shape[:2],out.size , x0, y0, x1, y1

def reduce_size_for_transparency_size(path_witout_bg, path_exit):
    # Read input image, and convert to NumPy array.
    img = Image.open(path_witout_bg)
    img = img.convert("RGBA")
    img = np.array(img) # img is 1080 rows by 1920 cols and 4 color channels, the 4'th channel is alpha.
    # Find indices of non-transparent pixels (indices where alpha channel value is above zero).
    idx = np.where(img[:, :, 3] > 0)
    # np.where(img[:, :, 3] > 0)


    # aux = np.where(img[:, :, 3] > 0, 0 , img[:, :, 0:3] )
    # out = Image.fromarray(aux)
    # out.save("test.png")
    # import cv2
    # image_np = cv2.imread("test.png")

    # Get minimum and maximum index in both axes (top left corner and bottom right corner)
    x0, y0, x1, y1 = idx[1].min(), idx[0].min(), idx[1].max(), idx[0].max()
    # Crop rectangle and convert to Image
    out = Image.fromarray(img[y0:y1 + 1, x0:x1 + 1, :]  )
    out = out.convert('RGB')
    # Save the result (RGBA color format).
    out.save(path_exit)
    # out.save("test.png")
    # import cv2
    # img = cv2.imread("test.png", cv2.IMREAD_UNCHANGED)
    # img[np.where(np.all(img[..., :3] == 255, -1))] = 0
    # cv2.imwrite("transparent.png", img)
    # image_np = cv2.imread("test.png")
    im = Image.open(path_witout_bg)
    bg = Image.new("RGB", im.size, (0, 0, 0))
    bg.paste(im, im)
    bg.save(path_exit)
    return img.shape[:2],out.size , x0, y0, x1, y1

# https://medium.com/@alexppppp/adding-objects-to-image-in-python-133f165b9a01
def add_img_noBG_from_jpeg(background, img, x, y):
    '''
    Arguments:
    background - background image in CV2 RGB format
    img - image of object in CV2 RGB format
    mask - mask of object in CV2 RGB format
    x, y - coordinates of the center of the object image
    0 < x < width of background
    0 < y < height of background

    Function returns background with added object in CV2 RGB format

    CV2 RGB format is a numpy array with dimensions width x height x 3
    '''

    # NECESARIO PARA EL  METODO remove_when_traparency_is_zero_png(image_np_png): para img
    # para ambas  image_fg = cv2.cvtColor(image_fg, cv2.COLOR_BGR2RGB)
    bg = background.copy()

    h_bg, w_bg = bg.shape[0], bg.shape[1]

    h, w = img.shape[0], img.shape[1]

    # Calculating coordinates of the top left corner of the object image
    x = x  - int(w / 2)
    y = y  - int(h / 2)

    mask_boolean = img[:, :, 0] > 0
    if (mask_boolean == 1).all() :
        print("WARN la imagen de frontal (el producto) tiene toda la mascara a True (deberia no tener fondo) , puede pasar al venir en formato .png (.jepg es el bueno cv2.IMWRITE_JPEG_QUALITY )", mask_boolean.shape)

    mask_rgb_boolean = np.stack([mask_boolean, mask_boolean, mask_boolean], axis=2)

    if x >= 0 and y >= 0:

        h_part = h - max(0, y + h - h_bg)  # h_part - part of the image which overlaps background along y-axis
        w_part = w - max(0, x + w - w_bg)  # w_part - part of the image which overlaps background along x-axis

        bg[y:y + h_part, x:x + w_part, :] = bg[y:y + h_part, x:x + w_part, :] * ~mask_rgb_boolean[0:h_part, 0:w_part,
                                                                                 :] + (img * mask_rgb_boolean)[0:h_part,
                                                                                      0:w_part, :]

    elif x < 0 and y < 0:

        h_part = h + y
        w_part = w + x

        bg[0:0 + h_part, 0:0 + w_part, :] = bg[0:0 + h_part, 0:0 + w_part, :] * ~mask_rgb_boolean[h - h_part:h,
                                                                                 w - w_part:w, :] + (
                                                                                                                img * mask_rgb_boolean)[
                                                                                                    h - h_part:h,
                                                                                                    w - w_part:w, :]

    elif x < 0 and y >= 0:

        h_part = h - max(0, y + h - h_bg)
        w_part = w + x

        bg[y:y + h_part, 0:0 + w_part, :] = bg[y:y + h_part, 0:0 + w_part, :] * ~mask_rgb_boolean[0:h_part,
                                                                                 w - w_part:w, :] + (
                                                                                                                img * mask_rgb_boolean)[
                                                                                                    0:h_part,
                                                                                                    w - w_part:w, :]

    elif x >= 0 and y < 0:

        h_part = h + y
        w_part = w - max(0, x + w - w_bg)

        bg[0:0 + h_part, x:x + w_part, :] = bg[0:0 + h_part, x:x + w_part, :] * ~mask_rgb_boolean[h - h_part:h,
                                                                                 0:w_part, :] + (
                                                                                                            img * mask_rgb_boolean)[
                                                                                                h - h_part:h, 0:w_part,
                                                                                                :]
    return bg, x, y


#Para evitar que las x, y, w, h. Se salgan de las imagenes
def avoid_overflow_photo_x_y_w_h(pos_x, pos_y, w_1, h_1, img_background_shape):
    img_w, img_h = img_background_shape[:2]
    x = pos_x
    y = pos_y
    w = w_1
    h = h_1
    x_com, y_com, w_com, h_com = pos_x, pos_y, (x + w), (y + h)
    if x_com < 0:
        w = w + x
        x = 0
    if y_com < 0:
        h = h + y
        y = 0
    if w_com > img_w:
        w = img_w - x
    if h_com > img_h:
        h = img_h - y
    return x, y, w, h


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'



def get_scalated_00_x_y_w_h(row):
    img_width, img_height = Image.open(row['path']).size
    x, y, w, h = row[['box_pos_x', 'box_pos_y', 'box_width', 'box_height']]
    x = x / img_width
    y = y / img_height
    w = w / img_width
    h = h / img_height
    return  x, y, w, h

def get_scalated_00_xmin_ymin_xmax_ymax(row):
    img_width, img_height = Image.open(row['path']).size
    x, y, x_max, y_max = row[["x_min", "y_min", "x_max", "y_max"]]
    w = (x_max - x) / img_width
    h = (y_max - y) / img_height
    x = x / img_width
    y = y / img_height
    return  x, y, w, h


# EL png tiene 4 dimensiones RGB y tranparencia , si la transparencia es zero pone a zero los otros R G B
# Read with this to enter cv2.imread(path_out_put, cv2.IMREAD_UNCHANGED)
def remove_when_traparency_is_zero_png(image_np_png):
    img = image_np_png[:, :, 0:4]
    alpha = image_np_png[:, :, 3]
    img[alpha == 0] = (0, 0, 0, 0)
    return img

def remove_white_simple_png(image_np2):
    # convert to 4 channels (bgr with opaque alpha) COLOR_BGR2RGB
    image_np2 = cv2.cvtColor(image_np2, cv2.COLOR_BGR2BGRA) # Todo NECESARIO ANTES DE LLAMAR
    # replace white with transparent using Numpy
    new_im = image_np2.copy()
    new_im[np.where((image_np2 == [255, 255, 255, 255]).all(axis=2))] = [255, 255, 255, 0]

    # new_im = cv2.cvtColor(new_im, cv2.COLOR_BGRA2BGR)
    return new_im

# COLOR_BGR2RGB
def recover_transparency_from_jpg_to_png(img_np_3array, type_img=cv2.COLOR_BGR2RGBA):
    # First create the image with alpha channel
    rgba = cv2.cvtColor(img_np_3array, type_img)
    # Then assign the mask to the last channel of the image
    img_np_4array = rgba[:, :, 0:4]
    alpha_r = rgba[:, :, 0]
    alpha_g = rgba[:, :, 1]
    alpha_b = rgba[:, :, 2]
    img_np_4array[(alpha_r == 0) & (alpha_g == 0) & (alpha_b == 0)] = (0, 0, 0, 0)

    return img_np_4array