import re

import imutils

import cv2 # OpenCV
import numpy as np
import random
random.seed(123)
secure_random = random.SystemRandom()
import matplotlib.pyplot as plt
import glob
import pandas as pd


# Original image, which is the background
from GT_Utils import reduce_size_for_transparency_size_png, add_img_noBG_from_jpeg, bcolors, avoid_overflow_photo_x_y_w_h, \
    remove_bg_from_png

df = pd.DataFrame()

IMAGE_WITHD_BG = 640
IMAGE_HIGNT_BG = 640

count_fondos = 1
ANGLES_TO_ROTATE_BG = [0, 90, 180, 270]
def get_fondo_back_ground():
    global count_fondos
    PATHS_FONTOS = glob.glob(r"E:\iaCarry_img_eroski\fondos" + '/*.jpg')

    path_fondo = PATHS_FONTOS[ count_fondos % len(PATHS_FONTOS)  ]
    count_fondos = count_fondos +1

    background = cv2.imread(path_fondo)
    background = cv2.resize(background, (IMAGE_WITHD_BG, IMAGE_HIGNT_BG))
    angle_bg = secure_random.choice(ANGLES_TO_ROTATE_BG)
    background = imutils.rotate(background, angle_bg)
    background = cv2.cvtColor(background,  cv2.COLOR_BGR2RGB)

    pos_x = random.randint(1, background.shape[1])
    pos_y = random.randint(1, background.shape[0])

    return background, pos_x, pos_y

def mix_img_rota_fon_with_background(path):
    img_background, rad_pos_x, rad_pos_y = get_fondo_back_ground()
    image_gt = cv2.imread(path)
    image_gt = cv2.cvtColor(image_gt, cv2.COLOR_BGR2RGB)
    if image_gt.shape[0] > IMAGE_WITHD_BG or image_gt.shape[1] > IMAGE_HIGNT_BG:
        print("WARN la imagen gt es mayor que e fondo ", path)
        image_gt = cv2.resize(image_gt, (0, 0), fx=0.5, fy=0.5)
    composition_1, pos_x, pos_y = add_img_noBG_from_jpeg(img_background, image_gt, rad_pos_x, rad_pos_y)
    path_out_reduce_size = path.replace("E:\\iaCarry_img_eroski\\FORNT\\frames_rota","E:\\iaCarry_img_eroski\\FORNT\\frames_rota_fon")
    plt.imsave(path_out_reduce_size, composition_1)
    print(path_out_reduce_size)
    return path_out_reduce_size, pos_x, pos_y, img_background.shape

MAX_REDUCTION_LABEL_IMG_PER = 50
MIN_REDUCTION_LABEL_IMG_PER = 32
def get_ramdon_minimaze_pixel():
    red_value =  float(random.randrange(MIN_REDUCTION_LABEL_IMG_PER, MAX_REDUCTION_LABEL_IMG_PER))/100
    return red_value

def plot_check_bbox_from_img(path_load, x, y, w, h):
    image_np_coco_val = cv2.imread(path_load)
    # image_np_coco_val = image_np.copy()
    cv2.rectangle(image_np_coco_val, (int(x), int(y)), (int((x + w)), int((y + h))), (255, 0, 0), 5)
    path_save = path_load.replace("E:\\iaCarry_img_eroski\\FORNT\\frames_rota_fon",
                                  "E:\\iaCarry_img_eroski\\FORNT\\frames_rota_fon_BBOX")
    cv2.imwrite(path_save, image_np_coco_val)
    print("\t Bbox img: ", path_save)


# path_out_put = r"E:\iaCarry_img_eroski\frames_no_bg_bbox\aguila_33cl_0002.png"
# # Importante para no perder la tranparencia  cv2.IMREAD_UNCHANGED
# image = cv2.imread(path_out_put,  cv2.IMREAD_UNCHANGED)

PATHS_IMG = glob.glob("E:\\iaCarry_img_eroski\\FORNT\\Base" + '/*.png')
# PATHS_IMG = glob.glob(r"E:\iaCarry_img_eroski\frames_no_bg_bbox\fanta_33cl_0002.png")# + '/*.png')
PATHS_IMG = glob.glob("E:\\iaCarry_img_eroski\\FORNT\\Base" + '/*.png')
for path_out_put in PATHS_IMG:
    image_np = cv2.imread(path_out_put, cv2.IMREAD_UNCHANGED)
    # image = cv2.imread(path_out_put, cv2.IMREAD_UNCHANGED)
    path_key = path_out_put.split("\\")[-1].split(".")[0]
    print("\n" + bcolors.HEADER + path_key + bcolors.ENDC)

    list_angles = [341, 0, 19, 88]
    for angle in list_angles:
        path__bound = "E:\\iaCarry_img_eroski\\FORNT\\frames_rota\\" +path_key +"_A" +str(angle) +".png"
        # # rotate_bound es que no se recorta la imagen al rotarla
        img_rotated_bound = imutils.rotate_bound(image_np, angle)
        # path_rot = "E:\\iaCarry_img_eroski\\FORNT\\frames_rota\\" +path_key +"_B" +str(angle) +".png"
        # # rotate_bound es que SE recorta la imagen al rotarla
        # img_rotated_rot  = imutils.rotate(image, angle)

        for path_img in [(path__bound,img_rotated_bound )   ]:
            # if random.randint(1, 6) != 1:
            #     print("\t\tramdom dont use")
            #     continue

            path = path_img[0]
            img_rotated = path_img[1]

            rad_value = get_ramdon_minimaze_pixel()
            img_rotated = cv2.resize(img_rotated, (0, 0), fx=rad_value, fy=rad_value)
            if img_rotated.shape[0] < 180 or img_rotated.shape[1] < 180:
                img_rotated = cv2.resize(img_rotated, (0, 0), fx=rad_value+ 1.3, fy=rad_value+ 1.3)
            cv2.imwrite(path, img_rotated)
            print("\t Rotated img: ", path)

            tu_size_ori ,tu_size_red, x0, y0, x1, y1 = reduce_size_for_transparency_size_png(path, path)
            path_out_reduce_size, pos_x, pos_y, img_background_shape = mix_img_rota_fon_with_background(path)
            x, y, w, h = avoid_overflow_photo_x_y_w_h(pos_x=pos_x, pos_y=pos_y, w_1=tu_size_red[0], h_1=tu_size_red[1], img_background_shape=img_background_shape)

            type_prod = re.match(r"(\w*_\w*)_\d{3,5}_[A-Za-z](\w*)*[.]png", path.split("\\")[-1] ).groups()[0]
            # dict_imgA = {"path_key": path.split("\\")[-1],"type_prod": type_prod, "box_width": tu_size_red[0], "box_height": tu_size_red[1],"box_pos_x": pos_x, "box_pos_y": pos_y, "shape": img_background_shape, "path": path_out_reduce_size }
            dict_imgA = {"path_key": path.split("\\")[-1],"type_prod": type_prod, "box_width": w,
                         "box_height": h,"box_pos_x": x, "box_pos_y": y, "shape": img_background_shape, "path": path_out_reduce_size }

            df = pd.concat([df, pd.DataFrame([dict_imgA])], ignore_index=True)
            plot_check_bbox_from_img(path_load=dict_imgA['path'], x=dict_imgA['box_pos_x'], y=dict_imgA['box_pos_y'],
                                     w=dict_imgA['box_width'], h=dict_imgA['box_height'])


print("\n", "E:\\iaCarry_img_eroski\\FORNT\\df_size_Rota_img.csv")
df.to_csv("E:\\iaCarry_img_eroski\\FORNT\\df_size_Rota_img.csv", sep="\t")


# print("Background shape:", img_background.shape)
# print("Image shape:", image.shape)