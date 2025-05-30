import os
import re
import uuid

import imutils

import cv2 # OpenCV
import numpy as np
import random

import GT_Utils
import GT_Utils_ImageAugmentation_presp

random.seed(123)
np.random.seed(123)
secure_random = random.SystemRandom()
import matplotlib.pyplot as plt
import glob
import pandas as pd
# Original image, which is the background
from GT_Utils import reduce_size_for_transparency_size_png, add_img_noBG_from_jpeg, bcolors, avoid_overflow_photo_x_y_w_h, \
    reduce_size_for_transparency_size

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
    return background

def get_ramdon_x_y_points(w_w1, h_h1, NUM_IMG_PER_COL ,NUM_IMG_PER_ROW, shape_img):
    exclude_pixes_x = int(shape_img[0] / NUM_IMG_PER_ROW / 4)
    exclude_pixes_y = int(shape_img[1] / NUM_IMG_PER_COL / 4)
    #Queremos imagenes que esten cretradas , que no se vayan por la esquina de abajo , pero que sean aleatorias
    pos_x = random.randint(int(w_w1[0])+25, int(w_w1[1]) - exclude_pixes_x)
    pos_y = random.randint(int(h_h1[0])+25, int(h_h1[1]) - exclude_pixes_y)
    return pos_x, pos_y

def get_type_prod(patH_key):
    type_prod = re.match(r"(\w*_\w*)_([A-Z]{1,2})\w*[ ]*\(?\d*\)?\w*[.]png", patH_key.split("\\")[-1]).groups()[0]
    type_prod_view = re.match(r"(\w*_\w*)_([A-Z]{1,2})\w*[ ]*\(?\d*\)?\w*[.]png", patH_key.split("\\")[-1]).groups()[1]
    return type_prod, type_prod_view


MAX_REDUCTION_LABEL_IMG_PER = 45
MIN_REDUCTION_LABEL_IMG_PER = 32
def get_ramdon_minimaze_pixel():
    red_value =  float(random.randrange(MIN_REDUCTION_LABEL_IMG_PER, MAX_REDUCTION_LABEL_IMG_PER))/100
    return red_value

def plot_check_bbox_from_img(image_np,path_load, x, y, w, h):
    image_np_coco_val = image_np #cv2.imread(path_load)
    # image_np_coco_val = image_np.copy()
    color = tuple(list(np.random.random(size=3) * 256))
    cv2.rectangle(image_np_coco_val, (int(x), int(y)), (int((x + w)), int((y + h))), color, 5)
    path_save = path_load.replace("E:\\iaCarry_img_eroski\\Blanco_Mix",
                                  "E:\\iaCarry_img_eroski\\Blanco_rota_222")
    cv2.imwrite(path_save, image_np_coco_val)
    print("\t Bbox img: ", path_save)

# def plot_check_bbox_from_img_2(path_load, x, y, w, h,x2, y2, w2, h2  ):
#     image_np_coco_val = cv2.imread(path_load)
#     # image_np_coco_val = image_np.copy()
#     cv2.rectangle(image_np_coco_val, (int(x), int(y)), (int((x + w)), int((y + h))), (0, 255, 0), 5)
#     cv2.rectangle(image_np_coco_val, (int(x2), int(y2)), (int((x2 + w2)), int((y2 + h2))), (0, 255, 255), 5)
#     path_save = path_load.replace("E:\\iaCarry_img_eroski\\FORNT\\frames_rota_fon",
#                                   "E:\\iaCarry_img_eroski\\FORNT\\frames_rota_fon_BBOX")
#     cv2.imwrite(path_save, image_np_coco_val)
#     print("\t Bbox img: ", path_save)


ROTATION_IMG_PATH = "E:\\iaCarry_img_eroski\\Blanco_rota\\"
LIST_RAMDON_GRADES_bound = [0,random.randint(2, 38), random.randint(2, 38), random.randint(325, 358), random.randint(325, 358)]
LIST_RAMDON_GRADES_no_bound = [0,random.randint(20, 45), random.randint(20, 45), random.randint(315, 340), random.randint(315, 340)]

def get_list_ramdon_path_imgnp(PATHS_IMG_general  , num_img_aprox = 4):
    PATHS_IMG = glob.glob(PATHS_IMG_general)
    dict_ramdon_path_imgnp = {}

    ram_a , ram_b = 1, 3
    if num_img_aprox <= 2:#si hay pocas images hay que imprimierlas todas
        ram_a, ram_b = 1, 2

    for i in range(0, num_img_aprox):

        path_img_ramdon = random.choice(PATHS_IMG)
        path_key_ram = path_img_ramdon.split("\\")[-1].split(".")[0]
        image_np = cv2.imread(path_img_ramdon, cv2.IMREAD_UNCHANGED)

        #NECESARIO PARA EL  METODO add_img_noBG_from_jpeg(background, img, x, y):
        image_np = GT_Utils.remove_when_traparency_is_zero_png(image_np)
        # reduce_size_for_transparency_size(path_img_ramdon, path_img_ramdon.replace(".png", "DELETE.png.png"))

        ram_chose = random.randint(ram_a , ram_b)
        if   ram_chose == 1 :#ramdon boolean
            random_angle = random.choice(LIST_RAMDON_GRADES_bound)
            path_rot = ROTATION_IMG_PATH +path_key_ram +"_g" +str(random_angle) +".png"
            # rotate_bound es que no se recorta la imagen al rotarla
            img_rotated = imutils.rotate_bound(image_np, random_angle)
        elif ram_chose == 2:
            random_angle = random.choice(LIST_RAMDON_GRADES_no_bound)
            path_rot = ROTATION_IMG_PATH +path_key_ram +"_gg" +str(random_angle) +".png"
            # rotate_bound es que SE recorta la imagen al rotarla
            img_rotated  = imutils.rotate(image_np, random_angle)
        else:
            img_rotated = None
            path_rot = "EMPTY_KEY_" +str(i)


        if  ram_chose == 1 or ram_chose == 2:
            rad_value = get_ramdon_minimaze_pixel()
            img_rotated = cv2.resize(img_rotated, (0, 0), fx=rad_value, fy=rad_value)
            # cv2.imwrite(path_rot, img_rotated)
        # print("\t Rotated img: ", path_rot)
        dict_ramdon_path_imgnp[path_rot] = img_rotated



    return dict_ramdon_path_imgnp


def resize_img(image_fg):
    image_fg = cv2.resize(image_fg, (0, 0), fx=rad_value, fy=rad_value)
    if image_fg.shape[0] < 180 or image_fg.shape[1] < 180:
        image_fg = cv2.resize(image_fg, (0, 0), fx=rad_value + 1.3, fy=rad_value + 1.3)
    return image_fg



def get_puntos_cuadriculas_en_la_img(composition_1, num_row = 2, num_col = 3 ):
    # La cuadricula de donde van los objetos list_col y list_row
    row_coeficient = composition_1.shape[0] / num_row  # filas Value
    col_coeficient = composition_1.shape[0] / num_col  # columnas Value
    list_col = []
    list_row = []
    for i in range(0, NUM_IMG + 1):
        list_col.append((i * col_coeficient % composition_1.shape[0],
                         (i * col_coeficient % composition_1.shape[0]) + col_coeficient))
        list_row.append((i * row_coeficient % composition_1.shape[1],
                         (i * row_coeficient % composition_1.shape[1]) + row_coeficient))
    return list_col, list_row , row_coeficient, col_coeficient



LIST_KEYS = ["asturiana_1l", "aguila_33cl", "axedrak_150ml", "chipsahoy_300g", "cocacola_33cl", "colacao_383g", "colgate_75ml", "estrellag_33cl", "fanta_33cl", "hysori_300ml", "mahou00_33cl", "mahou5_33cl", "smacks_330g", "tostarica_570g"]
NUM_IMG = 2
NUM_IMG_PER_ROW = 1
NUM_IMG_PER_COL = 2
mix_gen_count = 0
LIST_DICT_ELE = []

for k in LIST_KEYS:
    dict_ele_1 = {"num_imgs": 55, "num_ele": 1, "n_row": 1, "n_col": 1,
                  "path": "E:\\iaCarry_img_eroski\\Blanco_rb\\" +k +'*.png'}
    LIST_DICT_ELE.append(dict_ele_1)
dict_ele_2 = {"num_imgs":300, "num_ele": 2, "n_row": 1, "n_col":2,"path": "E:\\iaCarry_img_eroski\\Blanco_rb" + '/*.png'}
dict_ele_4 = {"num_imgs":150, "num_ele": 4, "n_row": 2, "n_col":2,"path": "E:\\iaCarry_img_eroski\\Blanco_rb" + '/*.png'}
dict_ele_6 = {"num_imgs":100, "num_ele": 6, "n_row": 2, "n_col":3,"path": "E:\\iaCarry_img_eroski\\Blanco_rb" + '/*.png'}
dict_ele_8 = {"num_imgs":65,  "num_ele": 8, "n_row": 3, "n_col":3,"path": "E:\\iaCarry_img_eroski\\Blanco_rb" + '/*.png'}
LIST_DICT_ELE.append(dict_ele_2)
LIST_DICT_ELE.append(dict_ele_4)
LIST_DICT_ELE.append(dict_ele_6)
LIST_DICT_ELE.append(dict_ele_8)


for dict_ele in LIST_DICT_ELE:
    print("\n" + bcolors.HEADER + str(dict_ele) + bcolors.ENDC)
    NUM_IMG = dict_ele['num_ele']
    NUM_IMG_PER_ROW = dict_ele['n_row']
    NUM_IMG_PER_COL = dict_ele['n_col']

    for _ in range(0,dict_ele['num_imgs']):
        dict_ramdon_path_imgnp = get_list_ramdon_path_imgnp( PATHS_IMG_general =  dict_ele["path"] , num_img_aprox=NUM_IMG)
        img_background = get_fondo_back_ground()
        composition_1 = img_background.copy()
        list_col, list_row , row_coeficient, col_coeficient = get_puntos_cuadriculas_en_la_img(composition_1, num_row = NUM_IMG_PER_ROW, num_col = NUM_IMG_PER_COL)

        i = -1
        num_ele_per_png = len([ x for x in dict_ramdon_path_imgnp.values() if x  is not  None])
        for patH_key, image_fg in dict_ramdon_path_imgnp.items():
            i = i + 1  # Al final , debe empezar en 0 para hacer bien la cuadricula
            if image_fg  is  None:
                continue
            try:
                rad_value = get_ramdon_minimaze_pixel()
                image_fg = resize_img(image_fg)
                reduce_size = round(  len( dict_ramdon_path_imgnp.keys() ) /NUM_IMG  , 3)
                image_fg = cv2.resize(image_fg, (0, 0), fx=reduce_size, fy=reduce_size)

                cv2.imwrite(patH_key, image_fg)
                if random.randint(0, 2) == 1: #una de cada 4
                    GT_Utils_ImageAugmentation_presp.get_img_ram_perspective_transform(patH_key, patH_key)

                print("\t Rotated img: ", patH_key, " \t ", image_fg.shape)
                _, tu_size_red, x0, y0, x1, y1 = reduce_size_for_transparency_size_png(patH_key,path_exit = patH_key )
                image_fg = cv2.imread(patH_key)
                image_fg = cv2.cvtColor(image_fg, cv2.COLOR_BGR2RGB)

                rad_pos_x, rad_pos_y = get_ramdon_x_y_points(list_col[i % NUM_IMG_PER_COL], list_row[i // NUM_IMG_PER_ROW], NUM_IMG_PER_COL ,NUM_IMG_PER_ROW, shape_img =composition_1.shape ) #Tiene que hacer una cuadricula con los numeros
                composition_1, pos_x, pos_y = add_img_noBG_from_jpeg(composition_1, image_fg, rad_pos_x, rad_pos_y)

                x, y, w, h = avoid_overflow_photo_x_y_w_h(pos_x=pos_x, pos_y=pos_y, w_1=tu_size_red[0], h_1=tu_size_red[1],img_background_shape=composition_1.shape)
                type_prod, type_prod_view =  get_type_prod(patH_key)

                id_img_mix = "mix_{:d}_".format(num_ele_per_png) + "{:05d}".format(mix_gen_count)
                path_out_reduce_size = "E:\\iaCarry_img_eroski\\Blanco_Mix\\" + id_img_mix + ".png"
                dict_imgA = { "id_box": str(uuid.uuid4().fields[-1])[:5] ,"id_mix": id_img_mix, "type_prod": type_prod, "view": type_prod_view, "box_width": w,
                             "box_height": h, "box_pos_x": x, "box_pos_y": y,"num_imgs": num_ele_per_png, "shape": composition_1.shape,
                             "path": path_out_reduce_size}

                df = pd.concat([df, pd.DataFrame([dict_imgA])], ignore_index=True)
                check_plot_path = dict_imgA['path'].replace(".png", "_{:d}.png").format(i)
                plot_check_bbox_from_img(image_np= cv2.cvtColor(composition_1, cv2.COLOR_RGB2BGR),path_load=check_plot_path,
                                         x=dict_imgA['box_pos_x'], y=dict_imgA['box_pos_y'],w=dict_imgA['box_width'], h=dict_imgA['box_height'])
            except Exception:
                print("WARN EXCEPCION ", Exception)
                continue


        mix_gen_count = mix_gen_count +1
        composition_1 = cv2.cvtColor(composition_1, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path_out_reduce_size, composition_1)
        print(path_out_reduce_size)

print("\n", "E:\\iaCarry_img_eroski\\df_size_BLANCO_Mix.csv")
df.to_csv("E:\\iaCarry_img_eroski\\df_size_BLANCO_Mix.csv", sep="\t")


print("end")




