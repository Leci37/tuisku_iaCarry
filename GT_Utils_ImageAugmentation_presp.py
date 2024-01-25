import numpy as np
import random
import os
import cv2
import copy
import glob


# https://github.com/bbying81/DataAugment-Opencv/blob/master/DataAugment-opencv.ipynb
import GT_Utils


def perspective_transform( img, factor1, factor2, factor3):
    """透视变换"""
    pts1 = np.float32(factor1)
    pts2 = np.float32(factor2)
    M = cv2.getPerspectiveTransform(pts1, pts2)
    res = cv2.warpPerspective(img, M, factor3)

    return res

# https://github.com/bbying81/DataAugment-Opencv/blob/master/DataAugment-opencv.ipynb
def random_scale( img, min_translation, max_translation):
    """随机缩放"""
    factor = random_vector(min_translation, max_translation)
    scale_matrix = np.array([[factor[0], 0, 0], [0, factor[1], 0], [0, 0, 1]])
    out_img = apply(img, scale_matrix)
    return scale_matrix, out_img

def apply( img, trans_matrix):
    """应用变换"""
    tmp_matrix = adjust_transform_for_image(img, trans_matrix)
    out_img = apply_transform(img, tmp_matrix)
    # if self.debug:
    #     self.show(out_img)
    return out_img

def apply_transform( img, transform):
    """仿射变换"""
    output = cv2.warpAffine(img, transform[:2, :], dsize=(img.shape[1], img.shape[0]),
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
                            borderValue=0, )  # cv2.BORDER_REPLICATE,cv2.BORDER_TRANSPARENT
    return output


def adjust_transform_for_image( img, trans_matrix):
    """根据图像调整当前变换矩阵"""
    transform_matrix = copy.deepcopy(trans_matrix)
    height, width, channels = img.shape
    transform_matrix[0:2, 2] *= [width, height]
    center = np.array((0.5 * width, 0.5 * height))
    transform_matrix = np.linalg.multi_dot(
        [basic_matrix(center), transform_matrix, basic_matrix(-center)])
    return transform_matrix

def basic_matrix(translation):
    """基础变换矩阵"""
    return np.array([[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]])

def random_vector( min, max):
    """生成范围矩阵"""
    min = np.array(min)
    max = np.array(max)
    print(min.shape, max.shape)
    assert min.shape == max.shape
    assert len(min.shape) == 1
    return np.random.uniform(min, max)


def get_img_ram_perspective_transform(img_path, img_path_save):
    X_imgs = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    # img = remove_when_traparency_is_zero_png(X_imgs)
    # img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    img = X_imgs[:, :, 0:3]
    alpha = X_imgs[:, :, 3]
    img[alpha == 0] = (0, 0, 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print("\tPerspective: ", img_path, " ", img.shape)
    ri = random.randint(2, 33)
    m = int(np.mean(img.shape) / 3)
    # All points are in format [cols, rows]
    factor1 = [[0, 0], [img.shape[0], 0], [0, img.shape[1]], [img.shape[0], img.shape[1]]]
    factor3 = (img.shape[0], img.shape[1])
    int_op = random.randint(0, 3)


    if int_op == 0:
        factor2 = [[m, m], [img.shape[0], 0], [0, img.shape[1]], [img.shape[0], img.shape[1]]]
    elif int_op == 1:
        factor2 = [[0, 0], [img.shape[0] + m, 0 + m], [0, img.shape[1]], [img.shape[0], img.shape[1]]]
    elif int_op == 2:
        factor2 = [[0, 0], [img.shape[0], 0], [0 + m, img.shape[1] + m], [img.shape[0], img.shape[1]]]
    elif int_op == 3:
        factor2 = [[0, 0], [img.shape[0], 0], [0, img.shape[1]], [img.shape[0] + m, img.shape[1] + m]]

    # factor2 = [[m, m], [img.shape[0], 0], [0, img.shape[1]], [img.shape[0], img.shape[1]]]
    img_perspective_transform = perspective_transform(img, factor1, factor2, factor3)
    # save_name = img_path_save + '/' + img_path[:-4] + '_' + 'persp' + '_'  + '.png'
    img_p = cv2.cvtColor(img_perspective_transform, cv2.COLOR_BGR2RGB)
    img_p = GT_Utils.recover_transparency_from_jpg_to_png(img_p, type_img=cv2.COLOR_RGB2RGBA)
    cv2.imwrite(img_path_save, img_p)
    # Necesario reduce_size_for_transparency_size_png pero se hace fuera
    # tu_size_ori, tu_size_red, x0, y0, x1, y1 = GT_Utils.reduce_size_for_transparency_size_png(img_path_save,img_path_save)  # cv2.cvtColor(img_perspective_transform, cv2.COLOR_BGR2RGB))
    # print(save_name)
    return img_p, img_path_save


def generate_img_random_scale(img_np):
    _, scale = random_scale(img_np, (1.2, 1.2), (1.3, 1.3))
    # save_name = save_path + '/' + img_str[:-4] + '_' + 'scale' + '_' + '.png'
    img_p = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    img_p = GT_Utils.recover_transparency_from_jpg_to_png(img_p, type_img=cv2.COLOR_RGB2RGBA)
    # cv2.imwrite(save_name, img_p)
    # tu_size_ori, tu_size_red, x0, y0, x1, y1 = GT_Utils.reduce_size_for_transparency_size_png(save_name, save_name)
    return img_p


# if __name__ == "__main__":
#     # demo = DataAugment(debug=True)
#     save_path = './img_aug'
#     imgs = glob.glob(r"E:\iaCarry_img_eroski\Blanco_cut" + '/*.png') # glob.glob('E:\iaCarry_img_eroski\Blanco_cut\*.jpg')
#     i = 0
#     for img in imgs[99:106]:
#         img_str = os.path.basename(img)
#         # bgr_img = cv2.imread(img)
#         # img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
#         get_img_ram_perspective_transform()
#
#         generate_img_random_scale()