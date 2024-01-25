# https://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames
import glob
import pandas as pd

from GT_Utils import remove_bg_from_png, reduce_size_for_transparency_size_png

df = pd.DataFrame()

PATHS_IMG = glob.glob(r"E:\iaCarry_img_eroski\frames" + '/*.png')
for p in PATHS_IMG:
    path_out_put = p.replace(r"E:\iaCarry_img_eroski\frames", r"E:\iaCarry_img_eroski\frames_no_bg")
    remove_bg_from_png(p, path_out_put)

    path_out_reduce_size = p.replace(r"E:\iaCarry_img_eroski\frames", r"E:\iaCarry_img_eroski\frames_no_bg_bbox")
    tu_size_ori ,tu_size_red, x0, y0, x1, y1 =  reduce_size_for_transparency_size_png(path_out_put, path_out_reduce_size)
    print(path_out_reduce_size, "\t Size:",tu_size_red )


    dict_img = {"path_key": path_out_put.split("\\")[-1], "cut_rmb_x":tu_size_red[0], "cut_rmb_y":tu_size_red[1], "rmb_x":tu_size_ori[1], "rmb_y":tu_size_ori[0] }
    df_img = pd.DataFrame([dict_img])
    df = pd.concat([df, df_img], ignore_index=True)

print("\n", r"E:\iaCarry_img_eroski\df_size_img.csv")
df.to_csv(r"E:\iaCarry_img_eroski\df_size_img.csv", sep="\t")





