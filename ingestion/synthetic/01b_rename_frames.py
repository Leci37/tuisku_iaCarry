import glob


PATHS_IMG = glob.glob(r"E:\iaCarry_img_eroski\Blanco" + '/*.png')
for p in PATHS_IMG:

    print(p)