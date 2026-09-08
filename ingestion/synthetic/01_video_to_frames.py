# https://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames
import glob

import cv2
import time
import os
import re

COUNT_EACH_FRAME_TAKE_IMAG = 10

def video_to_frames(path_video, path_exit, key_product):
    """Function to extract frames from input video file
    and save them as separate frames in an output directory.
    Args:
        path_video: Input video file.
        path_exit: Output directory to save the frames.
    Returns:
        None
    """
    try:
        os.mkdir(path_exit)
    except OSError:
        pass
    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(path_video)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print ("Number of frames: ", video_length, "Path: ", path_video)
    count = 0
    count_select = 0

    if "_up_" not in path_video:
        count_select = 100

    print ("Converting video..\n")
    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        if not ret:
            continue
        # Write the results back to output location.
        # cada 7 frames guardo uno
        if count % COUNT_EACH_FRAME_TAKE_IMAG == 0:
            count_select = count_select + 1
            cv2.imwrite(path_exit +"/" + key_product +"_%#04d.png" % (count_select + 1), frame)
        count = count + 1
        # If there are no more frames left
        if (count > (video_length-1)):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            print ("Done extracting frames.\n%d frames extracted" % count, " Number images get : ",count_select)
            print ("It took %d seconds forconversion." % (time_end-time_start))
            break

if __name__=="__main__":
    PATH_IMGS_SAVE = r"E:\iaCarry_img_eroski\frames_Mul"
    VIDEOS_PATH = glob.glob(r"E:\iaCarry_img_eroski\test_all_ele" + '/*.mp4')

    for vid_path in VIDEOS_PATH:
        key_master_name = re.search("E:\\\\iaCarry_img_eroski\\\\test_all_ele\\\\(\w*)[.]mp4", vid_path).group(1)
        key_name = key_master_name   #.split("_")[0] +"_"+ key_master_name.split("_")[2]
        print(key_name)
        video_to_frames(vid_path, PATH_IMGS_SAVE, key_name)

    # PARH_VIDEO = r"E:\iaCarry_img_eroski\fanta_lat_33cl.mp4"

