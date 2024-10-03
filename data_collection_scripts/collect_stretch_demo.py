#!/usr/bin/env python3
import pyrealsense2 as rs
import numpy as np
import cv2
import time, datetime
import os
import argparse
import imageio
import json

######################################
# Settings via command line arguments
######################################

parser = argparse.ArgumentParser(description='Tool to test the Realsense D435i Camera.')
parser.add_argument("--no_gui", action="store_true",
                    help="Show no GUI while reading images.")
parser.add_argument("--colormap", type=int, default=cv2.COLORMAP_OCEAN,
                    help="Valid OpenCV colormaps at 'https://docs.opencv.org/master/d3/d50/group__imgproc__colormap.html'.")
parser.add_argument("--save", nargs="?", type=str, const="",
                    help="Save as .mp4 video to given filepath at end of script.")
parser.add_argument("--save_limit", type=int, default=60,
                    help="The number of minutes of data to save.")
parser.add_argument("--view", type=str, choices=["topdown", "side", "topdown_side"], default="topdown_side",
                    help="The view to show.")
args, _ = parser.parse_known_args()
output_filepath = args.save

####################################
# Additional settings
####################################

# Whether to save a video of the color images and/or depth images.
save_video_color = True
save_video_depth = True
# Video streaming configuration. 
# Note: the actual fps will probably be lower than the target, especially if larger resolutions are used or multiple videos are saved.
fps_color = 6 #30 # 6 #15 #FPS for color-only videos and for color+depth videos
fps_depth_downsample_factor = 6 #5 # The depth frame rate will be fps_color/fps_depth_downsample_factor.
                                # Only used for the depth-only video stream (combined color+depth videos will be at the color fps).
resolution_color = [1280, 720] #[640, 480]
resolution_depth = [1280, 720] #[640, 480]

# Some image processing options.
apply_local_histogram_equalization = False
apply_global_histogram_equalization = False

####################################
# Initialize
####################################

# Configure depth and color streams.
# cam 1: topdown view camera, cam 2: sideview camera
if "topdown" in args.view:
    pipeline_1 = rs.pipeline()
    config_1 = rs.config()
    config_1.enable_device('045422060294')
    config_1.enable_stream(rs.stream.depth, resolution_depth[0], resolution_depth[1], rs.format.z16, fps_color) # note that the fps downsampling will be applied later
    config_1.enable_stream(rs.stream.color, resolution_color[0], resolution_color[1], rs.format.bgr8, fps_color)
    frame_width_color_1 = resolution_color[0]
    frame_height_color_1 = resolution_color[1]
    frame_width_depth_1 = resolution_color[0]
    frame_height_depth_1 = resolution_color[1]
else:
    pipeline_1 = None

if "side" in args.view:
    pipeline_2 = rs.pipeline()
    config_2 = rs.config()
    config_2.enable_device('936322070007')
    config_2.enable_stream(rs.stream.depth, resolution_depth[0], resolution_depth[1], rs.format.z16, fps_color) # note that the fps downsampling will be applied later
    config_2.enable_stream(rs.stream.color, resolution_color[0], resolution_color[1], rs.format.bgr8, fps_color)
    frame_width_color_2 = resolution_color[0]
    frame_height_color_2 = resolution_color[1]
    frame_width_depth_2 = resolution_depth[0]
    frame_height_depth_2 = resolution_depth[1]
else:
    pipeline_2 = None

# Create writer(s) to save the stream(s) if desired.
date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# make the directory if it doesn't exist
if not os.path.exists(f"./data/stretch/{date_str}"):
    os.makedirs(f"./data/stretch/{date_str}")

args.no_gui = False

####################################
# Stream!
####################################

frameBuffer_color_1 = []
frameBuffer_depth_1 = []
frameBuffer_color_2 = []
frameBuffer_depth_2 = []
data_info = []

if pipeline_1 is not None:
    pipeline_1.start(config_1)
if pipeline_2 is not None:
    pipeline_2.start(config_2)
start_capture_time_s = time.time()
end_capture_time_color_s = start_capture_time_s
end_capture_time_depth_s = start_capture_time_s
frame_count_color = 0
frame_count_depth = 0
episode_count = 0
# set video writer for the color and depth video
writer_color_1 = None
writer_depth_1 = None
writer_color_2 = None
writer_depth_2 = None

if save_video_color:
    if pipeline_1 is not None:
        writer_color_1 = imageio.get_writer(f"./data/stretch/{date_str}/traj{episode_count}_color.mp4", fps=fps_color)
    if pipeline_2 is not None:
        writer_color_2 = imageio.get_writer(f"./data/stretch/{date_str}/traj{episode_count}_color_sideview.mp4", fps=fps_color)
if save_video_depth:
    if pipeline_1 is not None:
        writer_depth_1 = imageio.get_writer(f"./data/stretch/{date_str}/traj{episode_count}_depth.mp4", fps=fps_color/fps_depth_downsample_factor)
    if pipeline_2 is not None:
        writer_depth_2 = imageio.get_writer(f"./data/stretch/{date_str}/traj{episode_count}_depth_sideview.mp4", fps=fps_color/fps_depth_downsample_factor)
traj_save_path = "./data/stretch/{}/traj{}".format(date_str, episode_count)
if not os.path.exists(traj_save_path):
    os.makedirs(traj_save_path)
    if pipeline_1 is not None:
        os.makedirs(os.path.join(traj_save_path, "color"))
        os.makedirs(os.path.join(traj_save_path, "depth"))
    if pipeline_2 is not None:
        os.makedirs(os.path.join(traj_save_path, "color_sideview"))
        os.makedirs(os.path.join(traj_save_path, "depth_sideview"))
data_save_path = "./data/stretch/{}/data_info.json".format(date_str)

color_frame_1 = None
depth_frame_1 = None
color_frame_2 = None
depth_frame_2 = None

# input
task_instruction = input("Task instruction: ")
motion_description = input("Motion description: ")

while True:
    # Get the latest frames from the camera.
    if pipeline_1 is not None:
        frames_1 = pipeline_1.wait_for_frames()
    if pipeline_2 is not None:
        frames_2 = pipeline_2.wait_for_frames()

    if pipeline_1 is not None:
        depth_frame_1 = frames_1.get_depth_frame()
        color_frame_1 = frames_1.get_color_frame()
    if pipeline_2 is not None:
        depth_frame_2 = frames_2.get_depth_frame()
        color_frame_2 = frames_2.get_color_frame()
    if pipeline_1 is not None and (not depth_frame_1 or not color_frame_1):
        continue
    if pipeline_2 is not None and (not depth_frame_2 or not color_frame_2):
        continue
    if pipeline_1 is None and pipeline_2 is None:
        raise ValueError("Both pipelines are None!")
    capture_time_s = time.time()

    # Convert images to numpy arrays.
    if pipeline_1 is not None:
        depth_image_1 = np.asanyarray(depth_frame_1.get_data())
        color_image_1 = np.asanyarray(color_frame_1.get_data())
    if pipeline_2 is not None:
        depth_image_2 = np.asanyarray(depth_frame_2.get_data())
        color_image_2 = np.asanyarray(color_frame_2.get_data())

    # Apply histogram equalization if desired.
    if apply_local_histogram_equalization:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        if pipeline_1 is not None:
            color_image_lab_1 = cv2.cvtColor(color_image_1, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
            color_image_l_1, color_image_a_1, color_image_b_1= cv2.split(color_image_lab_1)  # split on 3 different channels
            color_image_l_equalized_1 = clahe.apply(color_image_l_1)  # apply CLAHE to the L-channel
            color_image_lab_1 = cv2.merge((color_image_l_equalized_1, color_image_a_1, color_image_b_1))  # merge channels
            color_image_1 = cv2.cvtColor(color_image_lab_1, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR

        if pipeline_2 is not None:
            color_image_lab_2 = cv2.cvtColor(color_image_2, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
            color_image_l_2, color_image_a_2, color_image_b_2= cv2.split(color_image_lab_2)  # split on 3 different channels
            color_image_l_equalized_2 = clahe.apply(color_image_l_2)  # apply CLAHE to the L-channel
            color_image_lab_2 = cv2.merge((color_image_l_equalized_2, color_image_a_2, color_image_b_2))  # merge channels
            color_image_2 = cv2.cvtColor(color_image_lab_2, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR

    elif apply_global_histogram_equalization:
        if pipeline_1 is not None:
            color_image_lab_1 = cv2.cvtColor(color_image_1, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
            color_image_l_1, color_image_a_1, color_image_b_1 = cv2.split(color_image_lab_1)  # split on 3 different channels
            color_image_l_equalized_1 = cv2.equalizeHist(color_image_l_1) # apply global equalization to the L-channel
            color_image_lab_1 = cv2.merge((color_image_l_equalized_1, color_image_a_1, color_image_b_1))  # merge channels
            color_image_1 = cv2.cvtColor(color_image_lab_1, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR

        if pipeline_2 is not None:
            color_image_lab_2 = cv2.cvtColor(color_image_2, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
            color_image_l_2, color_image_a_2, color_image_b_2 = cv2.split(color_image_lab_2)  # split on 3 different channels
            color_image_l_equalized_2 = cv2.equalizeHist(color_image_l_2) # apply global equalization to the L-channel
            color_image_lab_2 = cv2.merge((color_image_l_equalized_2, color_image_a_2, color_image_b_2))  # merge channels
            color_image_2 = cv2.cvtColor(color_image_lab_2, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR

    # Apply colormap on depth image (image must be converted to 8-bit per pixel first).
    if pipeline_1 is not None:
        depth_image_1 = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_1, alpha=-0.04, beta=255.0), args.colormap)
    if pipeline_2 is not None:
        depth_image_2 = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_2, alpha=-0.04, beta=255.0), args.colormap)

    # resize image 2 to match the height of image 1
    if pipeline_1 is not None and pipeline_2 is not None:
        resized_frame_width_color_2 = int(frame_width_color_2 * frame_height_color_1 / frame_height_color_2)
        color_image_2 = cv2.resize(color_image_2, (resized_frame_width_color_2, frame_height_color_1))
        resized_color_image_2 = cv2.resize(color_image_2, (resized_frame_width_color_2, frame_height_color_1))
        # concat color_image_1 and resized_color_image_2
        multiview_color_image = np.concatenate((color_image_1, resized_color_image_2), axis=1)
    elif pipeline_1 is not None:
        multiview_color_image = color_image_1
    elif pipeline_2 is not None:
        multiview_color_image = color_image_2

    # Show stream if no_gui disabled.
    if not args.no_gui:
        cv2.namedWindow('Realsense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Realsense', multiview_color_image)
        key = cv2.waitKey(1) & 0xFF

    # if key "e" is pressed in terminal, save until the current frame and initialize the frame count and buffer
    if key == 101:
        print("Saving ... ")
        # Save the video
        if writer_color_1 is not None:
            writer_color_1.close()
        if writer_depth_1 is not None:
            writer_depth_1.close()
        if writer_color_2 is not None:
            writer_color_2.close()
        if writer_depth_2 is not None:
            writer_depth_2.close()
        for frame_idx in range(frame_count_color):
            if pipeline_1 is not None:
                cv2.imwrite(os.path.join(traj_save_path, "color", "image{}.jpg".format(frame_idx)), frameBuffer_color_1[frame_idx])
                cv2.imwrite(os.path.join(traj_save_path, "depth", "image{}.jpg".format(frame_idx)), frameBuffer_depth_1[frame_idx])
            if pipeline_2 is not None:
                cv2.imwrite(os.path.join(traj_save_path, "color_sideview", "image{}.jpg".format(frame_idx)), frameBuffer_color_2[frame_idx])
                cv2.imwrite(os.path.join(traj_save_path, "depth_sideview", "image{}.jpg".format(frame_idx)), frameBuffer_depth_2[frame_idx])

        data_info.append({"episode_idx": episode_count, "task_instruction": task_instruction, "motion_description": motion_description, "num_steps": frame_count_color,
                          "fps_color": fps_color, "fps_depth": fps_color/fps_depth_downsample_factor})

        print("="*10 + "Episode {} saved!".format(episode_count)+ "="*10)
        # Increment the episode count
        episode_count += 1
        # set video writer for the color and depth video
        if save_video_color:
            if pipeline_1 is not None:
                writer_color_1 = imageio.get_writer(f"./data/stretch/{date_str}/traj{episode_count}_color.mp4", fps=fps_color)
            if pipeline_2 is not None:
                writer_color_2 = imageio.get_writer(f"./data/stretch/{date_str}/traj{episode_count}_color_sideview.mp4", fps=fps_color)
        if save_video_depth:
            if pipeline_1 is not None:
                writer_depth_1 = imageio.get_writer(f"./data/stretch/{date_str}/traj{episode_count}_depth.mp4", fps=fps_color/fps_depth_downsample_factor)
            if pipeline_2 is not None:
                writer_depth_2 = imageio.get_writer(f"./data/stretch/{date_str}/traj{episode_count}_depth_sideview.mp4", fps=fps_color/fps_depth_downsample_factor)
        traj_save_path = "./data/stretch/{}/traj{}".format(date_str, episode_count)
        if not os.path.exists(traj_save_path):
            os.makedirs(traj_save_path)
            os.makedirs(os.path.join(traj_save_path, "color"))
            os.makedirs(os.path.join(traj_save_path, "depth"))
            os.makedirs(os.path.join(traj_save_path, "color_sideview"))
            os.makedirs(os.path.join(traj_save_path, "depth_sideview"))
        # save data info
        with open(data_save_path, "w") as f:
            json.dump(data_info, f, indent=4)
        # Reset the frame count
        frame_count_color = 0
        frame_count_depth = 0
        # Reset the buffer
        frameBuffer_color_1 = []
        frameBuffer_depth_1 = []
        frameBuffer_color_2 = []
        frameBuffer_depth_2 = []
    # if key "s" is pressed, reset the frame count and buffer
    elif key == 115:
        # print task instruction and motion description
        print("="*10 + "Episode {} Reset".format(episode_count) + "="*10)
        print("Task instruction: {}".format(task_instruction))
        print("Motion description: {}".format(motion_description))
        if save_video_color:
            if pipeline_1 is not None:
                writer_color_1 = imageio.get_writer(f"./data/stretch/{date_str}/traj{episode_count}_color.mp4", fps=fps_color)
            if pipeline_2 is not None:
                writer_color_2 = imageio.get_writer(f"./data/stretch/{date_str}/traj{episode_count}_color_sideview.mp4", fps=fps_color)
        if save_video_depth:
            if pipeline_1 is not None:
                writer_depth_1 = imageio.get_writer(f"./data/stretch/{date_str}/traj{episode_count}_depth.mp4", fps=fps_color/fps_depth_downsample_factor)
            if pipeline_2 is not None:
                writer_depth_2 = imageio.get_writer(f"./data/stretch/{date_str}/traj{episode_count}_depth_sideview.mp4", fps=fps_color/fps_depth_downsample_factor)
        # Reset the frame count
        frame_count_color = 0
        frame_count_depth = 0
        # Reset the buffer
        frameBuffer_color_1 = []
        frameBuffer_depth_1 = []
        frameBuffer_color_2 = []
        frameBuffer_depth_2 = []
        print("Frame count and buffer reset!")
    # if key "t" is pressed, input the task instruction
    elif key == 116:
        task_instruction = input("Task instruction: ")
    # if key "m" is pressed, input the motion description
    elif key == 109:
        motion_description = input("Motion description: ")
    # if key "q" is pressed, exit the loop
    elif key == 113:
        break
    
    # Maintain stream cache.
    # Important to use copy() here depending on how the stream is used.
    if pipeline_1 is not None:
        frameBuffer_color_1.append(color_image_1.copy())
        frameBuffer_depth_1.append(depth_image_1.copy())
    if pipeline_2 is not None:
        frameBuffer_color_2.append(color_image_2.copy())
        frameBuffer_depth_2.append(depth_image_2.copy())

    # Write to video output(s) if desired.
    if writer_color_1 is not None:
        writer_color_1.append_data(frameBuffer_color_1[-1][:, :, ::-1])
        end_capture_time_color_s = capture_time_s
    if writer_depth_1 is not None and frame_count_color % fps_depth_downsample_factor == 0:
        writer_depth_1.append_data(frameBuffer_depth_1[-1][:, :, ::-1])
        end_capture_time_depth_s = capture_time_s
        
    if writer_color_2 is not None:
        writer_color_2.append_data(color_image_2[:, :, ::-1])
        end_capture_time_color_s = capture_time_s
    if writer_depth_2 is not None and frame_count_color % fps_depth_downsample_factor == 0:
        writer_depth_2.append_data(depth_image_2[:, :, ::-1])
        end_capture_time_depth_s = capture_time_s
    
    frame_count_color += 1
    frame_count_depth += 1
            
print("Done!")