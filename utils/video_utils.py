import os
import cv2
import imageio
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import copy

def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)

def save_gif(frames, filename, print_path=False, fps=30):
    imageio.mimsave(filename, frames, fps=fps)

def save_video(frames, filename, print_path=False, fps=30):
    video_name = filename
    with imageio.get_writer(video_name, fps=fps) as writer:
        im_shape = frames[-1].shape
        for im in frames:
            # convert BGR to RGB
            im = im[:, :, ::-1]
            if (im.shape[0] != im_shape[0]) or (im.shape[1] != im_shape[1]):
                im = cv2.resize(im, (im_shape[1], im_shape[0]))
            writer.append_data(im.astype(np.uint8))
        writer.close()
    if print_path:
        print("Video saved to {}".format(video_name))

def draw_trajectory_on_video(episode_idx, image_dir, pos_list, output_dir, start_frame_idx=0, max_len=None, crop_info=None, save_all=False, save_last_frame=True, _save_video=True, return_video=False, save_original=False, reverse=False, cut_regions=None):
    if type(image_dir) == str:
        episode_len = len(os.listdir(image_dir))
        image_frames = []
    else:
        # image_dir itself is numpy array
        image_frames = image_dir
        episode_len = len(image_frames)

    if max_len is not None:
        episode_len = min(episode_len, max_len)
    motion_images = []

    # change pos to int
    pos_list = [(int(pos[0]), int(pos[1])) for pos in pos_list]

    start_color = np.array([255, 255, 255])  # White
    end_color = np.array([0, 255, 0])  # Green
    eef_color = (0, 0, 255)  # Red

    traj_output_dir = os.path.join(output_dir, "traj{}".format(episode_idx))
    last_frame_output_dir = os.path.join(output_dir, "last_frame")
    videos_output_dir = os.path.join(output_dir, "videos")
    if save_original:
        last_frame_wo_traj_output_dir = os.path.join(output_dir, "last_frame_wo_traj")
        videos_wo_traj_output_dir = os.path.join(output_dir, "videos_wo_traj")
    if save_all:
        if not os.path.exists(traj_output_dir):
            os.makedirs(traj_output_dir)
    if not os.path.exists(last_frame_output_dir):
        os.makedirs(last_frame_output_dir)
    if not os.path.exists(videos_output_dir):
        os.makedirs(videos_output_dir)
    if save_original:
        if not os.path.exists(last_frame_wo_traj_output_dir):
            os.makedirs(last_frame_wo_traj_output_dir)
        if not os.path.exists(videos_wo_traj_output_dir):
            os.makedirs(videos_wo_traj_output_dir)

    # load all images first
    cut_image_frames = []
    cut_pos_list = []
    for frame_idx in range(start_frame_idx, episode_len):
        if cut_regions is not None:
            if frame_idx in cut_regions:
                # skip the frame
                continue
        if type(image_dir) == str:
            image_frame_path = os.path.join(image_dir, "image{}.jpg".format(frame_idx))
            # Load the input image.
            image = cv2.imread(image_frame_path)
            cut_image_frames.append(copy.deepcopy(image))
        else:
            image = image_frames[frame_idx]
            cut_image_frames.append(copy.deepcopy(image))
        cut_pos_list.append(pos_list[frame_idx])
    if cut_regions is not None:
        episode_len -= len(cut_regions)

    if reverse:
        cut_image_frames = cut_image_frames[::-1]
        cut_pos_list = cut_pos_list[::-1]
    
    original_cut_image_frames = []
        
    for cut_frame_idx in range(episode_len - start_frame_idx):
        image = cut_image_frames[cut_frame_idx]
        if save_original:
            original_cropped_image = copy.deepcopy(image)
            if crop_info is not None:
                original_cropped_image = original_cropped_image[crop_info[1]:crop_info[3], crop_info[0]:crop_info[2], :]
            original_cut_image_frames.append(original_cropped_image)

        # draw line
        for j in range(1, cut_frame_idx + 1):
            if j < len(pos_list):
                color_ratio = j / (episode_len - start_frame_idx - 1)
                line_color = (1 - color_ratio) * start_color + color_ratio * end_color
                cv2.line(image, (cut_pos_list[j-1][0], cut_pos_list[j-1][1]), (cut_pos_list[j][0], cut_pos_list[j][1]), line_color, 5)

        if cut_frame_idx < len(cut_pos_list):
            cv2.circle(image, tuple(cut_pos_list[cut_frame_idx]), 10, eef_color, -1)

        if crop_info is not None:
            image = image[crop_info[1]:crop_info[3], crop_info[0]:crop_info[2], :]
        motion_images.append(image)

        # save the image
        if save_all:
            output_image_path = os.path.join(traj_output_dir, "image{}.jpg".format(cut_frame_idx))
            cv2.imwrite(output_image_path, image)

    # save the last frame
    if save_last_frame:
        last_frame_output_image_path = os.path.join(last_frame_output_dir, "traj{}.jpg".format(episode_idx))
        cv2.imwrite(last_frame_output_image_path, motion_images[-1])
        if save_original:
            last_frame_wo_traj_output_image_path = os.path.join(last_frame_wo_traj_output_dir, "traj{}.jpg".format(episode_idx))
            cv2.imwrite(last_frame_wo_traj_output_image_path, original_cut_image_frames[-1])

    # save the video
    if _save_video:
        output_video_path = os.path.join(videos_output_dir, "traj{}.mp4".format(episode_idx))
        save_video(motion_images, output_video_path)
        if save_original:
            output_video_path = os.path.join(videos_wo_traj_output_dir, "traj{}.mp4".format(episode_idx))
            save_video(original_cut_image_frames, output_video_path)

    if crop_info is not None:
        for idx in range(len(cut_pos_list)):
            cut_pos_list[idx] = (cut_pos_list[idx][0] - crop_info[1], cut_pos_list[idx][1] - crop_info[0])

    if return_video:
        return motion_images
    
    episode_info_dict = {"num_steps": episode_len-start_frame_idx, "trajectory": pos_list, 
                            "last_frame_path": last_frame_output_image_path,
                            "video_path": output_video_path, "reversed": reverse}
    return episode_info_dict
    

def visualize_trajectory_on_single_frame(image, eef_positions_2d, output_folder, episode_idx, save_image=True, eef_color=(255, 0, 0), show_eef=True):
    img = image #np.array(image)
    start_color = np.array([255, 255, 255])  # White
    end_color = np.array([0, 255, 0])  # Green

    # draw gradation color line for the trajectory until current frame. current position should be described with a red dot
    for i in range(len(eef_positions_2d)):
        # draw line
        for j in range(1, i):
            color_ratio = j / (len(eef_positions_2d) - 1)
            line_color = (1 - color_ratio) * start_color + color_ratio * end_color
            cv2.line(img, (int(eef_positions_2d[j-1][0]), int(eef_positions_2d[j-1][1])), (int(eef_positions_2d[j][0]), int(eef_positions_2d[j][1])), line_color, 5)

    # draw end effector only in the last frame
    if show_eef:
        cv2.circle(img, (int(eef_positions_2d[len(eef_positions_2d) - 1][0]), int(eef_positions_2d[len(eef_positions_2d) - 1][1])), 10, eef_color, -1)

    # save image
    if save_image:
        cv2.imwrite(os.path.join(output_folder, f'traj{episode_idx}.jpg'), img[:, :, ::-1])

    return img

def visualize_trajectory_on_multiple_frames(images, eef_positions_2d, output_folder, episode_idx, save_image=True, save_video=True):
    motion_images = []
    start_color = np.array([255, 255, 255])  # White
    end_color = np.array([0, 255, 0])  # Green

    # draw gradation color line for the trajectory until current frame. current position should be described with a red dot
    for i in range(len(images)):
        img = np.array(images[i])

        # draw line
        for j in range(1, i):
            color_ratio = j / (len(images) - 1)
            line_color = (1 - color_ratio) * start_color + color_ratio * end_color
            cv2.line(img, (int(eef_positions_2d[j-1][0]), int(eef_positions_2d[j-1][1])), (int(eef_positions_2d[j][0]), int(eef_positions_2d[j][1])), line_color, 5)

        # draw end effector
        cv2.circle(img, (int(eef_positions_2d[i][0]), int(eef_positions_2d[i][1])), 10, (255, 0, 0), -1)

        # save image
        if save_image:
            cv2.imwrite(os.path.join(output_folder, f'image{i}.jpg'), img[:, :, ::-1])

        motion_images.append(img[:, :, ::-1])

    # save last frame 
    if save_image:
        cv2.imwrite(os.path.join(os.path.dirname(output_folder), f'last_frame/traj{episode_idx}.jpg'), img[:, :, ::-1])

    if save_video:
        save_video(motion_images, os.path.join(os.path.dirname(output_folder), 'traj{}.mp4'.format(episode_idx)))
