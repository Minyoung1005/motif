import os
import sys
sys.path.append("../")
import numpy as np
import cv2
import matplotlib.pyplot as plt
import IPython.display as display
from utils.video_utils import save_video
from tqdm import tqdm
from glob import glob
from PIL import Image

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

class HandDetection():
    def __init__(self):
        # STEP 1: Load the MediaPipe Hands model.
        # STEP 2: Create an HandLandmarker object.
        base_options = python.BaseOptions(model_asset_path='../data_collection_scripts/hand_landmarker.task')
        options = vision.HandLandmarkerOptions(base_options=base_options,num_hands=2)
        self.detector = vision.HandLandmarker.create_from_options(options)

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
        annotated_image = np.copy(rgb_image)

        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]

            # Draw the hand landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())

            # Get the top left corner of the detected hand's bounding box.
            height, width, _ = annotated_image.shape
            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]
            text_x = int(min(x_coordinates) * width)
            text_y = int(min(y_coordinates) * height) - MARGIN

            # Draw handedness (left or right hand) on the image.
            cv2.putText(annotated_image, f"{handedness[0].category_name}",
                        (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                        FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

        return annotated_image

    def draw_trajectory_on_image(self, episode_idx, image_dir, output_dir, crop_info=None, save_all=False, save_last_frame=True, _save_video=True, hand=None, max_len=None, start=None):
        # frame_idx = 0
        episode_len = len(os.listdir(image_dir))
        if max_len is not None:
            episode_len = min(episode_len, max_len)
        if start is not None:
            start = min(max(start, 0), episode_len)
        motion_images = []
        latest_detection_result = None

        start_color = np.array([255, 255, 255])  # White
        end_color = np.array([0, 255, 0])  # Green
        transformed_pos_list = []
        detection_error_count = 0
        start_frame_idx = start #0

        traj_output_dir = os.path.join(output_dir, "traj{}".format(episode_idx))
        last_frame_output_dir = os.path.join(output_dir, "last_frame")
        if save_all:
            if not os.path.exists(traj_output_dir):
                os.makedirs(traj_output_dir)
        if not os.path.exists(last_frame_output_dir):
            os.makedirs(last_frame_output_dir)

        for frame_idx in range(start, episode_len):
            image_frame_path = os.path.join(image_dir, "image{}.jpg".format(frame_idx))

            # STEP 3: Load the input image.
            image = np.array(Image.open(image_frame_path))

            if crop_info is not None:
                image = image[crop_info[1]:crop_info[3], crop_info[0]:crop_info[2], :].copy()
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            else:
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image.copy())

            # STEP 4: Detect hand landmarks from the input image.
            detection_result = self.detector.detect(image)
            if detection_result is None or len(detection_result.hand_landmarks) == 0:
                if latest_detection_result is None:
                    print("No hand detected in the first frame. Skip the frame.")
                    start_frame_idx += 1
                    continue
                # print("No hand detected in the image. Use the latest detection result.")
                detection_error_count += 1
                detection_result = latest_detection_result
            else:
                latest_detection_result = detection_result

            # STEP 5: Process the classification result. In this case, visualize it.
            # only visualize the center of the hand
            annotated_image = np.copy(image.numpy_view())
            # average the landmarks
            if len(detection_result.hand_landmarks) == 1:
                hand_landmarks = detection_result.hand_landmarks[0]
                hand_landmarks = np.array([(landmark.x, landmark.y, landmark.z) for landmark in hand_landmarks])
                center = np.mean(hand_landmarks[:, :2], axis=0)
                center = center * np.array([image.width, image.height])
                center = center.astype(int)
            elif frame_idx < 10 and (hand is not None):
                handedness = detection_result.handedness
                if hand == "right":
                    hand_idx = 0 if handedness[0][0].category_name == "Right" else 1
                else:
                    hand_idx = 0 if handedness[0][0].category_name == "Left" else 1
                hand_landmarks = detection_result.hand_landmarks[hand_idx]
                hand_landmarks = np.array([(landmark.x, landmark.y, landmark.z) for landmark in hand_landmarks])
                center = np.mean(hand_landmarks[:, :2], axis=0)
                center = center * np.array([image.width, image.height])
                center = center.astype(int)
            else:
                hand_landmarks_list = [np.array([(landmark.x, landmark.y, landmark.z) for landmark in hand_landmarks]) for hand_landmarks in detection_result.hand_landmarks]
                center_list = [np.mean(hand_landmarks[:, :2], axis=0) for hand_landmarks in hand_landmarks_list]
                center_list = [center * np.array([image.width, image.height]) for center in center_list]
                center_list = [center.astype(int) for center in center_list]
                # choose the center that is closer to the previous center
                if len(transformed_pos_list) > 0:
                    prev_center = transformed_pos_list[-1]
                    min_dist = float("inf")
                    for center in center_list:
                        dist = np.linalg.norm(np.array(center) - np.array(prev_center))
                        if dist < min_dist:
                            min_dist = dist
                            closest_center = center
                    center = closest_center
                else:
                    center = center_list[0]
            
            # if center is far from the previous center more than 100 pixels, use the previous center
            if len(transformed_pos_list) > 0:
                prev_center = transformed_pos_list[-1]
                dist = np.linalg.norm(np.array(center) - np.array(prev_center))
                if dist > 300:
                    center = prev_center
            center = tuple(center)
            transformed_pos_list.append(center)

            # draw line
            for j in range(1, frame_idx-start_frame_idx+1):
                color_ratio = j / (episode_len - start - 1)
                line_color = (1 - color_ratio) * start_color + color_ratio * end_color
                cv2.line(annotated_image, (transformed_pos_list[j-1][0], transformed_pos_list[j-1][1]), (transformed_pos_list[j][0], transformed_pos_list[j][1]), line_color, 5)

            cv2.circle(annotated_image, center, 10, (255, 0, 0), -1)

            annotated_image = annotated_image[:, :, ::-1]
            motion_images.append(annotated_image)

            # save the image
            if save_all:
                output_image_path = os.path.join(traj_output_dir, "image{}.jpg".format(frame_idx - start_frame_idx))
                cv2.imwrite(output_image_path, annotated_image)

        print("detection error: {} / {}".format(detection_error_count, episode_len - start))

        # save the last frame
        if save_last_frame:
            last_frame_output_image_path = os.path.join(last_frame_output_dir, "traj{}.jpg".format(episode_idx))
            cv2.imwrite(last_frame_output_image_path, motion_images[-1])

        # save the video
        if _save_video:
            if "side" in image_dir:
                image_type = "color_sideview"
            else:
                image_type = "color"
            output_video_path = os.path.join(output_dir, "traj{}_{}.mp4".format(episode_idx, image_type))
            save_video(motion_images, output_video_path)

        episode_info_dict = {"num_steps": episode_len-start, "trajectory": transformed_pos_list, 
                             "last_frame_path": last_frame_output_image_path,
                                "video_path": output_video_path}
        return episode_info_dict
