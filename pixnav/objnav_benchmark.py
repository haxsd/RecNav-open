import habitat
import os
import random
import argparse
import csv
import json
import time
import cv2
import imageio
import numpy as np
import torch
from cv_utils.detection_tools import *
from tqdm import tqdm
from constants import *
from config_utils import hm3d_config
from gpt4v_planner import GPT4V_Planner
from policy_agent import Policy_Agent
from cv_utils.detection_tools import initialize_dino_model
from cv_utils.segmentation_tools import initialize_sam_model
from habitat.utils.visualizations.maps import colorize_draw_agent_and_fit_to_height
from habitat.config.read_write import read_write

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"
def write_metrics(metrics,path="objnav_hm3d.csv"):
    with open(path, mode="w", newline="") as csv_file:
        fieldnames = metrics[0].keys()
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)

def adjust_topdown(metrics):
    return cv2.cvtColor(colorize_draw_agent_and_fit_to_height(metrics['top_down_map'],1024),cv2.COLOR_BGR2RGB)

def record_observation(obs, habitat_env, recent_images, rgb_frames=None, topdown_frames=None):
    recent_images.append(obs['rgb'])
    if len(recent_images) > 12:
        recent_images.pop(0)
    if rgb_frames is not None:
        rgb_frames.append(obs['rgb'])
    if topdown_frames is not None:
        topdown_frames.append(adjust_topdown(habitat_env.get_metrics()))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_episodes",type=int,default=200)
    parser.add_argument("--seed",type=int,default=None,help="Random seed for reproducibility")
    parser.add_argument("--output_dir",type=str,default="./tmp",help="Output directory for trajectory videos")
    parser.add_argument("--csv_path",type=str,default=None,help="CSV output path (default: <output_dir>/objnav_hm3d.csv)")
    parser.add_argument("--save_rgb_video",dest="save_rgb_video",action="store_true",help="Write RGB trajectory videos")
    parser.add_argument("--no_save_rgb_video",dest="save_rgb_video",action="store_false",help="Disable RGB trajectory video writing")
    parser.add_argument("--save_topdown_video",dest="save_topdown_video",action="store_true",help="Write topdown metric videos")
    parser.add_argument("--no_save_topdown_video",dest="save_topdown_video",action="store_false",help="Disable topdown metric video writing")
    parser.add_argument("--save_planner_monitor",action="store_true",help="Save the planner panoramic image for debugging")
    parser.add_argument("--planner_image_scale",type=float,default=0.5,help="Scale factor for the planner panoramic image")
    parser.set_defaults(save_rgb_video=True, save_topdown_video=True)
    return parser.parse_known_args()[0]

def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

def detect_mask(image,category,detect_model):
    det_result = openset_detection(image,category,detect_model)
    if det_result.xyxy.shape[0] > 0:
        goal_image = image
        goal_mask_xyxy = det_result.xyxy[np.argmax(det_result.confidence)]
        goal_mask_x = int((goal_mask_xyxy[0]+goal_mask_xyxy[2])/2)
        goal_mask_y = int((goal_mask_xyxy[1]+goal_mask_xyxy[3])/2)
        goal_mask = np.zeros((goal_image.shape[0],goal_image.shape[1]),np.uint8)
        goal_mask = cv2.rectangle(goal_mask,(goal_mask_x-8,goal_mask_y-8),(goal_mask_x+8,goal_mask_y+8),(255,255,255),-1)
        return True,goal_image,goal_mask
    return False,[],[]


args = get_args()
set_seed(args.seed)
habitat_config = hm3d_config(stage='val',episodes=args.eval_episodes)
if args.seed is not None:
    with read_write(habitat_config):
        habitat_config.habitat.seed = args.seed
        habitat_config.habitat.environment.iterator_options.shuffle = False
os.makedirs(args.output_dir, exist_ok=True)
csv_path = args.csv_path or os.path.join(args.output_dir, "objnav_hm3d.csv")
habitat_env = habitat.Env(habitat_config)
detection_model = None
segmentation_model = None
try:
    detection_model = initialize_dino_model()
except Exception as exc:
    print(f"[WARN] GroundingDINO init failed, fallback planner will be used: {exc}")
try:
    segmentation_model = initialize_sam_model()
except Exception as exc:
    print(f"[WARN] SAM init failed, fallback planner will be used: {exc}")

nav_planner = GPT4V_Planner(
    detection_model,
    segmentation_model,
    save_monitor_image=args.save_planner_monitor,
    image_scale=args.planner_image_scale,
)
nav_executor = Policy_Agent(model_path=POLICY_CHECKPOINT)
evaluation_metrics = []

for i in tqdm(range(args.eval_episodes)):
    obs = habitat_env.reset()
    dir = os.path.join(args.output_dir, "trajectory_%d"%i)
    if args.save_rgb_video or args.save_topdown_video:
        os.makedirs(dir,exist_ok=True)
    heading_offset = 0

    nav_planner.reset(habitat_env.current_episode.object_category)
    recent_images = [obs['rgb']]
    rgb_frames = [obs['rgb']] if args.save_rgb_video else None
    topdown_frames = [adjust_topdown(habitat_env.get_metrics())] if args.save_topdown_video else None

    # a whole round planning process
    for _ in range(11):
        obs = habitat_env.step(3)
        record_observation(obs, habitat_env, recent_images, rgb_frames, topdown_frames)
    goal_image,goal_mask,debug_image,goal_rotate,goal_flag = nav_planner.make_plan(recent_images, return_debug_image=False)
    for j in range(min(11-goal_rotate,1+goal_rotate)):
        if goal_rotate <= 6:
            obs = habitat_env.step(3)
            record_observation(obs, habitat_env, recent_images, rgb_frames, topdown_frames)
        else:
            obs = habitat_env.step(2)
            record_observation(obs, habitat_env, recent_images, rgb_frames, topdown_frames)
    nav_executor.reset(goal_image,goal_mask)


    while not habitat_env.episode_over:
        action = nav_executor.step(obs['rgb'],habitat_env.sim.previous_step_collided,return_debug_image=False)
        if action != 0 or goal_flag:
            if action == 4:
                heading_offset += 1
            elif action == 5:
                heading_offset -= 1
            obs = habitat_env.step(action)
            record_observation(obs, habitat_env, recent_images, rgb_frames, topdown_frames)
        else:
            if habitat_env.episode_over:
                break
            
            for _ in range(0,abs(heading_offset)):
                if habitat_env.episode_over:
                    break
                if heading_offset > 0:
                    obs = habitat_env.step(5)
                    record_observation(obs, habitat_env, recent_images, rgb_frames, topdown_frames)
                    heading_offset -= 1
                elif heading_offset < 0:
                    obs = habitat_env.step(4)
                    record_observation(obs, habitat_env, recent_images, rgb_frames, topdown_frames)
                    heading_offset += 1
            
            # a whole round planning process
            for _ in range(11):
                if habitat_env.episode_over:
                    break
                obs = habitat_env.step(3)
                record_observation(obs, habitat_env, recent_images, rgb_frames, topdown_frames)
            goal_image,goal_mask,debug_image,goal_rotate,goal_flag = nav_planner.make_plan(recent_images, return_debug_image=False)
            for j in range(min(11-goal_rotate,goal_rotate+1)):
                if habitat_env.episode_over:
                    break
                if goal_rotate <= 6:
                    obs = habitat_env.step(3)
                    record_observation(obs, habitat_env, recent_images, rgb_frames, topdown_frames)
                else:
                    obs = habitat_env.step(2)
                    record_observation(obs, habitat_env, recent_images, rgb_frames, topdown_frames)
            nav_executor.reset(goal_image,goal_mask)
    
    if rgb_frames is not None:
        fps_writer = imageio.get_writer("%s/fps.mp4"%dir, fps=4)
        for image in rgb_frames:
            fps_writer.append_data(image)
        fps_writer.close()
    if topdown_frames is not None:
        topdown_writer = imageio.get_writer("%s/metric.mp4"%dir,fps=4)
        for topdown in topdown_frames:
            topdown_writer.append_data(topdown)
        topdown_writer.close()

    ep_metrics = habitat_env.get_metrics()
    evaluation_metrics.append({'episode_idx': i,
                               'episode_id': habitat_env.current_episode.episode_id,
                               'scene_id': habitat_env.current_episode.scene_id,
                               'success': ep_metrics['success'],
                               'spl': ep_metrics['spl'],
                               'soft_spl': ep_metrics['soft_spl'],
                               'distance_to_goal': ep_metrics['distance_to_goal'],
                               'object_goal': habitat_env.current_episode.object_category,
                               'seed': args.seed})
    write_metrics(evaluation_metrics, path=csv_path)
    

    

            

        

        


    
        
