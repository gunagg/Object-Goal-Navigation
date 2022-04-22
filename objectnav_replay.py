from collections import deque, defaultdict
import attr
import os
import pickle
import logging
import sys
import torch
import numpy as np

from envs import make_vec_envs
from arguments import get_args
from tqdm import tqdm

from habitat import logger
from habitat.utils.visualizations.utils import observations_to_image, images_to_video

from src.models.common import batch_obs
from utils.misc import write_json, write_gzip, save_image
from constants import coco_categories, coco_categories_to_mp3d, coco_categories_to_task_category_id

os.environ["OMP_NUM_THREADS"] = "1"

def make_videos(observations_list, output_prefix, ep_id):
    #print(observations_list[0][0].keys(), type(observations_list[0][0]))
    prefix = output_prefix + "_{}".format(ep_id)
    # make_video_cv2(observations_list[0], prefix=prefix, open_vid=False)
    images_to_video(observations_list[0], output_dir="demos", video_name=prefix)

def main():
    args = get_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Setup Logging
    log_dir = "{}/models/{}/".format(args.dump_location, args.exp_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(
        filename=log_dir + 'train.log',
        level=logging.INFO)
    print("Dumping at {}".format(log_dir))
    print(args)
    logger.info(args)

    args.device = torch.device("cuda:0" if args.cuda else "cpu")
    device = args.device

    # Starting environments
    torch.set_num_threads(1)
    envs = make_vec_envs(args)

    scene_episode_map = defaultdict(list)

    num_scenes = args.num_scenes
    num_iters = args.num_train_episodes * args.num_scenes // args.num_processes
    possible_actions = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "LOOK_UP", "LOOK_DOWN"]
    obs, infos = envs.reset()
    aggregate_metric = {
        "distance_to_goal": 0,
        "spl": 0,
        "success": 0
    }
    count = 0
    for i in tqdm(range(num_iters)):
        episodes = envs.current_episodes()
        episode = episodes[0]
        count += envs.num_envs
        #print(infos, len(episode["reference_replay"]))
        #print("Episode id: {}".format(episode["episode_id"]))
        
        step_infos = None
        observation_list = []
        for step in range(len(episode["reference_replay"][1:])):
            step_data = episode["reference_replay"][step + 1]
            action = possible_actions.index(step_data["action"])
            actions = torch.tensor([action])
            # print(actions, step_data["action"], step + 1)
            obs, _, done, infos = envs.step(actions)
            step_infos = infos[0]

            frame = observations_to_image({"rgb": obs[0]["rgb"]}, {})
            observation_list.append(frame)
        print("after ep end", step_infos)

        aggregate_metric["success"] += step_infos["success"]
        aggregate_metric["spl"] += step_infos["spl"]
        aggregate_metric["distance_to_goal"] += step_infos["distance_to_goal"]

        make_videos([observation_list], "demos", i)
    aggregate_metric = {k: v / count for k, v in aggregate_metric.items()}

    print(aggregate_metric)


if __name__ == "__main__":
    main()
