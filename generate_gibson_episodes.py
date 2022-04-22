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
from habitat.utils.visualizations.utils import observations_to_image

from src.models.common import batch_obs
from utils.misc import write_json, write_gzip, save_image
from constants import coco_categories, coco_categories_to_mp3d, coco_categories_to_task_category_id

os.environ["OMP_NUM_THREADS"] = "1"


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
    for i in tqdm(range(num_iters)):
        obs, infos = envs.reset()
        # batch = batch_obs(obs, device=device)
        # sys.exit(1)

        # frame = observations_to_image({"rgb": batch["rgb"][0].cpu().numpy(), "semantic": batch["semantic"][0].cpu().numpy()}, {})
        # save_image(frame, "top_down_mp_{}.jpg".format(i))
        episodes = envs.current_episodes()

        for episode, info in zip(episodes, infos):
            if info["distance_to_goal"] > 2.0:
                scene_episode_map[episode["scene_id"]].append(episode)

    category_to_mp3d_category_id = {'chair': 3, 'table': 5, 'picture': 6, 'cabinet': 7, 'cushion': 8, 'sofa': 10, 'bed': 11, 'chest_of_drawers': 13, 'plant': 14, 'sink': 15, 'toilet': 18, 'stool': 19, 'towel': 20, 'tv_monitor': 22, 'shower': 23, 'bathtub': 25, 'counter': 26, 'fireplace': 27, 'gym_equipment': 33, 'seating': 34, 'clothes': 38}
    category_to_task_category_id = {'chair': 0, 'table': 1, 'picture': 2, 'cabinet': 3, 'cushion': 4, 'sofa': 5, 'bed': 6, 'chest_of_drawers': 7, 'plant': 8, 'sink': 9, 'toilet': 10, 'stool': 11, 'towel': 12, 'tv_monitor': 13, 'shower': 14, 'bathtub': 15, 'counter': 16, 'fireplace': 17, 'gym_equipment': 18, 'seating': 19, 'clothes': 20}
    category_to_mp3d_category_id.update(coco_categories_to_mp3d)
    category_to_task_category_id.update(coco_categories_to_task_category_id)

    for scene, episodes in scene_episode_map.items():
        scene_id = scene.split("/")[-1].split(".")[0]
        output_path = args.output_path.format(scene.split("/")[-1].replace(".glb", ".json"))
        print("Scene: {}, Episodes: {}".format(scene, len(episodes)))
        goal_spec_path = "data/datasets/objectnav/gibson/v1.1/train_generated/goals/{}_objs.pkl".format(scene_id)
        print(goal_spec_path)
        goals_by_category = pickle.load(open(goal_spec_path, "rb"))
        for key, value in goals_by_category.items():
            items = [attr.asdict(item) for item in value]
            goals_by_category[key] = items
        

        dataset = {
            "episodes": episodes,
            "goals_by_category": goals_by_category,
            "category_to_category_id": coco_categories,
            "category_to_mp3d_category_id": category_to_mp3d_category_id,
            "category_to_task_category_id": category_to_task_category_id

        }
        write_json(dataset, output_path)
        write_gzip(output_path, output_path)


if __name__ == "__main__":
    main()
