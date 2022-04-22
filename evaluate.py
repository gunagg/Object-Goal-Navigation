from collections import deque, defaultdict
import os
import logging
import time
import json
import gym
import torch.nn as nn
import torch
import numpy as np

from model import RL_Policy, Semantic_Mapping
from utils.storage import GlobalRolloutStorage
from envs import make_vec_envs
from arguments import get_args
import algo

from habitat import logger

from src import POLICY_CLASSES
from src.default import get_config
from src.models.common import batch_obs
from src.models.rednet import load_rednet

from gym.spaces import Discrete, Dict, Box

os.environ["OMP_NUM_THREADS"] = "1"


def main():
    args = get_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Setup Logging
    log_dir = "{}/models/{}/".format(args.dump_location, args.exp_name)
    dump_dir = "{}/dump/{}/".format(args.dump_location, args.exp_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    logging.basicConfig(
        filename=log_dir + 'train.log',
        level=logging.INFO)
    print("Dumping at {}".format(log_dir))
    print(args)
    logger.info(args)

    # Logging and loss variables
    num_scenes = args.num_processes
    num_episodes = int(args.num_eval_episodes)
    device = args.device = torch.device("cuda:0" if args.cuda else "cpu")

    g_masks = torch.ones(num_scenes).float().to(device)

    best_g_reward = -np.inf

    if args.eval:
        episode_success = []
        episode_spl = []
        episode_dist = []
        for _ in range(args.num_processes):
            episode_success.append(deque(maxlen=num_episodes))
            episode_spl.append(deque(maxlen=num_episodes))
            episode_dist.append(deque(maxlen=num_episodes))

    else:
        episode_success = deque(maxlen=1000)
        episode_spl = deque(maxlen=1000)
        episode_dist = deque(maxlen=1000)

    finished = np.zeros((args.num_processes))
    wait_env = np.zeros((args.num_processes))

    g_episode_rewards = deque(maxlen=1000)

    g_value_losses = deque(maxlen=1000)
    g_action_losses = deque(maxlen=1000)
    g_dist_entropies = deque(maxlen=1000)

    per_step_g_rewards = deque(maxlen=1000)

    g_process_rewards = np.zeros((num_scenes))

    # Starting environments
    torch.set_num_threads(1)
    envs = make_vec_envs(args)
    obs, infos = envs.reset()
    batch = batch_obs(obs, device=device)

    torch.set_grad_enabled(False)

    # Global policy observation space
    ngc = 8 + args.num_sem_categories
    # g_observation_space = envs.observation_spaces[0]

    # Global policy action space
    g_action_space = gym.spaces.Box(low=0.0, high=0.99,
                                    shape=(2,), dtype=np.float32)

    # Global policy recurrent layer size
    g_hidden_size = args.global_hidden_size

    start = time.time()
    g_reward = 0

    torch.set_grad_enabled(False)
    spl_per_category = defaultdict(list)
    success_per_category = defaultdict(list)

    # Set up HabOnWeb models
    config_paths = "envs/habitat/configs/tasks/objectnav_gibson.yaml"
    DEFAULT_CONFIG = "habitat_configs/il_objectnav_sem_seg_ft.yaml"
    config = get_config([DEFAULT_CONFIG, DEFAULT_CONFIG],
                ['BASE_TASK_CONFIG_PATH', config_paths]).clone()

    ckpt_dict = torch.load(args.load, map_location=device)["state_dict"]
    ckpt_dict = {
        k.replace("model.", ""): v
        for k, v in ckpt_dict.items()
    }
    ckpt_dict = {
        k.replace("module.", ""): v
        for k, v in ckpt_dict.items()
    }

    # Config
    config = config
    config = config.clone()
    model_cfg = config.MODEL
    il_cfg = config.IL.BehaviorCloning
    task_cfg = config.TASK_CONFIG.TASK

    # Load spaces (manually)
    spaces = {
        "objectgoal": Box(
            low=0, high=20, # from matterport dataset
            shape=(1,),
            dtype=np.int64
        ),
        "depth": Box(
            low=0,
            high=1,
            shape=(config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT,
                    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH, 1),
            dtype=np.float32,
        ),
        "rgb": Box(
            low=0,
            high=255,
            shape=(config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT,
                    config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH, 3),
            dtype=np.uint8,
        ),
        "gps": Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(2,), # Spoof for model to be shaped correctly
            dtype=np.float32,
        ),
        "compass": Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float)
    }

    observation_spaces = Dict(spaces)
    if 'action_distribution.linear.bias' in ckpt_dict:
        num_acts = ckpt_dict['action_distribution.linear.bias'].size(0)
    action_spaces = Discrete(num_acts)

    is_objectnav = "ObjectNav" in task_cfg.TYPE
    additional_sensors = []
    embed_goal = False
    if is_objectnav:
        additional_sensors = ["gps", "compass"]
        embed_goal = True

    policy_class = POLICY_CLASSES[il_cfg.POLICY.name]
    model = policy_class(
        observation_space=observation_spaces,
        action_space=action_spaces,
        model_config=model_cfg,
        device=device,
        goal_sensor_uuid="objectgoal",
        additional_sensors=additional_sensors,
    ).to(device)

    model.load_state_dict(ckpt_dict, strict=True)
    model.eval()

    semantic_predictor = None

    if model_cfg.USE_SEMANTICS:
        # logger.info("setting up sem seg predictor")
        semantic_predictor = load_rednet(
            device,
            ckpt=model_cfg.SEMANTIC_ENCODER.rednet_ckpt,
            resize=True, # since we train on half-vision
            num_classes=model_cfg.SEMANTIC_ENCODER.num_classes
        )
        semantic_predictor.eval()
    logger.info("Model setup Done!")

    # Load other items
    test_recurrent_hidden_states = torch.zeros(
        model_cfg.STATE_ENCODER.num_recurrent_layers,
        1, # num_processes
        model_cfg.STATE_ENCODER.hidden_size,
        device=device,
    )
    done = torch.zeros(args.num_processes, device=device, dtype=torch.bool)
    not_done_masks = torch.zeros(args.num_processes, 1, device=device, dtype=torch.bool)
    prev_actions = torch.zeros(
        args.num_processes, 1, dtype=torch.long, device=device
    )
    episode_dones = 0

    while(episode_dones < args.num_eval_episodes):
        if finished.sum() == args.num_processes:
            break

        for e, x in enumerate(done):
            if x:
                spl = infos[e]['spl']
                success = infos[e]['success']
                dist = infos[e]['distance_to_goal']
                spl_per_category[infos[e]['goal_name']].append(spl)
                success_per_category[infos[e]['goal_name']].append(success)
                if args.eval:
                    episode_success[e].append(success)
                    episode_spl[e].append(spl)
                    episode_dist[e].append(dist)
                    if len(episode_success[e]) == num_episodes:
                        finished[e] = 1
                else:
                    episode_success.append(success)
                    episode_spl.append(spl)
                    episode_dist.append(dist)
                episode_dones += 1
                wait_env[e] = 1.

        with torch.no_grad():
            if semantic_predictor is not None:
                batch["semantic"] = semantic_predictor(batch["rgb"], batch["depth"])
                if model_cfg.SEMANTIC_ENCODER.is_thda:
                    batch["semantic"] = batch["semantic"] - 1
            (
                logits,
                test_recurrent_hidden_states,
            ) = model(
                batch,
                test_recurrent_hidden_states,
                prev_actions,
                not_done_masks,
            )

            actions = torch.argmax(logits, dim=1)
            prev_actions.copy_(actions.unsqueeze(1))  # type: ignore


        obs, _, done, infos = envs.step(actions)
        batch = batch_obs(obs, device=device)


        not_done_masks = torch.tensor(
            [[0.0] if x else [1.0] for x in done],
            dtype=torch.float,
            device=device,
        )

        log = ""
        for i in range(args.num_processes):
            # episode ended
            if not_done_masks[i].item() == 0:
                #print("Episode done: {}".format(len(total_success)))
                total_success = []
                total_spl = []
                total_dist = []
                for e in range(args.num_processes):
                    for acc in episode_success[e]:
                        total_success.append(acc)
                    for dist in episode_dist[e]:
                        total_dist.append(dist)
                    for spl in episode_spl[e]:
                        total_spl.append(spl)

                logger.info("Episode done: {}".format(len(total_spl)))
                if (episode_dones + 1) % 10 == 0:
                    log += " ObjectNav succ/spl/dtg:"
                    log += " {:.3f}/{:.3f}/{:.3f}({:.0f}),".format(
                        np.mean(total_success),
                        np.mean(total_spl),
                        np.mean(total_dist),
                        len(total_spl))
                    logger.info(log)
            # else:
            #     if episode_dones % 10 == 0:
            #         total_success = []
            #         total_spl = []
            #         total_dist = []
            #         for e in range(args.num_processes):
            #             for acc in episode_success[e]:
            #                 total_success.append(acc)
            #             for dist in episode_dist[e]:
            #                 total_dist.append(dist)
            #             for spl in episode_spl[e]:
            #                 total_spl.append(spl)
            #         log += " ObjectNav succ/spl/dtg:"
            #         log += " {:.3f}/{:.3f}/{:.3f}({:.0f}),".format(
            #             np.mean(total_success),
            #             np.mean(total_spl),
            #             np.mean(total_dist),
            #             len(total_spl))
            if len(log) > 0:
                logger.info(log)


    # Print and save model performance numbers during evaluation
    if args.eval:
        print("Dumping eval details...")
        
        total_success = []
        total_spl = []
        total_dist = []
        for e in range(args.num_processes):
            for acc in episode_success[e]:
                total_success.append(acc)
            for dist in episode_dist[e]:
                total_dist.append(dist)
            for spl in episode_spl[e]:
                total_spl.append(spl)

        if len(total_spl) > 0:
            log = "Final ObjectNav succ/spl/dtg:"
            log += " {:.3f}/{:.3f}/{:.3f}({:.0f}),".format(
                np.mean(total_success),
                np.mean(total_spl),
                np.mean(total_dist),
                len(total_spl))

        print(log)
        logger.info(log)
            
        # Save the spl per category
        log = "Success | SPL per category\n"
        for key in success_per_category:
            log += "{}: {} | {}\n".format(key,
                                          sum(success_per_category[key]) /
                                          len(success_per_category[key]),
                                          sum(spl_per_category[key]) /
                                          len(spl_per_category[key]))

        print(log)
        logger.info(log)

        with open('{}/{}_spl_per_cat_pred_thr.json'.format(
                dump_dir, args.split), 'w') as f:
            json.dump(spl_per_category, f)

        with open('{}/{}_success_per_cat_pred_thr.json'.format(
                dump_dir, args.split), 'w') as f:
            json.dump(success_per_category, f)


if __name__ == "__main__":
    main()