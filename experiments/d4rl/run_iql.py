import os
import sys
import copy
import itertools
import datetime
import random
from functools import reduce
from pathlib import Path

import gym
from experiments import get_launch_args, sweep_with_devices, launch, exp_id, parse_task_ids, launch_jobs, get_local_storage_path

import multiprocessing

def generate_mixed():

  env_names = [
    "ant",
    "hopper",
    "halfcheetah",
    "walker2d",
  ]

  levels = [
    "random-medium",
    "random-expert",
  ]

  dataset_ratios = [
    "0.01",
    "0.05",
    "0.1",    
    "0.5",
  ]

  return list(map(lambda t: dict(
      env_name=f"{t[0]}-{t[1]}-{t[2]}-v2",
      config="implicit_q_learning/configs/mujoco_config.py"),
      itertools.product(env_names, levels, dataset_ratios)))

def generate_mujoco_regular():

  dataset_types = [
    'random-v2',
    'medium-expert-v2',
    'medium-replay-v2',
    'full-replay-v2',
    'medium-v2',
    'expert-v2',
  ]

  envs = [
    'hopper',
    'halfcheetah',
    'ant',
    'walker2d'
  ]

  return list(map(lambda t: dict(
      env_name=f"{t[0]}-{t[1]}",
      config="implicit_q_learning/configs/mujoco_config.py"),
      itertools.product(envs, dataset_types)))

def generate_antmaze_regular():

  dataset_types = [
    "umaze-v0",
    "umaze-diverse-v0",
    "medium-diverse-v0",
    "medium-play-v0",
    "large-diverse-v0",
    "large-play-v0",
  ]

  envs = [
    "antmaze",
  ]

  return list(map(lambda t: dict(
      env_name=f"{t[0]}-{t[1]}",
      config="implicit_q_learning/configs/antmaze_config.py"),
      itertools.product(envs, dataset_types)))

def generate_maze_regular():

  dataset_types = [
    "umaze-v1",    
    "medium-v1",
    "large-v1",
    "open-v0",
    "open-dense-v0",
    "umaze-dense-v1",
    "medium-dense-v1",
    "large-dense-v1",
  ]

  envs = [
    "maze2d",
  ]

  return list(map(lambda t: dict(
      env_name=f"{t[0]}-{t[1]}",
      config="implicit_q_learning/configs/antmaze_config.py"),
      itertools.product(envs, dataset_types)))

def generate_hand_regular():
  objects = [
    "pen",
    "hammer",
    "door",
    "relocate",
  ]

  types = [
    "human-v1",
    "cloned-v1",
    "expert-v1",
  ]

  return list(map(lambda t: dict(
      env_name=f"{t[0]}-{t[1]}",
      config="implicit_q_learning/configs/mujoco_config.py"),
      itertools.product(objects, types)))

def generate_kitchen_regular():

  env_names = [
    "kitchen-complete-v0",
    "kitchen-partial-v0",
    "kitchen-mixed-v0",
  ]

  return list(map(lambda t: dict(
      env_name=f"{t}",
      config="implicit_q_learning/configs/kitchen_config.py"),
      env_names))

if __name__ == '__main__':
  experiment = f"{os.path.basename(os.path.dirname(Path(__file__)))}_iql"
  launch_args = get_launch_args(experiment)

  # Hyperparameters
  datasets_args = list(reduce(lambda a, b: a + b,
    [
      generate_mixed(),
      generate_mujoco_regular(),
      generate_antmaze_regular(),
      generate_maze_regular(),
      generate_hand_regular(),
      generate_kitchen_regular(),
    ]
  ))

  algos_scripts = [
    {"script": "CUDA_VISIBLE_DEVICES={gpu} python3 implicit_q_learning/train_offline.py",},
  ]

  samplers = [
    "uniform",
    *[f"Top-{p}" for p in [0.1]],
    *[f"RW-{alpha}" for alpha in [0.1]],
    *[f"AW-{alpha}" for alpha in ["0.1"]],
  ]

  seeds = [
    100,
    200,
    300,
    400,
    500
  ]

  common_exp_args = [
    "--project offline-subopt-iql",    
  ]

  if launch_args.debug:
    common_exp_args.append("--offline")
  else:
    common_exp_args.append("--track")

  
  all_job_args = []
  for job_idx, (n_tasks, device, 
          dataset_arg,
          algo_script,
          sampler,
          seed) in enumerate(
          sweep_with_devices(itertools.product(
            datasets_args,        
            algos_scripts,
            samplers,
            seeds),
            devices=launch_args.gpus,
            n_jobs=launch_args.n_jobs,
            n_parallel_task=1, shuffle=True)):
    job_args = []
    for task_idx in range(n_tasks):
      args = [
        algo_script[task_idx]["script"].format(gpu=device),
      ] + common_exp_args
      for k, v in dataset_arg[task_idx].items():
        args.append(f"--{k} {v}")

      args.append(f"--sampler {sampler[task_idx]}")
      args.append(f"--seed {seed[task_idx]}")
      args.append(f'--save_dir {get_local_storage_path()}/results/IQL/{dataset_arg[task_idx]["env_name"]}/{sampler[task_idx]}/{seed[task_idx]}')
      job_args.append(" ".join(args))
    all_job_args.append(job_args[0])

    if launch_args.debug:
      break

  print(f"Total: {len(all_job_args)}")

  launch_jobs(experiment,
    all_job_args,
    *parse_task_ids(launch_args.task_id),
    n_jobs=launch_args.n_jobs,
    mode=launch_args.mode,
    script="")

  print(f"Total: {len(all_job_args)}, num_gpus={len(launch_args.gpus)}")

