import os
import sys

import torch
import accelerate 
import logging
import hydra
from collections import Counter
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from diffusers.optimization import get_cosine_schedule_with_warmup
import numpy as np 
from torch import optim
from rich.progress import Progress, SpinnerColumn, BarColumn, MofNCompleteColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
import json

from utils.logging import setup_logging
import random

import tensorflow_hub as hub

from inference.multistep_sequences import get_sequences
from inference.utils_infer import get_env_state_for_initial_condition, join_vis_lang
from inference.rollout_video import RolloutVideo

from calvin_env.envs.play_table_env import get_env

logger = logging.getLogger(__name__)

def get_video_tag(i):
    # if dist.is_available() and dist.is_initialized():
    #     i = i * dist.get_world_size() + dist.get_rank()
    return f"sequence_{i}"

def count_success(results):
    count = Counter(results)
    step_success = []
    for i in range(1, 6):
        n_success = sum(count[j] for j in reversed(range(i, 6)))
        sr = n_success / len(results)
        step_success.append(sr)
    return step_success


def print_and_save(results, cfg, log_dir):
    
    sequences = get_sequences(cfg.inference.num_sequences)

    current_data = {}
    avg_seq_len = np.mean(results)
    chain_sr = {i + 1: sr for i, sr in enumerate(count_success(results))}
    logger.info(f"Average successful sequence length: {avg_seq_len}")
    logger.info("Success rates for i instructions in a row:")
    for i, sr in chain_sr.items():
        logger.info(f"{i}: {sr:.3f}%")

    cnt_success = Counter()
    cnt_fail = Counter()

    for result, (_, sequence) in zip(results, sequences):
        for successful_tasks in sequence[:result]:
            cnt_success[successful_tasks] += 1
        if result < len(sequence):
            failed_task = sequence[result]
            cnt_fail[failed_task] += 1

    total = cnt_success + cnt_fail
    
    task_info = {}
    for task in total:
        task_info[task] = {"success": cnt_success[task], "total": total[task]}
        logger.info(f"{task}: {cnt_success[task]} / {total[task]} |  SR: {cnt_success[task] / total[task] * 100:.1f}%")

    data = {"avg_seq_len": avg_seq_len, "chain_sr": chain_sr, "task_info": task_info}
    # wandb.log({"avrg_performance/avg_seq_len": avg_seq_len, "avrg_performance/chain_sr": chain_sr, "detailed_metrics/task_info": task_info})
    current_data = data

    json_data = current_data
    with open(os.path.join(log_dir, "results.json"), "w") as file:
        json.dump(json_data, file, indent=4)



def evaluate_policy(cfg, model, env, accelerator, log_dir):
    progress = Progress(
        TextColumn("{task.description}"),
        SpinnerColumn(),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        disable=not accelerator.is_main_process,
    ) 
    progress.start()

    task_oracle = hydra.utils.instantiate(cfg.task)
    eval_sequences = get_sequences(cfg.inference.num_sequences)
    results = []
    plans = []
    
    eval_task = progress.add_task("Evaluating sequences", total=len(eval_sequences))
    

    #####
    val_annotations = cfg.annotation
    lang_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    lang_embeddings = {}
    for task, annotation in val_annotations.items():
        lang_embeddings[task] = lang_model([annotation[0]]).numpy()
        # logger.info(f"Loaded language embedding for task {task}: {annotation} with shape {lang_embeddings[task].shape}")
    
    rollout_video = RolloutVideo(
            logger=logger,
            empty_cache=False,
            log_to_file=True,
            save_dir=os.path.join(log_dir, "rollout_videos"),
            resolution_scale=1,
        )

    rollout_video2 = RolloutVideo(
            logger=logger,
            empty_cache=False,
            log_to_file=True,
            save_dir=os.path.join(log_dir, "flow_videos"),
            resolution_scale=1,
        )

    record = True
    ###
    for i, (initial_state, eval_sequence) in enumerate(eval_sequences):
        result = evaluate_sequence(
            env, model, task_oracle, initial_state, eval_sequence, lang_embeddings, val_annotations, progress, cfg, record, rollout_video, rollout_video2, i
        )
        results.append(result)
        
        success_rates = count_success(results)
        average_rate = sum(success_rates) / len(success_rates) * 5
        description = " ".join([f"{i + 1}/5 : {v:.3f}% |" for i, v in enumerate(success_rates)])
        description += f" Average: {average_rate:.1f} |"    
        logger.info(description)

        if record:
            rollout_video.write_to_tmp()
            rollout_video._log_currentvideos_to_file(i, result, save_as_video=True)
            rollout_video2.write_to_tmp()
            rollout_video2._log_currentvideos_to_file(i, result, save_as_video=True)

        progress.update(eval_task, advance=1)

    progress.stop()
    return results

def evaluate_sequence(
    env, model, task_checker, initial_state, eval_sequence, lang_embeddings, val_annotations, progress, cfg, record, rollout_video, rollout_video2, i
):
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    if record:
        caption = " | ".join(eval_sequence)
        rollout_video.new_video(tag=get_video_tag(i), caption=caption)
        rollout_video2.new_video(tag=get_video_tag(i), caption=caption)
    success_counter = 0
    if cfg.debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    for idx, subtask in enumerate(eval_sequence):
        if record:
            rollout_video.new_subtask()
            rollout_video2.new_subtask()
        # success = random.randint(0, 1)
        success = rollout(env, model, task_checker, cfg, idx, subtask, lang_embeddings, val_annotations, progress, record, rollout_video, rollout_video2)
        if record:
            rollout_video.draw_outcome(success)
            rollout_video2.draw_outcome(success)
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter

def get_transform(image_size=128):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    return A.Compose([
        A.Resize(image_size, image_size),
        ToTensorV2(),
    ])
def rollout(env, model, task_oracle, cfg, idx, subtask, lang_embeddings, val_annotations, progress, record=False, rollout_video=None, rollout_video2=None):
    if cfg.debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
    obs = env.get_obs()
    # get lang annotation for subtask
    lang_annotation = val_annotations[subtask][0]
    # get language goal embedding

    goal = lang_embeddings[subtask]
    # goal['lang_text'] = val_annotations[subtask][0]
    # model.reset()
    start_info = env.get_info()

    transform = get_transform(cfg.inference.image_size)
    device = next(model.parameters()).device

    bar = progress.add_task(f"Rollout {idx}. {subtask}", total=cfg.inference.ep_len)
    for step in range(cfg.inference.ep_len):
        # action = torch.rand(7)
        inputs = {
            "rgb_static": transform(image=obs["rgb_obs"]["rgb_static"])["image"][None, None, ].to(device) / 255. ,
            "rgb_gripper": transform(image=obs["rgb_obs"]["rgb_gripper"])["image"][None, None].to(device) / 255. ,
            "language": lang_annotation, 
            "language_embedding": torch.tensor(goal).to(device),
        }
        action_output = model.step(inputs)
        action = action_output["action"]
        viz_flow = action_output["viz_flow"]
        # logger.info(f"Step {step + 1}: {action}")
        action[:-1] = action[:-1].clamp(-1, 1)  # Clamp action to [-1, 1]
        action[-1] = (action[-1] > 0).long() * 2 - 1
        #print('obs_max:',obs["rgb_obs"]['cond_static'].max())
        #print('obs_shape:', obs["rgb_obs"]['cond_static'].shape)
        obs, _, _, current_info = env.step(action)
        if cfg.debug:
            img = env.render(mode="rgb_array")
            join_vis_lang(img, lang_annotation)
            # time.sleep(0.1)
        if record:
            # update video
            rollout_video.update(obs["rgb_obs"]["rgb_static"])
            rollout_video2.update(viz_flow)
        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            model.reset()  # Reset model for next task
            progress.update(bar, advance=cfg.inference.ep_len - step)
            progress.remove_task(bar)

            if cfg.debug:
                print(colored("success", "green"), end=" ")
            if record:
                rollout_video.add_language_instruction(lang_annotation)
                rollout_video2.add_language_instruction(lang_annotation)
            return True

        else:
            progress.update(bar, advance=1)
    
    progress.remove_task(bar)
    logger.info(f"Failed to solve task {subtask}:{lang_annotation} in sequence {idx}.")
    if cfg.debug:
        print(colored("fail", "red"), end=" ")
    if record:
        rollout_video.add_language_instruction(lang_annotation)
        rollout_video2.add_language_instruction(lang_annotation)
    return False


def main(cfg):
    accelerator = accelerate.Accelerator(**cfg.accelerator)

    # accelerator.init_trackers(
    #     project_name=cfg.project,
    #     config=OmegaConf.to_container(cfg, resolve=True)
    # )
    accelerate.utils.set_seed(cfg.seed)

    log_dir = cfg.inference.save_dir
    setup_logging(accelerator.is_main_process, log_dir=log_dir)

    logger.info("Configuration:\n" + OmegaConf.to_yaml(cfg))

    model = hydra.utils.instantiate(cfg.model)

    if cfg.weights:
        logger.info(f"Loading model weights from {cfg.weights}.")
        logger.info(model.load_state_dict(torch.load(cfg.weights, map_location="cpu"), strict=False))
        # model.load_weights()
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix")
        model.imagine_model.vae = vae
        model.imagine_model.vae.requires_grad_(False)

    model = accelerator.prepare(model)
    model.eval()

    env = get_env(cfg.inference.dataset, show_gui=False)
    
    results = evaluate_policy(cfg, model, env, accelerator, log_dir=log_dir)
    print_and_save(results, cfg, log_dir=log_dir)

if __name__ == "__main__":
    with hydra.initialize(config_path="configs"):
        cfg = hydra.compose(config_name="infer", overrides=sys.argv[2:])
        OmegaConf.resolve(cfg)
        main(cfg)