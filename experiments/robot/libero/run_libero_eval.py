"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.

Usage:
    # OpenVLA:
    # IMPORTANT: Set `center_crop=True` if model is fine-tuned with augmentations
    python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
        --center_crop [ True | False ] \
        --run_id_note <OPTIONAL TAG TO INSERT INTO RUN ID FOR LOGGING> \
        --use_wandb [ True | False ] \
        --wandb_project <PROJECT> \
        --wandb_entity <ENTITY>
"""

import csv
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

import wandb

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.contact_metrics import (
    FirstContactTracker,
    aggregate_metrics,
    get_distractor_objects,
    get_gripper_tip_geoms,
    get_target_object,
    get_task_object_geoms,
    make_episode_metrics,
    update_first_contact_tracker,
)
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    task_start: int = 0                              # First task id to evaluate, inclusive
    task_end: Optional[int] = None                   # Last task id to evaluate, exclusive; defaults to all tasks
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs
    output_dir: str = "./experiments/logs/libero_eval_results"  # Directory for results.csv and summary.json
    save_videos: bool = False                        # Save rollout MP4s under output_dir/rollouts
    save_video_successes_per_task: int = 3            # Max successful rollout videos to keep per task
    save_video_failures_per_task: int = 3             # Max failed rollout videos to keep per task
    compute_rsa: bool = False                         # Compute referent selection accuracy
    compute_fca: bool = False                         # Compute first contact accuracy
    contact_debug: bool = False                       # Print first-contact debug lines

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    assert cfg.task_suite_name in ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"], (
        f"Unsupported task suite: {cfg.task_suite_name}"
    )
    assert cfg.num_trials_per_task > 0, "cfg.num_trials_per_task must be > 0"
    checkpoint_path = Path(cfg.pretrained_checkpoint).expanduser()
    assert checkpoint_path.exists(), (
        "cfg.pretrained_checkpoint must be a local checkpoint directory or file. "
        f"Refusing to load from non-local path: {cfg.pretrained_checkpoint}"
    )
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name

    # Load model
    model = get_model(cfg)

    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family == "openvla":
        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    # Initialize local logging
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    os.makedirs(cfg.output_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")
    results_csv_path = os.path.join(cfg.output_dir, "results.csv")
    summary_json_path = os.path.join(cfg.output_dir, "summary.json")
    episode_metrics_csv_path = os.path.join(cfg.output_dir, "episode_metrics.csv")
    episode_metrics_jsonl_path = os.path.join(cfg.output_dir, "episode_metrics.jsonl")
    summary_metrics_json_path = os.path.join(cfg.output_dir, "summary_metrics.json")
    summary_metrics_md_path = os.path.join(cfg.output_dir, "summary_metrics.md")
    geom_mapping_json_path = os.path.join(cfg.output_dir, "geom_name_mapping.json")
    print(f"Writing CSV results to: {results_csv_path}")
    print(f"Writing summary JSON to: {summary_json_path}")
    if cfg.compute_rsa or cfg.compute_fca:
        print(f"Writing episode metrics to: {episode_metrics_csv_path}")
        print(f"Writing summary metrics to: {summary_metrics_json_path}")
        with open(episode_metrics_jsonl_path, "w"):
            pass

    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    task_end = cfg.task_end if cfg.task_end is not None else num_tasks_in_suite
    assert 0 <= cfg.task_start < task_end <= num_tasks_in_suite, (
        f"Invalid task range [{cfg.task_start}, {task_end}) for suite {cfg.task_suite_name} "
        f"with {num_tasks_in_suite} tasks"
    )
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    result_rows = []
    episode_metrics_rows = []
    task_summaries = {}
    task_metric_rows = {}
    task_geom_mappings = {}
    max_steps_by_suite = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }
    for task_id in tqdm.tqdm(range(cfg.task_start, task_end)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)
        object_geoms = get_task_object_geoms(env)
        target_object = get_target_object(env)
        distractor_objects = get_distractor_objects(target_object, object_geoms)
        task_geom_mappings[str(task_id)] = {
            "task_language": task_description,
            "target_object": target_object,
            "distractor_objects": distractor_objects,
            "gripper_tip_geoms": get_gripper_tip_geoms(env),
            "object_geoms": object_geoms,
        }
        if cfg.compute_rsa or cfg.compute_fca:
            print(
                f"[contact-metrics] task={task_id} target_object={target_object} "
                f"distractors={distractor_objects}"
            )
            log_file.write(
                f"[contact-metrics] task={task_id} target_object={target_object} "
                f"distractors={distractor_objects}\n"
            )

        # Start episodes
        task_episodes, task_successes = 0, 0
        task_saved_success_videos, task_saved_failure_videos = 0, 0
        task_metric_rows[str(task_id)] = []
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            # Setup
            t = 0
            replay_images = []
            max_steps = max_steps_by_suite[cfg.task_suite_name]
            done = False
            error_message = ""
            video_path = ""
            contact_tracker = FirstContactTracker(target_object=target_object, object_geoms=object_geoms)

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            try:
                # Reset environment
                env.reset()

                # Set initial states
                if episode_idx >= len(initial_states):
                    raise IndexError(
                        f"Requested trial {episode_idx}, but task {task_id} only has "
                        f"{len(initial_states)} benchmark initial states"
                    )
                obs = env.set_init_state(initial_states[episode_idx])
                if cfg.compute_rsa or cfg.compute_fca:
                    update_first_contact_tracker(contact_tracker, env, t)

                while t < max_steps + cfg.num_steps_wait:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        if cfg.compute_rsa or cfg.compute_fca:
                            update_first_contact_tracker(contact_tracker, env, t)
                        t += 1
                        continue

                    # Get preprocessed image
                    img = get_libero_image(obs, resize_size)

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    # Prepare observations dict
                    # Note: OpenVLA does not take proprio state as input
                    observation = {
                        "full_image": img,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }

                    # Query model to get action
                    action = get_action(
                        cfg,
                        model,
                        observation,
                        task_description,
                        processor=processor,
                    )

                    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
                    action = normalize_gripper_action(action, binarize=True)

                    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
                    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
                    if cfg.model_family == "openvla":
                        action = invert_gripper_action(action)

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if cfg.compute_rsa or cfg.compute_fca:
                        update_first_contact_tracker(contact_tracker, env, t)
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

            except Exception as e:
                error_message = repr(e)
                print(f"Caught exception: {error_message}")
                log_file.write(f"Caught exception: {error_message}\n")

            task_episodes += 1
            total_episodes += 1
            episode_metrics = {}
            if cfg.compute_rsa or cfg.compute_fca:
                episode_metrics = make_episode_metrics(
                    episode_index=total_episodes,
                    seed=cfg.seed,
                    task_id=task_id,
                    task_name=getattr(task, "name", str(task_id)),
                    instruction=task_description,
                    target_object=target_object,
                    distractor_objects=distractor_objects,
                    tracker=contact_tracker,
                    success=bool(done),
                    episode_length=t,
                    error_message=error_message,
                    rsa_definition="first_contact_proxy",
                )
                task_metric_rows[str(task_id)].append(episode_metrics)
                episode_metrics_rows.append(episode_metrics)
                with open(episode_metrics_jsonl_path, "a") as f:
                    f.write(json.dumps(episode_metrics) + "\n")
                if cfg.contact_debug:
                    debug_line = (
                        f"[contact-debug] episode={total_episodes} "
                        f"first_contact_step={episode_metrics['first_contact_step']} "
                        f"robot_geom={episode_metrics['first_contact_geom_robot']} "
                        f"object_geom={episode_metrics['first_contact_geom_object']} "
                        f"first_contacted_object={episode_metrics['first_contacted_object']} "
                        f"first_grasped_object={episode_metrics['first_grasped_object']} "
                        f"first_moved_object={episode_metrics['first_moved_object']} "
                        f"rsa_method={episode_metrics['rsa_method']} "
                        f"target_object={episode_metrics['target_object']} "
                        f"FCA={int(episode_metrics['fca_correct'])} "
                        f"RSA={int(episode_metrics['rsa_correct'])}"
                    )
                    print(debug_line)
                    log_file.write(debug_line + "\n")
            result_rows.append(
                {
                    "task_id": task_id,
                    "task_language": task_description,
                    "trial_index": episode_idx,
                    "seed": cfg.seed,
                    "success": bool(done),
                    "episode_length": t,
                    "error_message": error_message,
                    "target_object": episode_metrics.get("target_object", target_object),
                    "first_contacted_object": episode_metrics.get("first_contacted_object"),
                    "first_contact_step": episode_metrics.get("first_contact_step"),
                    "first_grasped_object": episode_metrics.get("first_grasped_object"),
                    "first_grasped_step": episode_metrics.get("first_grasped_step"),
                    "first_moved_object": episode_metrics.get("first_moved_object"),
                    "first_moved_step": episode_metrics.get("first_moved_step"),
                    "rsa_object": episode_metrics.get("rsa_object"),
                    "rsa_method": episode_metrics.get("rsa_method"),
                    "fca_correct": episode_metrics.get("fca_correct"),
                    "rsa_correct": episode_metrics.get("rsa_correct"),
                    "rsa_definition": episode_metrics.get("rsa_definition"),
                    "no_contact": episode_metrics.get("no_contact"),
                    "video_path": video_path,
                }
            )

            # Save a replay video of the episode
            if cfg.save_videos:
                should_save_success = bool(done) and task_saved_success_videos < cfg.save_video_successes_per_task
                should_save_failure = (not bool(done)) and task_saved_failure_videos < cfg.save_video_failures_per_task
                if should_save_success or should_save_failure:
                    video_path = save_rollout_video(
                        replay_images,
                        total_episodes,
                        success=done,
                        task_description=f"task={task_id}--trial={episode_idx}--{task_description}",
                        log_file=log_file,
                        rollout_dir=os.path.join(cfg.output_dir, "rollouts"),
                    )
                    result_rows[-1]["video_path"] = video_path
                    if should_save_success:
                        task_saved_success_videos += 1
                    else:
                        task_saved_failure_videos += 1

            # Log current results
            print(f"Success: {done}")
            if cfg.compute_rsa or cfg.compute_fca:
                print(
                    f"FCA: {episode_metrics.get('fca_correct')} "
                    f"RSA: {episode_metrics.get('rsa_correct')} "
                    f"first_contacted_object={episode_metrics.get('first_contacted_object')} "
                    f"rsa_method={episode_metrics.get('rsa_method')}"
                )
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            if cfg.compute_rsa or cfg.compute_fca:
                log_file.write(
                    f"FCA: {episode_metrics.get('fca_correct')} "
                    f"RSA: {episode_metrics.get('rsa_correct')} "
                    f"first_contacted_object={episode_metrics.get('first_contacted_object')} "
                    f"rsa_method={episode_metrics.get('rsa_method')}\n"
                )
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

        # Log final results
        task_success_rate = float(task_successes) / float(task_episodes)
        task_summary = {
            "task_language": task_description,
            "success_count": task_successes,
            "num_trials": task_episodes,
            "success_rate": task_success_rate,
        }
        if cfg.compute_rsa or cfg.compute_fca:
            task_summary.update(aggregate_metrics(task_metric_rows[str(task_id)]))
        task_summaries[str(task_id)] = task_summary
        print(f"Current task success rate: {task_success_rate}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {task_success_rate}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()
        if cfg.use_wandb:
            wandb.log(
                {
                    f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
                    f"num_episodes/{task_description}": task_episodes,
                }
            )

        with open(results_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "task_id",
                    "task_language",
                    "trial_index",
                    "seed",
                    "success",
                    "episode_length",
                    "error_message",
                    "target_object",
                    "first_contacted_object",
                    "first_contact_step",
                    "first_grasped_object",
                    "first_grasped_step",
                    "first_moved_object",
                    "first_moved_step",
                    "rsa_object",
                    "rsa_method",
                    "fca_correct",
                    "rsa_correct",
                    "rsa_definition",
                    "no_contact",
                    "video_path",
                ],
            )
            writer.writeheader()
            writer.writerows(result_rows)
        if cfg.compute_rsa or cfg.compute_fca:
            with open(episode_metrics_csv_path, "w", newline="") as f:
                fieldnames = [
                    "episode_index",
                    "seed",
                    "task_id",
                    "task_name",
                    "instruction",
                    "target_object",
                    "target_object_normalized",
                    "distractor_objects",
                    "first_contacted_object",
                    "first_contact_step",
                    "first_contact_geom_robot",
                    "first_contact_geom_object",
                    "selected_object",
                    "first_grasped_object",
                    "first_grasped_step",
                    "first_moved_object",
                    "first_moved_step",
                    "rsa_object",
                    "rsa_method",
                    "rsa_definition",
                    "rsa_correct",
                    "fca_definition",
                    "fca_correct",
                    "no_contact",
                    "success",
                    "episode_length",
                    "error_message",
                    "warnings",
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(episode_metrics_rows)
            overall_metrics = aggregate_metrics(episode_metrics_rows)
            with open(summary_metrics_json_path, "w") as f:
                json.dump(
                    {
                        "checkpoint_path": str(checkpoint_path),
                        "suite": cfg.task_suite_name,
                        "task_start": cfg.task_start,
                        "task_end": task_end,
                        "seed": cfg.seed,
                        "run_datetime": datetime.now(timezone.utc).isoformat(),
                        "tasks": task_summaries,
                        "overall": overall_metrics,
                        "episode_metrics_csv": episode_metrics_csv_path,
                        "episode_metrics_jsonl": episode_metrics_jsonl_path,
                        "geom_name_mapping": geom_mapping_json_path,
                    },
                    f,
                    indent=2,
                )
            with open(summary_metrics_md_path, "w") as f:
                f.write("# LIBERO Contact Metrics Summary\n\n")
                f.write(f"- suite: `{cfg.task_suite_name}`\n")
                f.write(f"- episodes: `{overall_metrics['num_episodes']}`\n")
                f.write(f"- success_rate: `{overall_metrics['success_rate']:.4f}`\n")
                f.write(
                    f"- referent_selection_accuracy: "
                    f"`{overall_metrics['referent_selection_accuracy']:.4f}`\n"
                )
                f.write(f"- first_contact_accuracy: `{overall_metrics['first_contact_accuracy']:.4f}`\n")
                f.write(f"- rsa_fallback_rate: `{overall_metrics.get('rsa_fallback_rate', 0.0):.4f}`\n")
                f.write(f"- rsa_definition: `{overall_metrics['rsa_definition']}`\n")
                f.write(f"- rsa_note: {overall_metrics['rsa_note']}\n\n")
                f.write("| task_id | success_rate | RSA | FCA | no_contact | wrong_first_contact | rsa_fallback_rate |\n")
                f.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
                for summary_task_id, summary in task_summaries.items():
                    f.write(
                        f"| {summary_task_id} | {summary['success_rate']:.4f} | "
                        f"{summary.get('referent_selection_accuracy', 0.0):.4f} | "
                        f"{summary.get('first_contact_accuracy', 0.0):.4f} | "
                        f"{summary.get('num_no_contact_episodes', 0)} | "
                        f"{summary.get('num_wrong_first_contact_episodes', 0)} | "
                        f"{summary.get('rsa_fallback_rate', 0.0):.4f} |\n"
                    )
            with open(geom_mapping_json_path, "w") as f:
                json.dump(task_geom_mappings, f, indent=2)
        with open(summary_json_path, "w") as f:
            json.dump(
                {
                    "checkpoint_path": str(checkpoint_path),
                    "suite": cfg.task_suite_name,
                    "task_start": cfg.task_start,
                    "task_end": task_end,
                    "seed": cfg.seed,
                    "run_datetime": datetime.now(timezone.utc).isoformat(),
                    "tasks": task_summaries,
                    "overall_success_count": total_successes,
                    "overall_num_trials": total_episodes,
                    "overall_success_rate": float(total_successes) / float(total_episodes),
                    "results_csv": results_csv_path,
                    "episode_metrics_csv": episode_metrics_csv_path if (cfg.compute_rsa or cfg.compute_fca) else None,
                    "episode_metrics_jsonl": episode_metrics_jsonl_path if (cfg.compute_rsa or cfg.compute_fca) else None,
                    "summary_metrics_json": summary_metrics_json_path if (cfg.compute_rsa or cfg.compute_fca) else None,
                    "summary_metrics_md": summary_metrics_md_path if (cfg.compute_rsa or cfg.compute_fca) else None,
                    "geom_name_mapping": geom_mapping_json_path if (cfg.compute_rsa or cfg.compute_fca) else None,
                    "log_file": local_log_filepath,
                },
                f,
                indent=2,
            )

    # Save local log file
    log_file.close()

    # Push total metrics and local log file to wandb
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": float(total_successes) / float(total_episodes),
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)


if __name__ == "__main__":
    eval_libero()
