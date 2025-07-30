import cv2
import os
import re
from tqdm import tqdm
import json
from inference.rollout_video import RolloutVideo
import glob
import time 
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image
import numpy as np
# Import necessary components from the 'rich' library
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)


def process_episode(image_folder, log_dir, r_map, fps=24, image_format='jpg', sort=True):
    """
    Converts a sequence of images in a folder to a video file.

    Args:
        image_folder (str): Path to the folder containing the images.
        video_name (str): Name of the output video file (e.g., 'output.mp4').
        fps (int): Frames per second for the output video.
        image_format (str): The file extension of the images (e.g., 'png', 'jpg').
        sort (bool): Whether to sort the images alphanumerically.
    """
    idx = image_folder.split('/')[-1]
    metadata = json.load(open(os.path.join(image_folder, "metadata.json"), "r"))

    lang = metadata["language"]
    
    task = r_map[lang]
    if "push" not in task or "right" not in task:
        return ""
    log_dir = os.path.join(log_dir, task)
    video_name = f"{idx}_{lang}.mp4"

    # print(lang)
    image_folder = os.path.join(image_folder, "rgb_static")
    # Get all image file names from the folder
    images = [img for img in os.listdir(image_folder) if img.endswith(f".{image_format}")]

    if not images:
        print(f"No images found with the format '{image_format}' in the folder '{image_folder}'.")
        return

    # Sort the images by name to ensure correct order
    if sort:
        # This sorting key handles natural sorting for names like 'frame1', 'frame2', 'frame10'
        images.sort(key=lambda f: int(re.sub('\D', '', f)))

    # log_dir = "outputs/tests"

    rollout_video = RolloutVideo(
        logger=None,
        empty_cache=False,
        log_to_file=True,
        save_dir=os.path.join(log_dir),
        resolution_scale=1,
    )
    rollout_video.new_video(str(idx), caption="")
    rollout_video.new_subtask()
    
    print(f"Starting video creation: {video_name} with {len(images)} frames.")

    # Loop through all the images and write them to the video
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = Image.open(image_path).resize((256, 256))
        frame = np.array(frame)
        # frame = cv2.imread(image_path)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        rollout_video.update(frame)

    rollout_video.add_language_instruction(lang)
    rollout_video.write_to_tmp()
    
    path = rollout_video._log_currentvideos_to_file(idx, 0, save_as_video=True)
    wandb.log({
        f"{task}/{idx}": wandb.Video(path)
    })
    
    print(f"Video '{video_name}' created successfully in the script's directory.")

if __name__ == '__main__':
    # The desired frames per second for the video.
    frames_per_second = 30
    import wandb
    wandb.init(project="calvin_gripper_video")


    # The file format of your images ('png', 'jpg', etc.).
    img_format = 'png'
    
    from omegaconf import OmegaConf
    annotation = OmegaConf.load("/home/nero/Robotics/DAWN/configs/annotation/validation3.yaml")
    r_map = {}
    for k, v in annotation.items():
        for z in v:
            r_map[z] = k
    # print(r_map)

    path = '/home/nero/Robotics/DAWN/data/calvin/dataset_opt/task_ABC_D/training/episodes/*'
    output = 'outputs/visualize/calvin_ABC_D/train/'
    lst = glob.glob(path)
    
    num_episodes = len(lst)
    print(f"Total episodes to process: {num_episodes}")

        
    def prepare_tasks(lst):
        """
        A generator function that prepares tasks one by one.
        This function fetches an episode, converts its 'steps' to a NumPy list,
        and then 'yields' the complete, serializable task.
        (This function remains unchanged)
        """
        for idx, episode in enumerate(lst):
            yield (episode, output, r_map, frames_per_second, img_format)

    # Define the rich progress bar with custom columns for a clean look
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    )

    start_time = time.time()
    with progress:
        # Add two separate tasks to the progress bar
        preparing_task_id = progress.add_task("[cyan]Preparing tasks...", total=num_episodes)
        processing_task_id = progress.add_task("[green]Processing episodes...", total=num_episodes)

        with ProcessPoolExecutor(32) as executor:
            # 1. Prepare tasks and submit them to the executor
            #    The list of future objects is created here.
            futures = []
            for task in prepare_tasks(lst):
                futures.append(executor.submit(process_episode, *task))
                # process_episode(*task)  # Call the function directly for immediate processing
                # Update the "Preparing" progress bar as each task is submitted
                progress.update(preparing_task_id, advance=1)
            
            # Mark the preparation task as complete once all tasks are submitted
            progress.update(preparing_task_id, description="[bold cyan]Preparation complete ✔")

            # 2. Process results as they are completed
            results = []
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    # Using progress.console to print errors without breaking the bar
                    progress.console.print(f"A task generated an exception: {e}")
                finally:
                    # Update the "Processing" progress bar as each future completes
                    progress.update(processing_task_id, advance=1)

    print("\n--- Processing Complete ---")
    print(f"Time taken: {time.time() - start_time:.2f} seconds")