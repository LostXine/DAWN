import os
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn

# --- Configuration ---
# Path to the directory containing all the .webm video files.
VIDEO_DIR = "/nfs/bigflow/add_disk0/nero/datasets/robotics/ssv2/20bn-something-something-v2"
# Path to the directory where you want to save the extracted frames.
OUTPUT_DIR = "/nfs/bigflow/add_disk0/nero/datasets/robotics/ssv2/rawframes"
# Number of worker processes to use. It's often good practice to set this
# to the number of CPU cores on your machine.
NUM_WORKERS = 4
# Desired frame rate (frames per second) for extraction.
# The original videos are 25 fps, but 30 fps is a common choice for models.
FPS = 30
# Output image quality. 1 is the highest quality.
QUALITY = 1

# --- Function to extract frames from a single video ---
def extract_frames(video_path):
    """
    Extracts frames from a single video file using FFmpeg.
    """
    try:
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        output_folder = os.path.join(OUTPUT_DIR, video_id)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # FFmpeg command
        # -i: input file
        # -r: output frame rate
        # -q:v: video quality (1=highest)
        # -y: overwrite output files without asking
        # %05d: zero-padded file names (e.g., img_00001.jpg)
        command = (
            f"ffmpeg -i {video_path} -r {FPS} -q:v {QUALITY} "
            f"-y {os.path.join(output_folder, '%05d.jpg')} > /dev/null 2>&1"
        )
        
        # Execute the command
        os.system(command)
        
        return f"Successfully extracted frames for {video_id}"
    except Exception as e:
        return f"Failed to extract frames for {video_id}: {e}"

# --- Main execution block ---
if __name__ == "__main__":
    # Get a list of all video files
    video_files = glob.glob(os.path.join(VIDEO_DIR, "*.webm"))
    
    if not video_files:
        print(f"No video files found in {VIDEO_DIR}. Please check the path.")
    else:
        print(f"Found {len(video_files)} video files. Starting frame extraction...")
        
        # Create the main output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Use Rich for a beautiful progress bar
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
            TimeElapsedColumn(),
        ) as progress:
            
            # Create a task for the overall progress bar
            task = progress.add_task("[cyan]Processing videos...", total=len(video_files))
            
            # Use ProcessPoolExecutor to run tasks in parallel
            with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
                
                # `executor.map` applies a function to all items in an iterable
                # `as_completed` iterates over the futures as they complete
                futures = {executor.submit(extract_frames, video_file) for video_file in video_files}
                
                for future in as_completed(futures):
                    result = future.result()
                    progress.update(task, advance=1)
                    # Uncomment the line below if you want to print each result as it finishes
                    # print(result)
        
        print("Frame extraction complete!")