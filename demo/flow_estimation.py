import sys
sys.path.append('../')  # Adjust path to import from the parent directory if needed

import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import os
import tempfile

import hydra 

import logging
import accelerate
import torch
import numpy as np

from dawn.models.system2.flow_utils import visualize_flow_vectors_as_PIL, FlowNormalizer
        
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


config = {
    "_target_" : "dawn.models.system2.ldm_support.LatentMotionEstimation",
    "flow_to_rgb" : False,
    "use_cfg" : False,
    "use_interval" : True,
}

accelerator = accelerate.Accelerator()
accelerate.utils.set_seed(0)

weights = "../outputs/DAWN_stage_1/2025-07-08_16-03-55/checkpoints/model_0100000.pth"
print(f"Loading model configuration from {config}")
model = hydra.utils.instantiate(config)

if os.path.exists(weights):
    logger.info(f"Loading model weights from {weights}.")
    logger.info(model.load_state_dict(torch.load(weights, map_location="cpu"), strict=False))
else:
    logger.warning(f"Model weights file not found: {weights}. Skipping loading weights.")

model = accelerator.prepare(model)
model.eval()

normalizer = FlowNormalizer(256, 256)
                

# Function to process the image, text, and number
def process_images_and_flow(image_path_input: str, input_text: str, k_value: int, inference_step: int):
    """
    Loads an image from the given path, finds the K-th next image in the same directory,
    and generates placeholder images for optical flow.

    Args:
        image_path_input (str): The file path to the input image.
        input_text (str): A string that was previously added to the output images (now removed per request).
        k_value (int): The 'K' value to find the K-th next image in the folder.

    Returns:
        list: A list containing PIL Image objects and their labels for gr.Gallery.
    """
    output_data = [] # Will store (PIL.Image, label) tuples

    # Note: temp_dir is no longer strictly needed for saving images,
    # but it's kept here just in case you add other file-based operations.
    # If not, you can remove it.
    temp_dir = tempfile.mkdtemp() # Create a temporary directory for any necessary temporary files

    try:
        # --- 1. Load Original Image ---
        original_img = Image.open(image_path_input).convert("RGB").resize((256, 256))
        # No text overlay on original image as requested
        output_data.append((original_img, "Original Image")) # Store PIL Image directly

        # # --- 2. Find Goal Image (K-th next image) ---
        # image_directory = os.path.dirname(image_path_input)
        # if not image_directory: # If image_path_input is just a filename in CWD
        #     image_directory = "."

        # all_files_in_dir = sorted(os.listdir(image_directory))
        # image_files = [f for f in all_files_in_dir if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

        # if not image_files:
        #     raise gr.Error(f"No image files found in directory: {image_directory}")

        # # Find the index of the current image in the sorted list
        # current_image_filename = os.path.basename(image_path_input)
        # try:
        #     current_image_index = image_files.index(current_image_filename)
        # except ValueError:
        #     raise gr.Error(f"'{current_image_filename}' not found in the list of images in its directory. Please ensure the path is exact.")

        # # Calculate the index for the K-th next image
        # goal_image_index = min((current_image_index + k_value), len(image_files) - 1)
        # goal_image_path = os.path.join(image_directory, image_files[goal_image_index])
        goal_image_path = image_path_input
        goal_img = Image.open(goal_image_path).convert("RGB").resize((256, 256))
        # No text overlay on goal image as requested
        output_data.append((goal_img, f"Goal Image (K={k_value})")) # Store PIL Image directly

        # --- 3. Generate Placeholder Optical Flow Images ---
        img_width, img_height = original_img.size
        
        data = torch.stack([
            torch.from_numpy(np.array(original_img) / 255.0),
            torch.from_numpy(np.array(goal_img) / 255.0),
            torch.from_numpy(np.array(goal_img) / 255.0)
        ]).cuda()
        data = data.permute(0, 3, 1, 2).float()[None, ] # Convert to (B, C, H, W) format
        print(f"Input data shape: {data.shape}")
        # actual_flow = model.gen_flow(data)

        data = {
            "rgb_static": data,
            "language": [input_text],
            "skip_frame": torch.tensor(k_value).view(-1).cuda(),
        }
        with torch.no_grad():
            pred = model(data, inference_step=inference_step)
        gt = pred['gt_flow']
        pd = pred['generated_flow']

        gt_flow = normalizer.unnormalize(gt[0].permute(1, 2, 0).cpu().numpy())  # Convert to (H, W, C) format and unnormalize
        gt = visualize_flow_vectors_as_PIL(original_img, gt_flow, step=4, title="Ground Truth Optical Flow")
        pd_flow = normalizer.unnormalize(pd[0].permute(1, 2, 0).cpu().numpy())  # Convert to (H, W, C) format and unnormalize
        pd = visualize_flow_vectors_as_PIL(original_img, pd_flow, step=4, title="Predicted  Optical Flow")
        print("Done!")        
        output_data.append((gt, "Actual Optical Flow"))
        output_data.append((pd, "Predicted Optical Flow"))

        return output_data # Return a list of (PIL.Image, label) tuples for gr.Gallery

    except FileNotFoundError:
        raise gr.Error(f"Error: The image file was not found at '{image_path_input}'. Please ensure the path is correct and accessible.")
    except Exception as e:
        # Catch any other unexpected errors during processing
        raise gr.Error(f"An unexpected error occurred: {e}")
    finally:
        # Clean up the temporary directory if it was created
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            try:
                os.rmdir(temp_dir) # Only removes empty directories
            except OSError:
                # If directory is not empty (e.g., other temporary files were created/not cleaned),
                # you might need shutil.rmtree(temp_dir) for robust cleanup.
                pass

# Create the Gradio interface
iface = gr.Interface(
    fn=process_images_and_flow,
    inputs=[
        gr.Textbox(label="Input Image Path", placeholder="Enter the full path to your image (e.g., C:/images/my_photo.png)"),
        gr.Textbox(label="Additional Text", placeholder="Enter some descriptive text..."),
        gr.Slider(minimum=0, maximum=30, step=1, label="K-th Next Image (0-30)", value=1), # K-value for next image
        gr.Slider(minimum=1, maximum=50, step=1, label="Inference Denoising Step (1-50)", value=25) # K-value for next image
    ],
    outputs=gr.Gallery(
        label="Processed Images",
        columns=2,  # Display in 2 columns
        rows=2,     # Display in 2 rows
        object_fit="contain", # Ensures images fit within their grid cells
        height="auto" # Adjust height automatically
    ),
    show_progress="hidden",
    # title="Image Sequence & Optical Flow App",
    # description="Enter an image path, text, and a 'K' value. The app will display the original image, "
    #             "the K-th next image in its folder, and placeholder optical flow representations in a 2x2 grid."
    live=True,
)


# Launch the Gradio app
if __name__ == "__main__":
    iface.launch()
