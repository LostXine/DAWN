import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

class FlowNormalizer:
    def __init__(self, height, width):
        """
        Initialize the FlowNormalizer with the dimensions of the flow.

        Args:
            height (int): The height (H) of the flow.
            width (int): The width (W) of the flow.
        """
        self.height = height
        self.width = width

    def normalize(self, flow):
        """
        Normalize the flow tensor to the range [0, 1].

        Args:
            flow (numpy.ndarray): The flow tensor of shape (B, H, W, 2).

        Returns:
            numpy.ndarray: The normalized flow tensor of the same shape.
        """
        normalized_flow = np.zeros_like(flow, dtype=np.float32)
        # Normalize channel 0 (height)
        normalized_flow[..., 0] = (flow[..., 0] + self.height) / (2 * self.height)
        # Normalize channel 1 (width)
        normalized_flow[..., 1] = (flow[..., 1] + self.width) / (2 * self.width)
        return normalized_flow

    def unnormalize(self, normalized_flow):
        """
        Unnormalize the flow tensor from the range [0, 1] back to the original range.

        Args:
            normalized_flow (numpy.ndarray): The normalized flow tensor of shape (B, H, W, 2).

        Returns:
            numpy.ndarray: The unnormalized flow tensor of the same shape.
        """
        flow = np.zeros_like(normalized_flow, dtype=np.float32)
        # Unnormalize channel 0 (height)
        flow[..., 0] = (normalized_flow[..., 0] * 2 * self.height) - self.height
        # Unnormalize channel 1 (width)
        flow[..., 1] = (normalized_flow[..., 1] * 2 * self.width) - self.width
        return flow


def visualize_flow_vectors_as_PIL(image, flow=None, step=16, title="Optical Flow Vectors"):
    """
    Overlay optical flow vectors on an image.

    Parameters:
        image (numpy.ndarray): Input image (H, W, 3).
        flow (numpy.ndarray): Optical flow array (H, W, 2).
        step (int): Sampling step for displaying flow vectors.

    Returns:
        PIL.Image.Image: Visualization as a PIL image.
    """

    # Create a matplotlib figure
    fig = Figure(figsize=(4, 4))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)

    # Overlay flow vectors
    ax.imshow(image)

    if flow is not None:
        h, w = flow.shape[:2]
        y, x = np.mgrid[step // 2 : h : step, step // 2 : w : step].astype(np.int32)
        fx, fy = flow[x, y].T
        ax.quiver(x, y, fx, fy, color="red", angles="xy", scale_units="xy", scale=1, width=0.002)
    
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()

    # Render the figure to a PIL image
    canvas.draw()
    buf = canvas.buffer_rgba()
    pil_image = Image.frombuffer("RGBA", canvas.get_width_height(), buf, "raw", "RGBA", 0, 1)

    return pil_image
