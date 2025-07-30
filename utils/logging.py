import os
import logging 
from rich.logging import RichHandler
from rich.traceback import install
from rich.console import Console
from accelerate.logging import get_logger

logger = logging.getLogger(__name__)

def setup_logging(is_main_process: bool = True, device: str="0", log_dir: str = 'outputs/') -> None:
    """Setup logging according to `training_args`."""
    install()

    os.makedirs(log_dir, exist_ok=True)
    log_file = open(os.path.join(log_dir, f"log_{device}.ansi"), "w")
    console_file = Console(file=log_file, force_terminal=True, width=180, record=True, stderr=True)
    file_handler = RichHandler(console=console_file, rich_tracebacks=True, show_path=False, markup=True)
    
    rich_handler = RichHandler(rich_tracebacks=True, show_path=False, markup=True)
    logger = logging.getLogger() # Get the root logger
    if logger.hasHandlers():
        logger.handlers.clear()

    handlers = [file_handler]
    if is_main_process:
        handlers.append(rich_handler)

    logging.basicConfig(
        level=logging.INFO,
        # format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        format='[%(asctime)s] [bold green]{%(name)s}[/] - %(message)s',
        datefmt="%m/%d/%Y %H:%M:%S",
        # handlers=[logging.StreamHandler(sys.stdout)],
        handlers=handlers
    )
