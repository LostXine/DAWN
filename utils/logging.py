import os
import logging 
from rich.logging import RichHandler
from rich.traceback import install
from rich.console import Console
from accelerate.logging import get_logger

logger = logging.getLogger(__name__)

def setup_logging(is_main_process: bool = True, log_dir: str = 'outputs/') -> None:
    """Setup logging according to `training_args`."""
    install()

    if is_main_process:
        os.makedirs(log_dir, exist_ok=True)
        log_file = open(os.path.join(log_dir, "log.ansi"), "w")
        console_file = Console(file=log_file, force_terminal=True, width=120, record=True, stderr=True)
        file_handler = RichHandler(console=console_file, rich_tracebacks=True, show_path=False, markup=True)
    
    rich_handler = RichHandler(rich_tracebacks=True, show_path=False, markup=True)
    logger = logging.getLogger() # Get the root logger
    if logger.hasHandlers():
        logger.handlers.clear()

    logging.basicConfig(
        level=logging.INFO if is_main_process else logging.CRITICAL,
        # format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        format='[%(asctime)s] [bold green]{%(name)s}[/] - %(message)s',
        datefmt="%m/%d/%Y %H:%M:%S",
        # handlers=[logging.StreamHandler(sys.stdout)],
        handlers=[rich_handler, file_handler] if is_main_process else [rich_handler]
    )
