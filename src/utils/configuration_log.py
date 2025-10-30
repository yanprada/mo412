import logging
import sys


def configure_logging_global(console_level=logging.INFO, file_level=logging.DEBUG):
    """
    Configures the logging system globally.

    Args:
    console_level (int): Minimum level of messages to display in the console.
    file_level (int): Minimum level of messages to save to the file.
    """

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Desired format for messages
    format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # CONSOLE HANDLER (StreamHandler)
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(format)
        logger.addHandler(console_handler)

        # FILE HANDLER - Optional but recommended
        file_handler = logging.FileHandler("./logger.log", mode="w")
        file_handler.setLevel(file_level)
        file_handler.setFormatter(format)
        logger.addHandler(file_handler)

        # 4. Mensaje de inicio
        logger.info("Logging System Initialized.")
        logger.info(
            f"Console Level: {logging.getLevelName(console_level)}, File Level: {logging.getLevelName(file_level)}"
        )
