import logging
import os

LOG_LEVEL_ENV_VAR = "LOG_LEVEL"
KNOWN_MODELS = [
    "facebook/opt-125m",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",  # OOM on Tesla T4
    "meta-llama/Llama-2-7b-chat-hf",  # OOM on Tesla T4
    "neuralmagic/Llama-2-7b-chat-quantized.w8a8",
    "neuralmagic/Meta-Llama-3.1-8B-quantized.w8a16",
    "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16",
    "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a8",  # default
]

DEFAULT_MODEL = KNOWN_MODELS[-1]


def get_logger(
    name: str, level: str = os.getenv(LOG_LEVEL_ENV_VAR, "INFO")
) -> logging.Logger:
    """
    Helper to preconfigure a logger instance.

    Args:
        name: Name of the logger; generally, the module name should be used.
        level: Level of logs to send to the console.

    Returns:
        A logger instance.
    """
    logger = logging.getLogger(name)

    logger.setLevel(level)

    return logger
