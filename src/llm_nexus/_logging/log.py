import datetime
import logging
import os
import time
from datetime import timedelta

# Debate on how to format messages (f-strings vs. .format() vs. %): https://stackoverflow.com/questions/54367975/python-3-7-logging-f-strings-vs

class RelativeTimeFormatter(logging.Formatter):
    """
    A custom formatter that formats the time in the log record as a relative time from the start of the program.
    """

    def formatTime(self, record, datefmt=None):
        ct = record.created
        relative_ct = ct - self.converter(time.time())
        delta = timedelta(seconds=relative_ct)
        return str(delta)


# Create a log
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

# Create a file handler
timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
# Create a logs folder if it does not exist
if not os.path.exists("logs"):
    os.makedirs("logs")
log_file_handler = logging.FileHandler(f"logs/{timestamp_str}.log")
log_file_handler.setLevel(logging.DEBUG)

# Create a console handler
console_log_handler = logging.StreamHandler()
console_log_handler.setLevel(logging.INFO)

# Set the formatter
log_formatter = logging.Formatter(
    # "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    "%(relativeCreated)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s",
)
log_file_handler.setFormatter(log_formatter)
console_log_handler.setFormatter(log_formatter)

# Add the handlers to the log
log.addHandler(log_file_handler)
log.addHandler(console_log_handler)
