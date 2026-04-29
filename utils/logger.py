import os
import sys
import logging
import platform
from datetime import datetime
from logging.handlers import RotatingFileHandler
import motor.motor_asyncio
from config.settings import MONGO_CONFIG

# Create logs directory
logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(logs_dir, exist_ok=True)

# Define log file path
log_file = os.path.join(logs_dir, f'blackhole_{datetime.now().strftime("%Y%m%d")}.log')

# Configure root logger
root_logger = logging.getLogger()

# Determine log level based on environment
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
if log_level == 'DEBUG' and os.getenv('ENVIRONMENT') != 'production':
    root_logger.setLevel(logging.DEBUG)
else:
    root_logger.setLevel(logging.INFO)

# Console handler
use_utf8 = platform.system() != 'Windows'
if use_utf8:
    console_handler = logging.StreamHandler(sys.stdout)
else:
    class ASCIIStreamHandler(logging.StreamHandler):
        def emit(self, record):
            try:
                msg = self.format(record)
                msg = msg.replace('✅', '[SUCCESS]').replace('❌', '[ERROR]').replace('⚠️', '[WARNING]')
                stream = self.stream
                stream.write(msg + self.terminator)
                self.flush()
            except Exception:
                self.handleError(record)
    console_handler = ASCIIStreamHandler(sys.stdout)

# Set console log level based on environment
if os.getenv('ENVIRONMENT') == 'production':
    console_handler.setLevel(logging.WARNING)  # Less verbose in production
else:
    console_handler.setLevel(logging.INFO)
console_format = logging.Formatter('%(levelname)s - %(name)s - %(message)s')
console_handler.setFormatter(console_format)

# File handler
file_handler = RotatingFileHandler(
    log_file,
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5,
    encoding='utf-8'
)
# Set file log level based on environment
if os.getenv('ENVIRONMENT') == 'production':
    file_handler.setLevel(logging.INFO)  # Less verbose in production
else:
    file_handler.setLevel(logging.DEBUG)
file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
file_handler.setFormatter(file_format)

# Add handlers to root logger
root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)

# Async MongoDB client
mongo_client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_CONFIG['uri'])
mongo_db = mongo_client[MONGO_CONFIG['database']]
mongo_collection = mongo_db[MONGO_CONFIG['collection']]

def get_logger(name):
    """
    Get a logger with the specified name.

    Args:
        name (str): The name of the logger, typically __name__ of the calling module.

    Returns:
        logging.Logger: A configured logger instance.
    """
    logger = logging.getLogger(name)
    return logger

async def log_to_mongo(task_id: str, agent: str, input_data: str, output_data: dict):
    """
    Log task details to MongoDB asynchronously.

    Args:
        task_id (str): Unique task identifier.
        agent (str): Agent processing the task.
        input_data (str): Input data for the task.
        output_data (dict): Output from the agent.
    """
    try:
        log_entry = {
            'task_id': task_id,
            'agent': agent,
            'input': input_data,
            'output': output_data,
            'timestamp': datetime.now().isoformat()
        }
        await mongo_collection.insert_one(log_entry)
        logger.info(f"Logged task {task_id} to MongoDB")
    except Exception as e:
        logger.error(f"Failed to log task {task_id} to MongoDB: {str(e)}")

if __name__ == "__main__":
    logger = get_logger(__name__)
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")
    import asyncio
    asyncio.run(log_to_mongo("test_task", "edumentor_agent", "Sample PDF", {"result": "Test summary"}))
    print(f"Log file created at: {log_file}")


# import os
# import sys
# import logging
# import platform
# from datetime import datetime
# from logging.handlers import RotatingFileHandler
# from pymongo import MongoClient
# from config.settings import MONGO_CONFIG

# # Create logs directory
# logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
# os.makedirs(logs_dir, exist_ok=True)

# # Define log file path
# log_file = os.path.join(logs_dir, f'blackhole_{datetime.now().strftime("%Y%m%d")}.log')

# # Configure root logger
# root_logger = logging.getLogger()
# root_logger.setLevel(logging.INFO)

# # Console handler
# use_utf8 = platform.system() != 'Windows'
# if use_utf8:
#     console_handler = logging.StreamHandler(sys.stdout)
# else:
#     class ASCIIStreamHandler(logging.StreamHandler):
#         def emit(self, record):
#             try:
#                 msg = self.format(record)
#                 msg = msg.replace('✅', '[SUCCESS]').replace('❌', '[ERROR]').replace('⚠️', '[WARNING]')
#                 stream = self.stream
#                 stream.write(msg + self.terminator)
#                 self.flush()
#             except Exception:
#                 self.handleError(record)
#     console_handler = ASCIIStreamHandler(sys.stdout)

# console_handler.setLevel(logging.INFO)
# console_format = logging.Formatter('%(levelname)s - %(name)s - %(message)s')
# console_handler.setFormatter(console_format)

# # File handler
# file_handler = RotatingFileHandler(
#     log_file,
#     maxBytes=10*1024*1024,  # 10MB
#     backupCount=5,
#     encoding='utf-8'
# )
# file_handler.setLevel(logging.DEBUG)
# file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
# file_handler.setFormatter(file_format)

# # Add handlers to root logger
# root_logger.addHandler(console_handler)
# root_logger.addHandler(file_handler)

# # MongoDB client
# mongo_client = MongoClient(MONGO_CONFIG['uri'])
# mongo_db = mongo_client[MONGO_CONFIG['database']]
# mongo_collection = mongo_db[MONGO_CONFIG['collection']]

# def get_logger(name):
#     """
#     Get a logger with the specified name.

#     Args:
#         name (str): The name of the logger, typically __name__ of the calling module.

#     Returns:
#         logging.Logger: A configured logger instance.
#     """
#     logger = logging.getLogger(name)
#     return logger

# def log_to_mongo(task_id: str, agent: str, input_data: str, output_data: dict):
#     """
#     Log task details to MongoDB.

#     Args:
#         task_id (str): Unique task identifier.
#         agent (str): Agent processing the task.
#         input_data (str): Input data for the task.
#         output_data (dict): Output from the agent.
#     """
#     try:
#         log_entry = {
#             'task_id': task_id,
#             'agent': agent,
#             'input': input_data,
#             'output': output_data,
#             'timestamp': datetime.now().isoformat()
#         }
#         mongo_collection.insert_one(log_entry)
#         logger.info(f"Logged task {task_id} to MongoDB")
#     except Exception as e:
#         logger.error(f"Failed to log task {task_id} to MongoDB: {str(e)}")

# if __name__ == "__main__":
#     logger = get_logger(__name__)
#     logger.debug("Debug message")
#     logger.info("Info message")
#     logger.warning("Warning message")
#     logger.error("Error message")
#     logger.critical("Critical message")
#     log_to_mongo("test_task", "edumentor_agent", "Sample PDF", {"result": "Test summary"})
#     print(f"Log file created at: {log_file}")


