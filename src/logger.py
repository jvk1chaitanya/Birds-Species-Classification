import logging
import os
import datetime as dt

log_file = f"{dt.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log"
log_path = os.path.join(os.getcwd(), "logs", log_file)
os.makedirs(os.path.dirname(log_path), exist_ok=True)

logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s'
)