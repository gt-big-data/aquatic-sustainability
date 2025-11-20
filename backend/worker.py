# worker.py

import os
import sys

from redis import Redis
from rq import Worker, Queue
from app.config import Config

# Ensure backend/app is importable
sys.path.append(os.path.dirname(__file__))

listen = [Config.RQ_DEFAULT_QUEUE]

redis_url = os.getenv("REDIS_URL", Config.REDIS_URL)
conn = Redis.from_url(redis_url)

if __name__ == "__main__":
    queue_list = [Queue(name, connection=conn) for name in listen]
    worker = Worker(queue_list, connection=conn)
    worker.work()
