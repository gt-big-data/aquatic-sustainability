import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev")
    GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")
    #For login
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

    MONGODB_URI = os.getenv("MONGODB_URI", "")
    EDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
    RQ_DEFAULT_QUEUE = os.environ.get('RQ_DEFAULT_QUEUE', 'flood-jobs')

    REDIS_URL = "redis://localhost:6379/0"

        # (You can add Supabase and other config vars here)

