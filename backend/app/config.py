import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY")
    if not SECRET_KEY:
        raise RuntimeError("SECRET_KEY in .env file not set. Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\"")
    GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")
    #For login
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

    MONGODB_URI = os.getenv("MONGODB_URI", "")
    EDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
    RQ_DEFAULT_QUEUE = os.environ.get('RQ_DEFAULT_QUEUE', 'flood-jobs')
    NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

    REDIS_URL = "redis://localhost:6379/0"

        # (You can add Supabase and other config vars here)

