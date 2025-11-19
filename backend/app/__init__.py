from flask import Flask, send_from_directory
from flask_cors import CORS
from supabase import create_client
from .config import Config
import os

# Initialize supabase client only if credentials are available
supabase = None
if Config.SUPABASE_URL and Config.SUPABASE_ANON_KEY:
    try:
        supabase = create_client(Config.SUPABASE_URL, Config.SUPABASE_ANON_KEY)
    except Exception as e:
        print(f"Warning: Failed to initialize Supabase client: {e}")
        supabase = None

def create_app():
    app = Flask(__name__, template_folder="../templates", static_folder="../static")
    app.config.from_object(Config)
    CORS(app, resources={r"/api/*": {"origins": "https://aquatic-sustainability.vercel.app"}})

    # registering API routes
    from .routes import bp as api_bp
    app.register_blueprint(api_bp, url_prefix="/api")

    # Homepage - landing page
    @app.route("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")
    @app.route("/login")
    def login():
        return send_from_directory(app.static_folder + "/pages", "login.html")
    @app.route("/register")
    def register():
        return send_from_directory(app.static_folder + "/pages", "register.html")
    # this is based on the figma design
    # Pollution & Marine Health
    @app.route("/oil-spill")
    def oilSpill():
        return send_from_directory(app.static_folder + "/pages", "oil-spill.html")
    # Climate Impact & Erosion Trends
    @app.route("/flood-drought")
    def floodDrought():
        return send_from_directory(app.static_folder + "/pages", "flood-drought.html")
    # Water Resources & Availability
    @app.route("/water")
    def water():
        return send_from_directory(app.static_folder + "/pages", "water.html")

    # Pages routes
    #@app.route("/pages/flood-drought.html")
    #def flood_drought():
    #     return send_from_directory(app.static_folder, "pages/flood-drought.html")
    
    # @app.route("/pages/oil-spill.html")
    # def oil_spill():
    #     return send_from_directory(app.static_folder, "pages/oil-spill.html")
    
    # @app.route("/pages/login.html")
    # def login():
    #     return send_from_directory(app.static_folder, "pages/login.html")
    
    # @app.route("/pages/register.html")
    # def register():
    #     return send_from_directory(app.static_folder, "pages/register.html")
    
    # Serve static files (CSS, JS, images, etc.) from root paths
    # This must be last to avoid intercepting specific routes
    @app.route("/<path:filename>")
    def serve_static(filename):
        # Check if file exists in static folder
        file_path = os.path.join(app.static_folder, filename)
        if os.path.isfile(file_path):
            return send_from_directory(app.static_folder, filename)
        # Return 404 if file doesn't exist
        return "File not found", 404

    return app