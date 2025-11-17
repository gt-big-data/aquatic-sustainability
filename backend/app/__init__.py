from flask import Flask, send_from_directory, redirect, url_for
from flask_cors import CORS
from supabase import create_client
from .config import Config

supabase = create_client(Config.SUPABASE_URL, Config.SUPABASE_ANON_KEY)

def create_app():
    app = Flask(__name__, template_folder="../templates", static_folder="../static")
    app.config.from_object(Config)
    CORS(app, resources={r"/api/*": {"origins": "https://aquatic-sustainability.vercel.app"}})

    # registering API routes
    from .routes import bp as api_bp
    app.register_blueprint(api_bp, url_prefix="/api")

    # Redirect homepage , this is pollution for now, we can create a landing page
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

    return app