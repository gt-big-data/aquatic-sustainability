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

    # Homepage - landing page
    @app.route("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")
    # this is based on the figma design
    # Pollution & Marine Health
    @app.route("/pollution")
    def pollution():
        return send_from_directory(app.static_folder, "pollution.html")
    # Drought & Climate Impact
    @app.route("/drought")
    def drought():
        return send_from_directory(app.static_folder, "climate.html")
    # Flood Resources & Availability
    @app.route("/flood")
    def flood():
        return send_from_directory(app.static_folder, "water.html")

    return app