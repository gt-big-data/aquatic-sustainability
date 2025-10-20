from flask import Flask, send_from_directory, redirect, url_for
from flask_cors import CORS
from .config import Config

def create_app():
    app = Flask(__name__, template_folder="../templates", static_folder="../static")
    app.config.from_object(Config)
    CORS(app)

    # registering API routes
    from .routes import bp as api_bp
    app.register_blueprint(api_bp, url_prefix="/api")

    # Redirect homepage , this is pollution for now, we can create a landing page
    @app.route("/")
    def index():
        return redirect(url_for("pollution"))
    # this is based on the figma design
    # Pollution & Marine Health
    @app.route("/pollution")
    def pollution():
        return send_from_directory(app.static_folder, "pollution.html")
    # Climate Impact & Erosion Trends
    @app.route("/climate")
    def climate():
        return send_from_directory(app.static_folder, "climate.html")
    # Water Resources & Availability
    @app.route("/water")
    def water():
        return send_from_directory(app.static_folder, "water.html")

    return app