import os
from app import create_app

app = create_app()

if __name__ == "__main__":
    # Bind to 0.0.0.0 so Render can access it
    # Use the port Render provides, defaulting to 5000 if not set
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)