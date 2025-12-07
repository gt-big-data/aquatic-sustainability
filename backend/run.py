import os
from app import create_app

# Define the app object at module level so Gunicorn can import it
app = create_app()

# Only run Flask’s dev server if executed directly (not under Gunicorn)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
