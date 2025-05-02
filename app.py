"""
Flask server for the interactive medical chat interface.
"""

from flask import Flask, render_template
from src.routes.v2_routes import v2_bp
from src.routes.v1_routes import v1_bp

# Create Flask app
app = Flask(__name__)

# Register blueprints
app.register_blueprint(v1_bp)
app.register_blueprint(v2_bp)


# Root route redirects to v2
@app.route("/")
def home():
    return render_template("chat_v2.html")


if __name__ == "__main__":
    app.run(debug=True)
