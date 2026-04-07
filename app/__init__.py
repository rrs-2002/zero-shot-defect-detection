from flask import Flask

def create_app():
    app = Flask(__name__)
    
    # Configuration
    from src.config import Config
    app.config.from_object(Config)
    
    # Register Blueprints
    from .routes import main
    app.register_blueprint(main)
    
    return app
