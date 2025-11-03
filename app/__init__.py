from flask import Flask
from flask_cors import CORS
from config import config
from app.models.database import db
import logging
from pathlib import Path


def create_app(config_name='default'):
    """Application factory pattern"""
    app = Flask(__name__)
    
    # Load configuration
    app.config.from_object(config[config_name])
    
    # Initialize extensions
    db.init_app(app)
    CORS(app)
    
    # Setup logging
    setup_logging(app)
    
    # Create database tables
    with app.app_context():
        db.create_all()
    
    # Register blueprints
    from app.routes import face_routes
    app.register_blueprint(face_routes.bp)
    
    # Health check endpoint
    @app.route('/health')
    def health_check():
        return {
            'status': 'healthy',
            'version': '1.0.0',
            'model': app.config['DEEPFACE_MODEL'],
            'detector': app.config['DEEPFACE_DETECTOR']
        }, 200
    
    # Root endpoint
    @app.route('/')
    def index():
        return {
            'message': 'DeepFace Face Recognition API',
            'version': '1.0.0',
            'endpoints': {
                'health': '/health',
                'verify': '/api/v1/verify',
                'recognize': '/api/v1/recognize',
                'enroll': '/api/v1/enroll',
                'users': '/api/v1/users',
                'attendance': '/api/v1/attendance'
            }
        }, 200
    
    return app


def setup_logging(app):
    """Setup application logging"""
    log_file = Path(app.config['LOG_FOLDER']) / 'app.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    app.logger.setLevel(logging.INFO)
