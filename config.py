import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Base configuration"""
    # Flask
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-this'
    
    # Paths
    BASE_DIR = Path(__file__).parent
    UPLOAD_FOLDER = BASE_DIR / 'uploads'
    DATABASE_FOLDER = BASE_DIR / 'face_database'
    LOG_FOLDER = BASE_DIR / 'logs'
    
    # File upload settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    
    # DeepFace settings
    DEEPFACE_MODEL = os.environ.get('DEEPFACE_MODEL', 'Facenet')
    DEEPFACE_DETECTOR = os.environ.get('DEEPFACE_DETECTOR', 'opencv')
    DISTANCE_METRIC = os.environ.get('DISTANCE_METRIC', 'cosine')
    
    # Database
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        f'sqlite:///{BASE_DIR / "face_recognition.db"}'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # AWS settings (for future phases)
    AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
    AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')
    S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')
    
    # Redis settings (for future caching)
    REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
    
    # Create necessary directories
    UPLOAD_FOLDER.mkdir(exist_ok=True)
    DATABASE_FOLDER.mkdir(exist_ok=True)
    LOG_FOLDER.mkdir(exist_ok=True)


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False


class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
