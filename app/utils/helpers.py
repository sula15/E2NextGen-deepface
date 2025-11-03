import os
import uuid
import base64
from pathlib import Path
from werkzeug.utils import secure_filename
from flask import current_app
import logging

logger = logging.getLogger(__name__)


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']


def save_uploaded_file(file_data, filename=None, subfolder=None):
    """
    Save uploaded file and return path
    
    Args:
        file_data: File data (base64 string or file object)
        filename: Optional filename
        subfolder: Optional subfolder within uploads
        
    Returns:
        Path to saved file
    """
    try:
        if filename is None:
            filename = f"{uuid.uuid4()}.jpg"
        else:
            filename = secure_filename(filename)
        
        # Determine save directory
        if subfolder:
            save_dir = Path(current_app.config['UPLOAD_FOLDER']) / subfolder
            save_dir.mkdir(exist_ok=True)
        else:
            save_dir = Path(current_app.config['UPLOAD_FOLDER'])
        
        filepath = save_dir / filename
        
        # Handle base64 string
        if isinstance(file_data, str):
            # Remove data URL prefix if present
            if ',' in file_data:
                file_data = file_data.split(',')[1]
            
            img_data = base64.b64decode(file_data)
            with open(filepath, 'wb') as f:
                f.write(img_data)
        else:
            # Handle file object
            file_data.save(str(filepath))
        
        logger.info(f"File saved: {filepath}")
        return str(filepath)
        
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        raise Exception(f"File save failed: {str(e)}")


def delete_file(filepath):
    """Safely delete a file"""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"File deleted: {filepath}")
    except Exception as e:
        logger.error(f"Error deleting file {filepath}: {str(e)}")


def encode_image_to_base64(image_path):
    """Encode image file to base64 string"""
    try:
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding image: {str(e)}")
        raise Exception(f"Image encoding failed: {str(e)}")
