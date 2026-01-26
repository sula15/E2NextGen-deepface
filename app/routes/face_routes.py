from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import base64
from pathlib import Path
import uuid
import os
from datetime import datetime

from app.services.face_service import FaceRecognitionService
from app.models.database import db, User, AttendanceRecord
from app.utils.helpers import save_uploaded_file, delete_file
import logging

logger = logging.getLogger(__name__)

bp = Blueprint('face', __name__, url_prefix='/api/v1')

# Initialize face service (will be configured from app config)
def get_face_service():
    return FaceRecognitionService(
        model_name=current_app.config['DEEPFACE_MODEL'],
        detector_backend=current_app.config['DEEPFACE_DETECTOR'],
        distance_metric=current_app.config['DISTANCE_METRIC']
    )


@bp.route('/verify', methods=['POST'])
def verify():
    """
    Verify if two faces match (1:1 matching)
    
    Request body:
    {
        "image1": "base64_encoded_image",
        "image2": "base64_encoded_image"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'image1' not in data or 'image2' not in data:
            return jsonify({
                'success': False,
                'error': 'Both image1 and image2 are required'
            }), 400
        
        face_service = get_face_service()
        
        # Save images temporarily
        img1_path = save_uploaded_file(data['image1'], f"verify1_{uuid.uuid4()}.jpg")
        img2_path = save_uploaded_file(data['image2'], f"verify2_{uuid.uuid4()}.jpg")
        
        try:
            # Verify faces
            result = face_service.verify_faces(img1_path, img2_path)
            
            return jsonify({
                'success': True,
                'result': result
            }), 200
            
        finally:
            # Cleanup temporary files
            delete_file(img1_path)
            delete_file(img2_path)
        
    except Exception as e:
        logger.error(f"Verify endpoint error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@bp.route('/recognize', methods=['POST'])
def recognize():
    """
    Recognize face from database (1:N matching)
    
    Request body:
    {
        "image": "base64_encoded_image",
        "log_attendance": false (optional)
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'Image is required'
            }), 400
        
        face_service = get_face_service()
        
        # Save uploaded image temporarily
        img_path = save_uploaded_file(data['image'], f"recognize_{uuid.uuid4()}.jpg")
        
        try:
            # Recognize face
            # Fetch all active users with embeddings
            users = User.query.filter(User.embedding.isnot(None), User.is_active == True).all()
            known_embeddings = [
                {'user_id': user.user_id, 'embedding': user.embedding}
                for user in users
            ]

            results = face_service.recognize_from_memory(img_path, known_embeddings)
            
            if results and len(results) > 0:
                # Get best match
                best_match = results[0]
                user = User.query.filter_by(user_id=best_match['user_id']).first()
                
                if user:
                    # Optionally log attendance
                    if data.get('log_attendance', False):
                        attendance = AttendanceRecord(
                            user_id=user.id,
                            confidence_score=1 - (best_match['distance'] / best_match['threshold']) if best_match['threshold'] > 0 else 0,
                            location=data.get('location', 'Unknown'),
                            status='present',
                            image_path=img_path  # Keep image for record
                        )
                        db.session.add(attendance)
                        db.session.commit()
                        img_path = None  # Don't delete if logged
                    
                    return jsonify({
                        'success': True,
                        'recognized': True,
                        'user': user.to_dict(),
                        'confidence': 1 - (best_match['distance'] / best_match['threshold']) if best_match['threshold'] > 0 else 0,
                        'distance': best_match['distance'],
                        'threshold': best_match['threshold'],
                        'all_matches': results[:5]  # Return top 5 matches
                    }), 200
            
            return jsonify({
                'success': True,
                'recognized': False,
                'message': 'No matching face found in database'
            }), 200
            
        finally:
            # Cleanup temporary file if not kept for attendance
            if img_path:
                delete_file(img_path)
        
    except Exception as e:
        logger.error(f"Recognize endpoint error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@bp.route('/enroll', methods=['POST'])
def enroll():
    """
    Enroll new face in database
    
    Request body:
    {
        "user_id": "unique_user_id",
        "name": "Full Name",
        "email": "email@example.com",
        "phone": "+1234567890",
        "department": "Engineering",
        "image": "base64_encoded_image"
    }
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        if not data or 'user_id' not in data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'user_id and image are required'
            }), 400
        
        user_id = data['user_id']
        name = data.get('name', '')
        email = data.get('email', '')
        phone = data.get('phone', '')
        department = data.get('department', '')
        
        # Check if user already exists
        existing_user = User.query.filter_by(user_id=user_id).first()
        if existing_user:
            return jsonify({
                'success': False,
                'error': f'User with ID {user_id} already enrolled'
            }), 400
        
        face_service = get_face_service()
        
        # Create user directory in face database
        user_dir = Path(current_app.config['DATABASE_FOLDER']) / user_id
        user_dir.mkdir(exist_ok=True)
        
        # Save face image
        filename = f"{user_id}_{uuid.uuid4()}.jpg"
        img_path = user_dir / filename
        
        img_data = base64.b64decode(data['image'].split(',')[1] if ',' in data['image'] else data['image'])
        with open(img_path, 'wb') as f:
            f.write(img_data)
        
        try:
            # Extract embedding to verify face is detected
            embedding = face_service.extract_embedding(str(img_path))
            
            if embedding is None:
                delete_file(str(img_path))
                return jsonify({
                    'success': False,
                    'error': 'No face detected in image. Please provide a clear face image.'
                }), 400
            
            # Save to database
            new_user = User(
                user_id=user_id,
                name=name,
                email=email,
                phone=phone,
                department=department,
                face_image_path=str(img_path),
                embedding=embedding
            )
            db.session.add(new_user)
            db.session.commit()
            
            logger.info(f"User enrolled: {user_id}")
            
            return jsonify({
                'success': True,
                'user_id': user_id,
                'message': 'User enrolled successfully',
                'embedding_size': len(embedding),
                'user': new_user.to_dict()
            }), 201
            
        except Exception as e:
            # Cleanup on error
            delete_file(str(img_path))
            raise e
        
    except Exception as e:
        logger.error(f"Enroll endpoint error: {str(e)}")
        db.session.rollback()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@bp.route('/users', methods=['GET'])
def get_users():
    """Get all enrolled users"""
    try:
        users = User.query.filter_by(is_active=True).all()
        return jsonify({
            'success': True,
            'count': len(users),
            'users': [user.to_dict() for user in users]
        }), 200
    except Exception as e:
        logger.error(f"Get users error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@bp.route('/users/<user_id>', methods=['GET'])
def get_user(user_id):
    """Get specific user details"""
    try:
        user = User.query.filter_by(user_id=user_id).first()
        if not user:
            return jsonify({
                'success': False,
                'error': 'User not found'
            }), 404
        
        return jsonify({
            'success': True,
            'user': user.to_dict()
        }), 200
    except Exception as e:
        logger.error(f"Get user error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@bp.route('/users/<user_id>', methods=['DELETE'])
def delete_user(user_id):
    """Delete a user"""
    try:
        user = User.query.filter_by(user_id=user_id).first()
        if not user:
            return jsonify({
                'success': False,
                'error': 'User not found'
            }), 404
        
        # Delete face image
        if user.face_image_path and os.path.exists(user.face_image_path):
            delete_file(user.face_image_path)
            # Try to delete user directory if empty
            try:
                user_dir = Path(user.face_image_path).parent
                if user_dir.exists() and not any(user_dir.iterdir()):
                    user_dir.rmdir()
            except:
                pass
        
        db.session.delete(user)
        db.session.commit()
        
        logger.info(f"User deleted: {user_id}")
        
        return jsonify({
            'success': True,
            'message': f'User {user_id} deleted successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Delete user error: {str(e)}")
        db.session.rollback()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@bp.route('/attendance', methods=['GET'])
def get_attendance():
    """Get attendance records"""
    try:
        # Optional query parameters
        user_id = request.args.get('user_id')
        date = request.args.get('date')  # Format: YYYY-MM-DD
        limit = request.args.get('limit', 100, type=int)
        
        query = AttendanceRecord.query
        
        if user_id:
            user = User.query.filter_by(user_id=user_id).first()
            if user:
                query = query.filter_by(user_id=user.id)
        
        if date:
            try:
                date_obj = datetime.strptime(date, '%Y-%m-%d')
                query = query.filter(
                    db.func.date(AttendanceRecord.timestamp) == date_obj.date()
                )
            except ValueError:
                return jsonify({
                    'success': False,
                    'error': 'Invalid date format. Use YYYY-MM-DD'
                }), 400
        
        records = query.order_by(AttendanceRecord.timestamp.desc()).limit(limit).all()
        
        return jsonify({
            'success': True,
            'count': len(records),
            'records': [record.to_dict() for record in records]
        }), 200
        
    except Exception as e:
        logger.error(f"Get attendance error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@bp.route('/cache/clear', methods=['POST'])
def clear_cache():
    """Manually clear DeepFace cache"""
    try:
        # Cache clearing is no longer needed as we use database embeddings
        return jsonify({
            'success': True,
            'message': 'Cache cleared successfully (No-op: Database embeddings used)'
        }), 200
    except Exception as e:
        logger.error(f"Clear cache error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400