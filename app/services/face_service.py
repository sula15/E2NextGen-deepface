from deepface import DeepFace
import cv2
import numpy as np
from pathlib import Path
import base64
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class FaceRecognitionService:
    """Service for face recognition operations using DeepFace"""
    
    def __init__(self, model_name='Facenet', detector_backend='opencv', distance_metric='cosine'):
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.distance_metric = distance_metric
        logger.info(f"FaceRecognitionService initialized with model={model_name}, detector={detector_backend}")
    
    def decode_base64_image(self, base64_string: str) -> np.ndarray:
        """
        Decode base64 image to numpy array
        
        Args:
            base64_string: Base64 encoded image string
            
        Returns:
            numpy array of the image
        """
        try:
            # Remove data URL prefix if present
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            img_data = base64.b64decode(base64_string)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("Failed to decode image")
            
            return img
        except Exception as e:
            logger.error(f"Error decoding base64 image: {str(e)}")
            raise Exception(f"Image decoding failed: {str(e)}")
    
    def verify_faces(self, img1_path: str, img2_path: str) -> Dict:
        """
        Compare two faces (1:1 matching)
        
        Args:
            img1_path: Path to first image
            img2_path: Path to second image
            
        Returns:
            Dictionary with verification results
        """
        try:
            logger.info(f"Verifying faces: {img1_path} vs {img2_path}")
            
            result = DeepFace.verify(
                img1_path=img1_path,
                img2_path=img2_path,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                distance_metric=self.distance_metric,
                enforce_detection=False
            )
            
            return {
                'verified': result['verified'],
                'distance': result['distance'],
                'threshold': result['threshold'],
                'model': self.model_name,
                'detector': self.detector_backend,
                'similarity': 1 - (result['distance'] / result['threshold'])
            }
        except Exception as e:
            logger.error(f"Face verification failed: {str(e)}")
            raise Exception(f"Verification failed: {str(e)}")
    
    def recognize_face(self, img_path: str, db_path: str) -> List[Dict]:
        """
        Identify face from database (1:N matching)
        
        Args:
            img_path: Path to image to recognize
            db_path: Path to face database directory
            
        Returns:
            List of matches with identity and distance
        """
        try:
            logger.info(f"Recognizing face from: {img_path}")
            
            dfs = DeepFace.find(
                img_path=img_path,
                db_path=db_path,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                distance_metric=self.distance_metric,
                enforce_detection=False,
                silent=True
            )
            
            if len(dfs) > 0 and not dfs[0].empty:
                results = []
                for _, row in dfs[0].iterrows():
                    # Extract user_id from identity path
                    identity_path = Path(row['identity'])
                    user_id = identity_path.parent.name
                    
                    results.append({
                        'user_id': user_id,
                        'identity': str(row['identity']),
                        'distance': float(row[self.distance_metric]),
                        'threshold': float(row['threshold']),
                        'verified': row[self.distance_metric] < row['threshold']
                    })
                
                # Sort by distance (best match first)
                results.sort(key=lambda x: x['distance'])
                logger.info(f"Found {len(results)} matches")
                return results
            
            logger.info("No matches found")
            return []
            
        except Exception as e:
            logger.error(f"Face recognition failed: {str(e)}")
            raise Exception(f"Recognition failed: {str(e)}")
    
    def extract_embedding(self, img_path: str) -> Optional[List[float]]:
        """
        Extract face embedding vector
        
        Args:
            img_path: Path to image
            
        Returns:
            List of floats representing the embedding vector
        """
        try:
            logger.info(f"Extracting embedding from: {img_path}")
            
            embedding_objs = DeepFace.represent(
                img_path=img_path,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=False
            )
            
            if embedding_objs and len(embedding_objs) > 0:
                embedding = embedding_objs[0]['embedding']
                logger.info(f"Embedding extracted: {len(embedding)} dimensions")
                return embedding
            
            logger.warning("No face detected for embedding extraction")
            return None
            
        except Exception as e:
            logger.error(f"Embedding extraction failed: {str(e)}")
            raise Exception(f"Embedding extraction failed: {str(e)}")
    
    def analyze_face(self, img_path: str) -> Dict:
        """
        Analyze face for attributes (age, gender, emotion, race)
        
        Args:
            img_path: Path to image
            
        Returns:
            Dictionary with analysis results
        """
        try:
            logger.info(f"Analyzing face: {img_path}")
            
            analysis = DeepFace.analyze(
                img_path=img_path,
                actions=['age', 'gender', 'emotion', 'race'],
                detector_backend=self.detector_backend,
                enforce_detection=False,
                silent=True
            )
            
            if analysis and len(analysis) > 0:
                return analysis[0]
            
            return {}
            
        except Exception as e:
            logger.error(f"Face analysis failed: {str(e)}")
            raise Exception(f"Analysis failed: {str(e)}")
