from deepface import DeepFace
import cv2
import numpy as np
from pathlib import Path
import base64
from typing import Dict, List, Optional
import logging
import pandas as pd

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
                'similarity': 1 - (result['distance'] / result['threshold']) if result['threshold'] > 0 else 0
            }
        except Exception as e:
            logger.error(f"Face verification failed: {str(e)}")
            raise Exception(f"Verification failed: {str(e)}")
    
    def recognize_face(self, img_path: str, db_path: str, known_users: Optional[List[Dict]] = None) -> List[Dict]:
        """
        Identify face from database (1:N matching)
        
        Args:
            img_path: Path to image to recognize
            db_path: Path to face database directory
            known_users: Optional list of users with pre-calculated embeddings for faster lookup
            
        Returns:
            List of matches with identity and distance
        """
        try:
            logger.info(f"Recognizing face from: {img_path}")

            # fast path: in-memory vectorized search
            if known_users is not None:
                logger.info(f"Using vectorized recognition with {len(known_users)} known users")
                embedding = self.extract_embedding(img_path)
                if embedding:
                    return self._vectorized_recognition(embedding, known_users)
                return []

            logger.info(f"Searching database: {db_path}")
            
            # Check if database has any users
            db_path_obj = Path(db_path)
            if not db_path_obj.exists() or not any(db_path_obj.iterdir()):
                logger.warning("Face database is empty")
                return []
            
            try:
                dfs = DeepFace.find(
                    img_path=img_path,
                    db_path=db_path,
                    model_name=self.model_name,
                    detector_backend=self.detector_backend,
                    distance_metric=self.distance_metric,
                    enforce_detection=False,
                    silent=True
                )
            except Exception as find_error:
                logger.error(f"DeepFace.find error: {str(find_error)}")
                # If find fails, try manual comparison
                return self._manual_recognition(img_path, db_path)
            
            results = []
            
            # Handle different return formats from DeepFace
            if isinstance(dfs, list) and len(dfs) > 0:
                df = dfs[0]
                
                if isinstance(df, pd.DataFrame) and not df.empty:
                    for _, row in df.iterrows():
                        try:
                            # Extract user_id from identity path
                            identity_path = Path(row['identity'])
                            user_id = identity_path.parent.name
                            
                            # Get distance value based on metric
                            distance_col = f'{self.model_name}_{self.distance_metric}'
                            if distance_col in row:
                                distance = float(row[distance_col])
                            elif self.distance_metric in row:
                                distance = float(row[self.distance_metric])
                            else:
                                # Fallback: try to find any distance column
                                distance_cols = [col for col in row.index if 'distance' in col.lower() or self.distance_metric in col.lower()]
                                if distance_cols:
                                    distance = float(row[distance_cols[0]])
                                else:
                                    logger.warning(f"Could not find distance column. Available columns: {list(row.index)}")
                                    continue
                            
                            # Get threshold
                            threshold_col = f'{self.model_name}_threshold'
                            if threshold_col in row:
                                threshold = float(row[threshold_col])
                            elif 'threshold' in row:
                                threshold = float(row['threshold'])
                            else:
                                # Use default threshold for the model
                                threshold = self._get_default_threshold()
                            
                            results.append({
                                'user_id': user_id,
                                'identity': str(row['identity']),
                                'distance': distance,
                                'threshold': threshold,
                                'verified': distance < threshold
                            })
                        except Exception as row_error:
                            logger.error(f"Error processing row: {str(row_error)}")
                            continue
                    
                    # Sort by distance (best match first)
                    results.sort(key=lambda x: x['distance'])
                    logger.info(f"Found {len(results)} matches")
            
            return results
            
        except Exception as e:
            logger.error(f"Face recognition failed: {str(e)}")
            raise Exception(f"Recognition failed: {str(e)}")
    
    def _vectorized_recognition(self, target_embedding: List[float], known_users: List[Dict]) -> List[Dict]:
        """
        Perform vectorized face recognition using numpy (fast in-memory)

        Args:
            target_embedding: Embedding of the face to recognize
            known_users: List of dicts with keys 'user_id', 'embedding', etc.

        Returns:
            List of matches
        """
        if not known_users:
            return []

        try:
            target = np.array(target_embedding)
            # Ensure known_users have embeddings
            valid_users = [u for u in known_users if u.get('embedding')]
            if not valid_users:
                return []

            known_embeddings = np.array([u['embedding'] for u in valid_users])

            if self.distance_metric == 'cosine':
                # Cosine distance = 1 - (A . B) / (||A|| * ||B||)
                target_norm = np.linalg.norm(target)
                target_normalized = target / target_norm if target_norm > 0 else target

                known_norms = np.linalg.norm(known_embeddings, axis=1, keepdims=True)
                # Avoid division by zero
                known_norms[known_norms == 0] = 1
                known_normalized = known_embeddings / known_norms

                similarities = np.dot(known_normalized, target_normalized)
                distances = 1 - similarities

            elif self.distance_metric in ['euclidean', 'euclidean_l2']:
                # Euclidean distance
                diff = known_embeddings - target
                distances = np.sqrt(np.sum(diff**2, axis=1))

            else:
                logger.warning(f"Metric {self.distance_metric} not supported for vectorized search. Using fallback.")
                return []

            results = []
            threshold = self._get_default_threshold()

            # Filter results
            for i, distance in enumerate(distances):
                if distance <= threshold:
                    user = valid_users[i]
                    results.append({
                        'user_id': user['user_id'],
                        'identity': user.get('name', user['user_id']),
                        'distance': float(distance),
                        'threshold': threshold,
                        'verified': True
                    })

            # Sort by distance
            results.sort(key=lambda x: x['distance'])
            logger.info(f"Vectorized recognition found {len(results)} matches")
            return results

        except Exception as e:
            logger.error(f"Vectorized recognition error: {str(e)}")
            return []

    def _manual_recognition(self, img_path: str, db_path: str) -> List[Dict]:
        """
        Manual recognition by comparing against all database images
        Fallback when DeepFace.find fails
        """
        try:
            logger.info("Using manual recognition fallback")
            results = []
            
            # Extract embedding from query image
            query_embedding = self.extract_embedding(img_path)
            if query_embedding is None:
                return []
            
            # Iterate through all user directories
            db_path_obj = Path(db_path)
            for user_dir in db_path_obj.iterdir():
                if not user_dir.is_dir():
                    continue
                
                user_id = user_dir.name
                
                # Compare against all images for this user
                for img_file in user_dir.glob('*.jpg'):
                    try:
                        result = self.verify_faces(img_path, str(img_file))
                        
                        results.append({
                            'user_id': user_id,
                            'identity': str(img_file),
                            'distance': result['distance'],
                            'threshold': result['threshold'],
                            'verified': result['verified']
                        })
                    except Exception as e:
                        logger.error(f"Error comparing with {img_file}: {str(e)}")
                        continue
            
            # Sort by distance
            results.sort(key=lambda x: x['distance'])
            logger.info(f"Manual recognition found {len(results)} matches")
            
            return results
            
        except Exception as e:
            logger.error(f"Manual recognition failed: {str(e)}")
            return []
    
    def _get_default_threshold(self) -> float:
        """Get default threshold for the current model and metric"""
        thresholds = {
            'VGG-Face': {'cosine': 0.40, 'euclidean': 0.60},
            'Facenet': {'cosine': 0.40, 'euclidean': 10},
            'Facenet512': {'cosine': 0.30, 'euclidean': 23.56},
            'ArcFace': {'cosine': 0.68, 'euclidean': 4.15},
            'Dlib': {'cosine': 0.07, 'euclidean': 0.6},
            'SFace': {'cosine': 0.593, 'euclidean': 10.734},
        }
        
        return thresholds.get(self.model_name, {}).get(self.distance_metric, 0.40)
    
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