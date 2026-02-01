import sys
from unittest.mock import MagicMock
import numpy as np
import pytest

# Mock dependencies before importing the service
sys.modules["deepface"] = MagicMock()
sys.modules["cv2"] = MagicMock()
sys.modules["pandas"] = MagicMock()

# We also need to mock flask related things if they are imported,
# but FaceRecognitionService seems to be independent of Flask (except maybe logging config?)
# It imports logging.

from app.services.face_service import FaceRecognitionService

class TestOptimization:
    def test_recognize_from_embeddings_cosine(self):
        """Test vectorized recognition with cosine distance"""
        service = FaceRecognitionService(distance_metric='cosine')

        # Mock extract_embedding to return a fixed vector [1, 0]
        # We'll use 2D vectors for simplicity.
        service.extract_embedding = MagicMock(return_value=[1.0, 0.0])

        # Known embeddings:
        # User A: [1, 0] -> Identical. Distance should be 0.
        # User B: [0, 1] -> Orthogonal. Cosine sim = 0. Distance = 1.
        # User C: [-1, 0] -> Opposite. Cosine sim = -1. Distance = 2.
        # User D: [0.7071, 0.7071] -> 45 degrees. Cosine sim ~0.7071. Distance ~0.2929.

        known_embeddings = [
            {'user_id': 'user_a', 'embedding': [1.0, 0.0]},
            {'user_id': 'user_b', 'embedding': [0.0, 1.0]},
            {'user_id': 'user_c', 'embedding': [-1.0, 0.0]},
            {'user_id': 'user_d', 'embedding': [0.70710678, 0.70710678]},
        ]

        results = service.recognize_from_embeddings("fake_path.jpg", known_embeddings)

        # Sort by distance
        results.sort(key=lambda x: x['distance'])

        # Check first match (User A)
        assert results[0]['user_id'] == 'user_a'
        assert abs(results[0]['distance'] - 0.0) < 1e-6
        assert results[0]['verified'] is True

        # Check User D (should be match if threshold > 0.29)
        # Default threshold for cosine/Facenet is 0.40
        match_d = next(r for r in results if r['user_id'] == 'user_d')
        assert abs(match_d['distance'] - (1 - 0.70710678)) < 1e-4
        assert match_d['verified'] is True

        # Check User B (Distance 1.0)
        match_b = next(r for r in results if r['user_id'] == 'user_b')
        assert abs(match_b['distance'] - 1.0) < 1e-6
        assert match_b['verified'] is False

    def test_recognize_from_embeddings_euclidean(self):
        """Test vectorized recognition with euclidean distance"""
        # Note: Facenet euclidean threshold is usually around 10.
        service = FaceRecognitionService(distance_metric='euclidean', model_name='Facenet')

        service.extract_embedding = MagicMock(return_value=[1.0, 0.0])

        known_embeddings = [
            {'user_id': 'user_a', 'embedding': [1.0, 0.0]}, # dist 0
            {'user_id': 'user_b', 'embedding': [4.0, 0.0]}, # dist 3
            {'user_id': 'user_c', 'embedding': [1.0, 4.0]}, # dist 4
        ]

        results = service.recognize_from_embeddings("fake_path.jpg", known_embeddings)
        results.sort(key=lambda x: x['distance'])

        assert results[0]['user_id'] == 'user_a'
        assert results[0]['distance'] == 0.0

        assert results[1]['user_id'] == 'user_b'
        assert results[1]['distance'] == 3.0

        assert results[2]['user_id'] == 'user_c'
        assert results[2]['distance'] == 4.0

    def test_recognize_from_embeddings_empty(self):
        """Test with no known embeddings"""
        service = FaceRecognitionService()
        service.extract_embedding = MagicMock(return_value=[1.0, 0.0])

        results = service.recognize_from_embeddings("fake_path.jpg", [])
        assert results == []

    def test_recognize_from_embeddings_no_face(self):
        """Test when input image has no face"""
        service = FaceRecognitionService()
        service.extract_embedding = MagicMock(return_value=None)

        results = service.recognize_from_embeddings("fake_path.jpg", [{'user_id': 'a', 'embedding': [1,0]}])
        assert results == []
