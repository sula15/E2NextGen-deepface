import sys
from unittest.mock import MagicMock

# Mock deepface before importing app.services.face_service
deepface_mock = MagicMock()
sys.modules['deepface'] = deepface_mock
sys.modules['deepface.DeepFace'] = deepface_mock

import pytest
import numpy as np
import time

# Now import the class
from app.services.face_service import FaceRecognitionService

class TestFaceOptimization:

    @pytest.fixture
    def service(self):
        service = FaceRecognitionService(model_name='Facenet', distance_metric='cosine')
        return service

    def test_distance_calculation_performance(self):
        """
        Benchmark the distance calculation logic (vectorized)
        """
        db_size = 1000
        dim = 128

        # Random vectors
        db_embeddings = np.random.rand(db_size, dim)
        target_embedding = np.random.rand(dim)

        start_time = time.time()

        # Vectorized cosine distance simulation
        dot_products = np.dot(db_embeddings, target_embedding)
        norm_target = np.linalg.norm(target_embedding)
        norms_db = np.linalg.norm(db_embeddings, axis=1)
        similarities = dot_products / (norms_db * norm_target)
        distances = 1 - similarities

        end_time = time.time()
        duration = end_time - start_time

        print(f"\nTime to match against {db_size} users (vectorized): {duration*1000:.4f} ms")
        assert duration < 0.1  # Should be extremely fast (< 10ms usually)

    def test_recognize_from_embeddings_vectorized(self, service):
        """
        Test the actual vectorized recognize_from_embeddings method
        """
        # Mock extract_embedding
        service.extract_embedding = MagicMock(return_value=[1.0, 0.0])
        service._get_default_threshold = MagicMock(return_value=0.5)

        known = [
            {'user_id': 'u1', 'embedding': [1.0, 0.0]},     # Same -> dist 0
            {'user_id': 'u2', 'embedding': [0.0, 1.0]},     # Orthogonal -> dist 1
            {'user_id': 'u3', 'embedding': [0.707, 0.707]}  # 45 deg -> dist ~0.29
        ]

        # u3 distance: 1 - (0.707*1 + 0.707*0) / (1*1) = 1 - 0.707 = 0.293
        # Threshold 0.5 should include u1 and u3

        results = service.recognize_from_embeddings("dummy_path", known)

        assert len(results) == 2
        assert results[0]['user_id'] == 'u1'
        assert np.isclose(results[0]['distance'], 0.0)
        assert results[1]['user_id'] == 'u3'
        assert results[1]['distance'] < 0.5
