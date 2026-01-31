import sys
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

# Mock heavy dependencies before they are imported
# These mocks prevent the actual libraries from being loaded,
# which solves missing dependency issues and speeds up tests.
sys.modules['deepface'] = MagicMock()
sys.modules['cv2'] = MagicMock()
sys.modules['pandas'] = MagicMock()
sys.modules['flask'] = MagicMock()
sys.modules['flask_sqlalchemy'] = MagicMock()
sys.modules['flask_cors'] = MagicMock()
sys.modules['dotenv'] = MagicMock()

# Import the service
# We need to make sure we can import app.services.face_service
# This import might trigger app/__init__.py, which imports config.py
from app.services.face_service import FaceRecognitionService

class TestFaceRecognitionPerf:
    def setup_method(self):
        self.service = FaceRecognitionService()
        # Mock extract_embedding to avoid DeepFace calls
        self.service.extract_embedding = MagicMock()

    def test_vectorized_recognition_cosine(self):
        """Test vectorized recognition with Cosine distance"""
        self.service.distance_metric = 'cosine'

        # Setup data
        query_vec = [1.0, 0.0]
        self.service.extract_embedding.return_value = query_vec

        known_users = [
            {'user_id': 'perfect_match', 'embedding': [1.0, 0.0]},      # dist = 0
            {'user_id': 'orthogonal', 'embedding': [0.0, 1.0]},         # dist = 1
            {'user_id': 'opposite', 'embedding': [-1.0, 0.0]},          # dist = 2 (1 - (-1)) = 2? No, cosine dist is usually 1 - sim. range [0, 2]
        ]

        # We expect the method to be implemented.
        if not hasattr(self.service, 'recognize_face_from_embeddings'):
            pytest.fail("Method recognize_face_from_embeddings not implemented yet")

        results = self.service.recognize_face_from_embeddings('dummy_path', known_users)

        assert len(results) > 0
        assert results[0]['user_id'] == 'perfect_match'
        assert abs(results[0]['distance'] - 0.0) < 1e-6

    def test_vectorized_recognition_euclidean(self):
        """Test vectorized recognition with Euclidean distance"""
        self.service.distance_metric = 'euclidean'

        query_vec = [0.0, 0.0]
        self.service.extract_embedding.return_value = query_vec

        known_users = [
            {'user_id': 'close', 'embedding': [1.0, 0.0]},       # dist = 1
            {'user_id': 'far', 'embedding': [0.0, 5.0]},         # dist = 5
            {'user_id': 'exact', 'embedding': [0.0, 0.0]},       # dist = 0
        ]

        if not hasattr(self.service, 'recognize_face_from_embeddings'):
             pytest.fail("Method recognize_face_from_embeddings not implemented yet")

        results = self.service.recognize_face_from_embeddings('dummy_path', known_users)

        assert len(results) > 0
        assert results[0]['user_id'] == 'exact'
        assert results[0]['distance'] == 0.0
        assert results[1]['user_id'] == 'close'
        assert results[1]['distance'] == 1.0

    def test_performance_comparison(self):
        """
        Micro-benchmark to demonstrate speedup.
        We simulate a larger dataset.
        """
        # Create 1000 dummy users with 128-d embeddings
        n_users = 1000
        dim = 128
        known_users = [
            {'user_id': f'u{i}', 'embedding': list(np.random.rand(dim))}
            for i in range(n_users)
        ]

        self.service.distance_metric = 'cosine'
        self.service.extract_embedding.return_value = list(np.random.rand(dim))

        import time

        if not hasattr(self.service, 'recognize_face_from_embeddings'):
             pytest.fail("Method recognize_face_from_embeddings not implemented yet")

        start_time = time.time()
        self.service.recognize_face_from_embeddings('dummy', known_users)
        duration = time.time() - start_time

        print(f"\nTime for {n_users} users: {duration:.6f}s")
        # Assert it's reasonably fast (e.g., under 50ms for 1000 users)
        assert duration < 0.05
