import sys
from unittest.mock import MagicMock

# Mock heavy dependencies before they are imported
sys.modules['deepface'] = MagicMock()
sys.modules['cv2'] = MagicMock()
sys.modules['pandas'] = MagicMock()
sys.modules['flask'] = MagicMock()
sys.modules['flask_sqlalchemy'] = MagicMock()
sys.modules['flask_cors'] = MagicMock()
sys.modules['dotenv'] = MagicMock()

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from app.services.face_service import FaceRecognitionService

class TestVectorizedRecognition:

    @pytest.fixture
    def service(self):
        return FaceRecognitionService(model_name='Facenet', distance_metric='cosine')

    def test_cosine_distance_calculation(self, service):
        # This test verifies the math helper we will add
        if not hasattr(service, '_find_cosine_distance'):
            pytest.skip("Method _find_cosine_distance not implemented yet")

        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        dist = service._find_cosine_distance(v1, v2)
        # Cosine distance between orthogonal vectors is 1.0
        assert abs(dist - 1.0) < 1e-6

        v3 = np.array([1.0, 0.0])
        dist_same = service._find_cosine_distance(v1, v3)
        # Cosine distance between identical vectors is 0.0
        assert abs(dist_same - 0.0) < 1e-6

    def test_vectorized_recognition_integration(self, service):
        # Verify that recognize_face uses the optimized path when known_embeddings is provided
        known_embeddings = [
            {'user_id': 'user1', 'identity': 'user1_img', 'embedding': [1.0, 0.0]},
            {'user_id': 'user2', 'identity': 'user2_img', 'embedding': [0.0, 1.0]}
        ]

        # Query close to user1
        # [0.99, 0.01] is very close to [1.0, 0.0] (cosine distance small)
        query_embedding = [0.99, 0.01]

        with patch.object(service, 'extract_embedding', return_value=query_embedding):
            # We expect recognize_face to handle the known_embeddings arg
            try:
                results = service.recognize_face("dummy_path", "dummy_db", known_embeddings=known_embeddings)
            except TypeError:
                pytest.fail("recognize_face does not accept known_embeddings yet")

            assert len(results) >= 1
            assert results[0]['user_id'] == 'user1'
            # Check if distance is small
            assert results[0]['distance'] < 0.1
