import pytest
from app.services.face_service import FaceRecognitionService
import numpy as np

class TestFaceService:
    @pytest.fixture
    def service(self):
        return FaceRecognitionService(model_name='Facenet', distance_metric='cosine')

    def test_calculate_distance_cosine(self, service):
        # Two identical vectors -> distance 0
        v1 = [0.1, 0.2, 0.3]
        d = service.calculate_distance(v1, v1)
        # Cosine distance is 1 - similarity. Sim of identical is 1. Dist is 0.
        assert d < 1e-6

        # Orthogonal vectors -> distance 1
        v2 = [1, 0, 0]
        v3 = [0, 1, 0]
        d = service.calculate_distance(v2, v3)
        assert abs(d - 1.0) < 1e-6

    def test_calculate_distance_euclidean(self, service):
        service.distance_metric = 'euclidean'
        v1 = [0, 0, 0]
        v2 = [3, 4, 0]
        d = service.calculate_distance(v1, v2)
        assert abs(d - 5.0) < 1e-6

    def test_recognize_from_memory(self, service, mocker):
        # Mock extract_embedding
        mocker.patch.object(service, 'extract_embedding', return_value=[1.0, 0.0])

        known_embeddings = [
            {'user_id': 'user1', 'embedding': [1.0, 0.0]}, # Perfect match
            {'user_id': 'user2', 'embedding': [0.0, 1.0]}, # Far
        ]

        # Mock threshold to ensure user1 passes. Cosine dist: 0 for user1, 1 for user2.
        mocker.patch.object(service, '_get_default_threshold', return_value=0.5)

        results = service.recognize_from_memory('dummy_path', known_embeddings)

        assert len(results) == 1
        assert results[0]['user_id'] == 'user1'
        assert results[0]['verified'] is True
        assert results[0]['distance'] < 1e-6
