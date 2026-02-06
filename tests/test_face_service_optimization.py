
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import sys

# Mock DeepFace before importing service
sys.modules['deepface'] = MagicMock()
from app.services.face_service import FaceRecognitionService

class TestFaceServiceOptimization:

    def setup_method(self):
        self.service = FaceRecognitionService()
        # Mock default threshold return
        self.service._get_default_threshold = MagicMock(return_value=0.4)

    def test_compute_face_matches_cosine(self):
        # Setup known embeddings
        # User 1: [1, 0]
        # User 2: [0, 1]
        known_embeddings = [
            {'user_id': 'user1', 'embedding': [1.0, 0.0], 'face_image_path': 'path/to/img1.jpg'},
            {'user_id': 'user2', 'embedding': [0.0, 1.0], 'face_image_path': 'path/to/img2.jpg'}
        ]

        # Target: [1, 0] -> Should match user1 exactly (dist=0)
        target_embedding = [1.0, 0.0]

        results = self.service.compute_face_matches(target_embedding, known_embeddings)

        assert len(results) >= 1
        assert results[0]['user_id'] == 'user1'
        assert results[0]['distance'] < 1e-5

        # Target: [0.707, 0.707] -> Should match both with some distance
        # Cosine dist = 1 - (A.B) / (|A||B|)
        # A=[1,0], B=[0.7,0.7] -> dot=0.7, norms=1, 1 -> dist = 1-0.7 = 0.3 < 0.4

        target_embedding = [0.70710678, 0.70710678]
        results = self.service.compute_face_matches(target_embedding, known_embeddings)

        assert len(results) == 2
        # Both distances should be around 0.293
        assert abs(results[0]['distance'] - (1 - 0.70710678)) < 1e-4

    def test_compute_face_matches_threshold(self):
        known_embeddings = [
            {'user_id': 'user1', 'embedding': [1.0, 0.0]}
        ]

        # Target: [-1, 0] -> Opposite direction. Cosine dist = 1 - (-1) = 2.
        target_embedding = [-1.0, 0.0]

        results = self.service.compute_face_matches(target_embedding, known_embeddings)

        # Should be filtered out by threshold (0.4)
        assert len(results) == 0

    def test_empty_input(self):
        assert self.service.compute_face_matches([], []) == []
        assert self.service.compute_face_matches([1,0], []) == []

    @patch('app.services.face_service.FaceRecognitionService.extract_embedding')
    def test_recognize_face_uses_optimization(self, mock_extract):
        mock_extract.return_value = [1.0, 0.0]

        known_embeddings = [
            {'user_id': 'user1', 'embedding': [1.0, 0.0], 'face_image_path': 'path/1'}
        ]

        # We don't care about db_path since we provide known_embeddings
        results = self.service.recognize_face('fake_img_path', 'fake_db_path', known_embeddings=known_embeddings)

        assert len(results) == 1
        assert results[0]['user_id'] == 'user1'

        # Verify deepface.find was NOT called
        # Since we mocked the module, we can check if it was accessed?
        # Actually in recognize_face logic:
        # if known_embeddings: ... return self.compute_face_matches(...)
        # So it returns early.
