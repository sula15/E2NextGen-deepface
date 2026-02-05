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

# We can now import the service, but we might need to handle other top-level imports
# The service imports:
# from deepface import DeepFace
# import cv2
# import numpy as np
# from pathlib import Path
# import base64
# from typing import Dict, List, Optional
# import logging
# import pandas as pd

from app.services.face_service import FaceRecognitionService

class TestFaceServiceOptimization:

    @pytest.fixture
    def face_service(self):
        # We need to ensure the mocks are set up correctly for the instance
        service = FaceRecognitionService(model_name='Facenet', distance_metric='euclidean')
        return service

    def test_compute_face_matches_empty(self, face_service):
        target = [0.1] * 128
        known = []
        results = face_service.compute_face_matches(target, known)
        assert results == []

    def test_compute_face_matches_euclidean(self, face_service):
        face_service.distance_metric = 'euclidean'

        target = [0.0, 0.0]
        known = [
            {'user_id': 'u1', 'embedding': [0.0, 0.0]}, # Dist 0
            {'user_id': 'u2', 'embedding': [0.0, 1.0]}, # Dist 1
            {'user_id': 'u3', 'embedding': [3.0, 4.0]}, # Dist 5
        ]

        results = face_service.compute_face_matches(target, known, threshold=2.0)

        assert len(results) == 2
        assert results[0]['user_id'] == 'u1'
        assert results[0]['distance'] == 0.0
        assert results[1]['user_id'] == 'u2'
        assert results[1]['distance'] == 1.0

    def test_compute_face_matches_cosine(self, face_service):
        face_service.distance_metric = 'cosine'

        target = [1.0, 0.0]
        known = [
            {'user_id': 'same', 'embedding': [1.0, 0.0]},
            {'user_id': 'ortho', 'embedding': [0.0, 1.0]},
            {'user_id': 'opposite', 'embedding': [-1.0, 0.0]},
        ]

        # Use large threshold to get all
        results = face_service.compute_face_matches(target, known, threshold=2.1)

        assert results[0]['user_id'] == 'same'
        # Check almost equal
        assert abs(results[0]['distance'] - 0.0) < 1e-6

        assert results[1]['user_id'] == 'ortho'
        assert abs(results[1]['distance'] - 1.0) < 1e-6

        assert results[2]['user_id'] == 'opposite'
        assert abs(results[2]['distance'] - 2.0) < 1e-6

    def test_compute_face_matches_no_embedding_in_list(self, face_service):
        target = [0.1] * 128
        known = [
            {'user_id': 'u1', 'embedding': None},
            {'user_id': 'u2'} # Missing key
        ]
        results = face_service.compute_face_matches(target, known)
        assert results == []
