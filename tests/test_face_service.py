import sys
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

# Mock heavy dependencies before import
sys.modules['deepface'] = MagicMock()
sys.modules['cv2'] = MagicMock()
sys.modules['pandas'] = MagicMock()

# Now we can import
from app.services.face_service import FaceRecognitionService

class TestFaceService(unittest.TestCase):
    def setUp(self):
        self.service = FaceRecognitionService()

    def test_compute_face_matches_cosine(self):
        """Test computing matches using cosine distance"""
        # A: [1, 0], B: [0, 1]. Cosine distance = 1.0 (orthagonal)
        # A: [1, 0], C: [1, 0]. Cosine distance = 0.0 (identical)

        known_embeddings = [
            {'user_id': 'user1', 'embedding': [1.0, 0.0]},
            {'user_id': 'user2', 'embedding': [0.0, 1.0]},
            {'user_id': 'user3', 'embedding': [0.70710678, 0.70710678]}, # 45 degrees
        ]

        target = [1.0, 0.0]

        # This method doesn't exist yet, but we will add it
        matches = self.service.compute_face_matches(target, known_embeddings)

        self.assertEqual(len(matches), 3)
        self.assertEqual(matches[0]['user_id'], 'user1')
        self.assertAlmostEqual(matches[0]['distance'], 0.0, places=5)

        self.assertEqual(matches[1]['user_id'], 'user3')
        # distance = 1 - cos(theta). cos(0) = 1 -> dist=0. cos(45) ~= 0.707 -> dist ~= 0.293
        # Wait, DeepFace usually uses Cosine Distance = 1 - Cosine Similarity.
        # Cosine Sim = (A . B) / (||A|| ||B||)
        # 1.0 * 0.707 / (1 * 1) = 0.707. Dist = 1 - 0.707 = 0.293
        self.assertAlmostEqual(matches[1]['distance'], 1.0 - 0.70710678, places=5)

        self.assertEqual(matches[2]['user_id'], 'user2')
        # Dist to user2: orthogonal, dist = 1.0
        self.assertAlmostEqual(matches[2]['distance'], 1.0, places=5)

    def test_compute_face_matches_euclidean(self):
        """Test computing matches using euclidean distance"""
        self.service.distance_metric = 'euclidean'

        known_embeddings = [
            {'user_id': 'user1', 'embedding': [3.0, 0.0]},
            {'user_id': 'user2', 'embedding': [0.0, 4.0]},
        ]

        target = [0.0, 0.0]

        matches = self.service.compute_face_matches(target, known_embeddings)

        # Dist to user1: sqrt((3-0)^2 + 0) = 3
        # Dist to user2: sqrt(0 + (4-0)^2) = 4

        self.assertEqual(matches[0]['user_id'], 'user1')
        self.assertEqual(matches[0]['distance'], 3.0)
        self.assertEqual(matches[1]['user_id'], 'user2')
        self.assertEqual(matches[1]['distance'], 4.0)
