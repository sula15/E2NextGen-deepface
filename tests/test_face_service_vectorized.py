
import unittest
import sys
from unittest.mock import MagicMock, patch
import numpy as np

# Mock DeepFace before importing face_service
sys.modules['deepface'] = MagicMock()
sys.modules['cv2'] = MagicMock()

# Now we can import
from app.services.face_service import FaceRecognitionService

class TestVectorizedRecognition(unittest.TestCase):
    def setUp(self):
        self.service = FaceRecognitionService(model_name='Facenet', distance_metric='cosine')

    def test_vectorized_recognition_cosine(self):
        # Mock inputs
        target_embedding = [1.0, 0.0, 0.0]
        known_users = [
            {'user_id': 'user1', 'embedding': [1.0, 0.0, 0.0], 'name': 'User 1'}, # Dist 0
            {'user_id': 'user2', 'embedding': [0.0, 1.0, 0.0], 'name': 'User 2'}, # Dist 1
            {'user_id': 'user3', 'embedding': [-1.0, 0.0, 0.0], 'name': 'User 3'} # Dist 2
        ]

        # Override _get_default_threshold to verify behavior
        # Default is 0.40 for Facenet Cosine

        results = self.service._vectorized_recognition(target_embedding, known_users)

        # Expect only user1 (dist 0) to match because user2 (dist 1) > 0.40
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['user_id'], 'user1')
        self.assertAlmostEqual(results[0]['distance'], 0.0)
        self.assertTrue(results[0]['verified'])

    def test_vectorized_recognition_euclidean(self):
        self.service.distance_metric = 'euclidean'
        # Mock inputs
        target_embedding = [1.0, 0.0]
        known_users = [
            {'user_id': 'user1', 'embedding': [1.0, 0.0], 'name': 'User 1'}, # Dist 0
            {'user_id': 'user2', 'embedding': [4.0, 4.0], 'name': 'User 2'}, # Dist 5
            {'user_id': 'user3', 'embedding': [1.0, 2.0], 'name': 'User 3'} # Dist 2
        ]

        # Default threshold for Facenet Euclidean is 10
        results = self.service._vectorized_recognition(target_embedding, known_users)

        # user1: 0 < 10
        # user2: 5 < 10
        # user3: 2 < 10
        # All should match
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]['user_id'], 'user1')
        self.assertEqual(results[1]['user_id'], 'user3') # sorted by distance
        self.assertEqual(results[2]['user_id'], 'user2')

if __name__ == '__main__':
    unittest.main()
