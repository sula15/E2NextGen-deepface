import sys
from unittest.mock import MagicMock, patch
import os
import numpy as np

# Mock heavy dependencies before importing the service
mock_deepface_module = MagicMock()
mock_deepface_class = MagicMock()
mock_deepface_module.DeepFace = mock_deepface_class
sys.modules['deepface'] = mock_deepface_module

sys.modules['cv2'] = MagicMock()
sys.modules['pandas'] = MagicMock()
sys.modules['tensorflow'] = MagicMock()

# Now import the service
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.services.face_service import FaceRecognitionService

def test_recognize_face_uses_deepface_find():
    print("Testing recognize_face baseline behavior (without optimization)...")

    # Initialize service
    service = FaceRecognitionService()

    # Mock Path to simulate non-empty database
    with patch('app.services.face_service.Path') as MockPath:
        mock_path_instance = MockPath.return_value
        mock_path_instance.exists.return_value = True
        mock_path_instance.iterdir.return_value = [MagicMock()]

        # Mock DeepFace.find result
        mock_df = MagicMock()
        mock_df.__iter__.return_value = []
        mock_df.empty = True

        # DeepFace.find returns list of dataframes
        mock_deepface_class.find.return_value = [mock_df]

        # Call recognize_face WITHOUT known_embeddings
        try:
            service.recognize_face("dummy_img.jpg", "dummy_db_path")
        except Exception:
            pass

        # Assert DeepFace.find was called
        if mock_deepface_class.find.called:
            print("✓ recognize_face calls DeepFace.find when no embeddings provided")
        else:
            print("✗ recognize_face DID NOT call DeepFace.find when no embeddings provided")

def test_compute_face_matches_logic():
    print("\nTesting compute_face_matches logic...")

    service = FaceRecognitionService(distance_metric='cosine')

    # Create dummy embeddings
    target_embedding = [1.0, 0.0]

    # Known embeddings
    known_embeddings = [
        {'user_id': 'user1', 'embedding': [1.0, 0.0]},       # Distance 0
        {'user_id': 'user2', 'embedding': [0.0, 1.0]},       # Distance 1 (Orthogonal)
        {'user_id': 'user3', 'embedding': [-1.0, 0.0]}       # Distance 2 (Opposite)
    ]

    # Test with default threshold (0.40)
    matches = service.compute_face_matches(target_embedding, known_embeddings)

    # Verification
    if len(matches) == 1:
        print(f"✓ Returned {len(matches)} match (filtered by threshold)")
    else:
        print(f"✗ Returned {len(matches)} matches, expected 1")

    if matches and matches[0]['user_id'] == 'user1':
        print("✓ Best match identified correctly")

    # Test with custom threshold (1.5) to include more matches
    matches_loose = service.compute_face_matches(target_embedding, known_embeddings, threshold=1.5)

    if len(matches_loose) == 2:
        print(f"✓ Returned {len(matches_loose)} matches with loose threshold")
        # Should include user1 (0.0) and user2 (1.0), but not user3 (2.0)
    else:
        # Distance for user3 is 2.0 (1 - (-1)) = 2.0. So 1.5 should exclude it.
        # Wait, cosine similarity of [-1, 0] and [1, 0] is -1. Distance = 1 - (-1) = 2.
        # So user3 is distance 2.
        # User2 is distance 1.
        # User1 is distance 0.
        # Threshold 1.5 should include 0 and 1.
        print(f"✗ Returned {len(matches_loose)} matches with loose threshold, expected 2")


def test_recognize_face_uses_optimization():
    print("\nTesting recognize_face WITH optimization...")
    service = FaceRecognitionService()

    # Mock extract_embedding to return a valid embedding
    service.extract_embedding = MagicMock(return_value=[1.0, 0.0])

    # Reset DeepFace.find mock
    mock_deepface_class.find.reset_mock()

    known_embeddings = [{'user_id': 'test', 'embedding': [1.0, 0.0]}]

    # Call recognize_face WITH known_embeddings
    results = service.recognize_face("img.jpg", "db_path", known_embeddings=known_embeddings)

    # Verify DeepFace.find was NOT called
    if not mock_deepface_class.find.called:
        print("✓ DeepFace.find was skipped (Optimization Active)")
    else:
        print("✗ DeepFace.find WAS called (Optimization Failed)")

    # Verify results
    if len(results) == 1 and results[0]['user_id'] == 'test':
        print("✓ Results returned correctly")

if __name__ == "__main__":
    test_recognize_face_uses_deepface_find()
    test_compute_face_matches_logic()
    test_recognize_face_uses_optimization()
