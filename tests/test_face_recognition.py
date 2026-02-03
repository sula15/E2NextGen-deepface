import sys
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

# Mock heavy dependencies before they are imported by app code
sys.modules['deepface'] = MagicMock()
sys.modules['cv2'] = MagicMock()

# Mock pandas carefully so isinstance works
mock_pd = MagicMock()
class MockDataFrame:
    def iterrows(self): pass
    @property
    def empty(self): return False
    @property
    def index(self): return []

class MockSeries(dict):
    @property
    def index(self):
        return list(self.keys())

mock_pd.DataFrame = MockDataFrame
mock_pd.Series = MockSeries
sys.modules['pandas'] = mock_pd

sys.modules['tensorflow'] = MagicMock()
sys.modules['flask_sqlalchemy'] = MagicMock()
sys.modules['flask_cors'] = MagicMock()

# Mock flask
mock_flask = MagicMock()
sys.modules['flask'] = mock_flask

# Mock config
sys.modules['config'] = MagicMock()

# Mock app.models.database to avoid SQLA dependencies
mock_db = MagicMock()
sys.modules['app.models.database'] = mock_db

# Now we can import the service
# Note: Importing app.services.face_service might trigger app.__init__ execution if not careful
# But mocking sys.modules should prevent the real modules from loading.
from app.services.face_service import FaceRecognitionService

class TestRecognitionOptimization:
    def setup_method(self):
        self.service = FaceRecognitionService()
        self.service.model_name = 'Facenet'
        self.service.distance_metric = 'cosine'
        self.service.detector_backend = 'opencv'

    @patch('app.services.face_service.DeepFace')
    def test_recognize_face_uses_db_embeddings(self, mock_deepface):
        """Test that recognize_face uses DB embeddings when available"""

        # Mock DeepFace.represent to return a query embedding
        # Let's say query vector is [1.0, 0.0]
        mock_deepface.represent.return_value = [{'embedding': [1.0, 0.0]}]

        # Mock User model
        MockUser = MagicMock()
        mock_db.User = MockUser

        # Create some mock users with embeddings
        # User 1: [1.0, 0.0] -> Distance 0 (Match)
        # User 2: [0.0, 1.0] -> Distance 1 (No Match)

        user1 = MagicMock()
        user1.user_id = "user1"
        user1.embedding = [1.0, 0.0]
        user1.face_image_path = "/path/to/user1.jpg"

        user2 = MagicMock()
        user2.user_id = "user2"
        user2.embedding = [0.0, 1.0]
        user2.face_image_path = "/path/to/user2.jpg"

        # Mock the query
        mock_query = MockUser.query.filter.return_value.with_entities.return_value
        mock_query.all.return_value = [user1, user2]

        results = self.service.recognize_face("dummy_img_path", "dummy_db_path")

        # Check that we got results
        assert len(results) > 0
        best_match = results[0]

        assert best_match['user_id'] == "user1"
        # Distance calculation check
        # With current impl (DeepFace.find), this test is expected to FAIL or behave differently
        # because we haven't implemented the optimization yet.
        # But we are asserting that we implemented it.

        # Verify DeepFace.find was NOT called
        mock_deepface.find.assert_not_called()

    @patch('app.services.face_service.DeepFace')
    @patch('app.services.face_service.Path')
    def test_recognize_face_fallback(self, mock_path, mock_deepface):
        """Test fallback to DeepFace.find when DB is empty"""

        # Mock DeepFace.represent
        mock_deepface.represent.return_value = [{'embedding': [1.0, 0.0]}]

        # Mock User model query to return empty list
        MockUser = MagicMock()
        mock_db.User = MockUser
        MockUser.query.filter.return_value.with_entities.return_value.all.return_value = []

        # Mock Path to exist and have files
        mock_path_obj = MagicMock()
        mock_path.return_value = mock_path_obj
        mock_path_obj.exists.return_value = True
        mock_path_obj.iterdir.return_value = [MagicMock()] # non-empty iterator

        # Mock DeepFace.find to return something (simulating fallback success)
        import pandas as pd
        # Create a mock that passes isinstance(df, pd.DataFrame)
        mock_df = MagicMock(spec=pd.DataFrame)
        mock_df.empty = False

        # Mock iterrows
        mock_df.iterrows.return_value = [
            (0, pd.Series({
                'identity': '/path/to/user1/img.jpg',
                'Facenet_cosine': 0.1,
                'Facenet_threshold': 0.4
            }))
        ]

        # DeepFace.find returns a list of DataFrames
        mock_deepface.find.return_value = [mock_df]

        results = self.service.recognize_face("dummy_img_path", "dummy_db_path")

        # Verify DeepFace.find WAS called
        mock_deepface.find.assert_called_once()
        assert len(results) > 0
