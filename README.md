# DeepFace Face Recognition API

A production-ready face recognition API built with Flask and DeepFace, designed for attendance systems and identity verification.

## Features

- 👤 Face Recognition (1:N matching)
- ✅ Face Verification (1:1 matching)
- 📝 User Enrollment
- 📊 Attendance Tracking
- 🗄️ Database Integration
- 🔒 Secure Image Handling
- 📈 Logging and Monitoring

## Quick Start

### 1. Setup Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Mac/Linux
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Server

```bash
python run.py
```

The API will be available at `http://localhost:5000`

### 4. Test the API

```bash
# Check health
curl http://localhost:5000/health

# Or use the interactive test client
python test_client.py
```

## API Endpoints

### Health Check
```
GET /health
```

### Enroll User
```
POST /api/v1/enroll
Content-Type: application/json

{
  "user_id": "john_doe",
  "name": "John Doe",
  "email": "john@example.com",
  "department": "Engineering",
  "image": "<base64_encoded_image>"
}
```

### Recognize Face
```
POST /api/v1/recognize
Content-Type: application/json

{
  "image": "<base64_encoded_image>",
  "log_attendance": true
}
```

### Verify Faces
```
POST /api/v1/verify
Content-Type: application/json

{
  "image1": "<base64_encoded_image>",
  "image2": "<base64_encoded_image>"
}
```

### Get Users
```
GET /api/v1/users
```

### Get Attendance
```
GET /api/v1/attendance?user_id=john_doe&date=2024-01-01
```

## Configuration

Edit `.env` file to configure:

- `DEEPFACE_MODEL`: Choose from VGG-Face, Facenet, ArcFace, etc.
- `DEEPFACE_DETECTOR`: Choose from opencv, ssd, mtcnn, retinaface
- `DISTANCE_METRIC`: cosine or euclidean
- `DATABASE_URL`: Database connection string

## Project Structure

```
deepface-api/
├── app/
│   ├── __init__.py           # Application factory
│   ├── models/
│   │   └── database.py       # Database models
│   ├── routes/
│   │   └── face_routes.py    # API endpoints
│   ├── services/
│   │   └── face_service.py   # Face recognition logic
│   └── utils/
│       └── helpers.py        # Utility functions
├── uploads/                  # Temporary file uploads
├── face_database/           # Enrolled face images
├── logs/                    # Application logs
├── config.py               # Configuration
├── run.py                  # Application entry point
├── test_client.py          # Test client
├── requirements.txt        # Python dependencies
└── .env                    # Environment variables
```

## Testing

Use the interactive test client:

```bash
python test_client.py
```

Or test with curl:

```bash
# Health check
curl http://localhost:5000/health

# Get all users
curl http://localhost:5000/api/v1/users
```

## License

MIT License
