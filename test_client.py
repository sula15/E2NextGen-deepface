#!/usr/bin/env python3
"""
Test client for DeepFace API
Usage: python test_client.py
"""

import requests
import base64
import json
import sys
from pathlib import Path


BASE_URL = 'http://localhost:5001/api/v1'


def encode_image(image_path):
    """Encode image to base64"""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def health_check():
    """Check API health"""
    print("\n=== Health Check ===")
    response = requests.get('http://localhost:5001/health')
    print(json.dumps(response.json(), indent=2))


def enroll_user(user_id, name, image_path, email='', department=''):
    """Enroll a new user"""
    print(f"\n=== Enrolling User: {user_id} ===")
    url = f"{BASE_URL}/enroll"
    
    data = {
        'user_id': user_id,
        'name': name,
        'email': email,
        'department': department,
        'image': encode_image(image_path)
    }
    
    response = requests.post(url, json=data)
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    return response.json()


def recognize_face(image_path, log_attendance=False):
    """Recognize a face"""
    print(f"\n=== Recognizing Face ===")
    url = f"{BASE_URL}/recognize"
    
    data = {
        'image': encode_image(image_path),
        'log_attendance': log_attendance
    }
    
    response = requests.post(url, json=data)
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    return response.json()


def verify_faces(image1_path, image2_path):
    """Verify if two faces match"""
    print(f"\n=== Verifying Faces ===")
    url = f"{BASE_URL}/verify"
    
    data = {
        'image1': encode_image(image1_path),
        'image2': encode_image(image2_path)
    }
    
    response = requests.post(url, json=data)
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    return response.json()


def get_users():
    """Get all users"""
    print(f"\n=== Getting All Users ===")
    url = f"{BASE_URL}/users"
    
    response = requests.get(url)
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    return response.json()


def get_attendance(user_id=None, date=None):
    """Get attendance records"""
    print(f"\n=== Getting Attendance Records ===")
    url = f"{BASE_URL}/attendance"
    
    params = {}
    if user_id:
        params['user_id'] = user_id
    if date:
        params['date'] = date
    
    response = requests.get(url, params=params)
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    return response.json()


def interactive_menu():
    """Interactive test menu"""
    while True:
        print("\n" + "="*50)
        print("DeepFace API Test Client")
        print("="*50)
        print("1. Health Check")
        print("2. Enroll User")
        print("3. Recognize Face")
        print("4. Verify Two Faces")
        print("5. Get All Users")
        print("6. Get Attendance Records")
        print("0. Exit")
        print("="*50)
        
        choice = input("\nEnter your choice: ").strip()
        
        if choice == '0':
            print("Goodbye!")
            break
        elif choice == '1':
            health_check()
        elif choice == '2':
            user_id = input("Enter user ID: ").strip()
            name = input("Enter name: ").strip()
            email = input("Enter email (optional): ").strip()
            dept = input("Enter department (optional): ").strip()
            img_path = input("Enter image path: ").strip()
            if Path(img_path).exists():
                enroll_user(user_id, name, img_path, email, dept)
            else:
                print(f"Error: Image not found at {img_path}")
        elif choice == '3':
            img_path = input("Enter image path: ").strip()
            log = input("Log attendance? (y/n): ").strip().lower() == 'y'
            if Path(img_path).exists():
                recognize_face(img_path, log)
            else:
                print(f"Error: Image not found at {img_path}")
        elif choice == '4':
            img1_path = input("Enter first image path: ").strip()
            img2_path = input("Enter second image path: ").strip()
            if Path(img1_path).exists() and Path(img2_path).exists():
                verify_faces(img1_path, img2_path)
            else:
                print("Error: One or both images not found")
        elif choice == '5':
            get_users()
        elif choice == '6':
            user_id = input("Enter user ID (optional, press Enter to skip): ").strip() or None
            date = input("Enter date YYYY-MM-DD (optional, press Enter to skip): ").strip() or None
            get_attendance(user_id, date)
        else:
            print("Invalid choice. Please try again.")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Command line arguments provided
        print("Use interactive mode: python test_client.py")
    else:
        interactive_menu()
