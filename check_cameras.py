#!/usr/bin/env python3
import cv2
import subprocess
import os

print("=" * 70)
print("CAMERA DIAGNOSTIC TOOL FOR SIGN LANGUAGE APP")
print("=" * 70)

# Check video devices
print("\n1. Checking for video devices...")
try:
    result = subprocess.run(['ls', '-la', '/dev/video*'], 
                           capture_output=True, text=True, shell=True)
    if result.stdout:
        print("   Found video devices:")
        print(result.stdout)
    else:
        print("   ❌ No /dev/video* devices found!")
        print("   Your camera may not be connected or recognized.")
except Exception as e:
    print(f"   Error checking devices: {e}")

# Check permissions
print("\n2. Checking video group permissions...")
try:
    result = subprocess.run(['groups'], capture_output=True, text=True)
    groups = result.stdout
    if 'video' in groups:
        print("   ✓ User is in 'video' group")
    else:
        print("   ❌ User is NOT in 'video' group!")
        print("   Run: sudo usermod -a -G video $USER")
        print("   Then logout and login again")
except Exception as e:
    print(f"   Error checking groups: {e}")

# List cameras with v4l2
print("\n3. Checking camera details with v4l2...")
try:
    result = subprocess.run(['v4l2-ctl', '--list-devices'], 
                           capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout)
    else:
        print("   ⚠ v4l2-ctl not installed")
        print("   Install with: sudo apt-get install v4l-utils")
except FileNotFoundError:
    print("   ⚠ v4l2-ctl not found")
    print("   Install with: sudo apt-get install v4l-utils")

# Test OpenCV camera access
print("\n4. Testing OpenCV camera access...")
available_cameras = []

for i in range(10):
    print(f"   Testing camera index {i}...", end=" ")
    cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
    
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            backend = cap.getBackendName()
            print(f"✓ WORKS! ({w}x{h}, {backend})")
            available_cameras.append(i)
        else:
            print("✗ Opens but can't read")
        cap.release()
    else:
        print("✗ Can't open")

print("\n" + "=" * 70)
print("DIAGNOSTIC SUMMARY")
print("=" * 70)

if available_cameras:
    print(f"✓ SUCCESS! Found {len(available_cameras)} working camera(s)")
    print(f"  Camera indices: {available_cameras}")
    print(f"\n  Recommended: Use camera index {available_cameras[0]}")
    print(f"\n  Update final_pred.py line 48 to:")
    print(f"  app = Application(camera_index={available_cameras[0]})")
else:
    print("❌ NO WORKING CAMERAS FOUND!")
    print("\nTROUBLESHOOTING STEPS:")
    print("1. Check if camera LED is on (if applicable)")
    print("2. Run: sudo chmod 666 /dev/video*")
    print("3. Run: sudo usermod -a -G video $USER (then logout/login)")
    print("4. Close other apps using camera (Zoom, Skype, Cheese, etc.)")
    print("5. Check: lsof /dev/video*")
    print("6. Reboot your system")
    print("7. If using external webcam, try different USB port")
    print("8. Test with: cheese (sudo apt install cheese)")

print("=" * 70)