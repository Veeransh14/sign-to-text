import cv2
import sys

print("=" * 70)
print("CAMERA DIAGNOSTIC TOOL FOR SIGN LANGUAGE APP (Windows/Cross-Platform)")
print("=" * 70)

# Test OpenCV camera access
print("\nTesting OpenCV camera access...")
available_cameras = []

# Check first 10 indices
for i in range(10):
    print(f"   Testing camera index {i}...", end=" ")
    try:
        # cv2.CAP_DSHOW is often faster/more reliable on Windows
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) if sys.platform == 'win32' else cv2.VideoCapture(i)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"[OK] WORKS! (Resolution: {w}x{h})")
                available_cameras.append(i)
            else:
                print("[X] Opens but can't read frame")
            cap.release()
        else:
            print("[X] Can't open")
    except Exception as e:
        print(f"[X] Error: {e}")

print("\n" + "=" * 70)
print("DIAGNOSTIC SUMMARY")
print("=" * 70)

if available_cameras:
    print(f"[OK] SUCCESS! Found {len(available_cameras)} working camera(s)")
    print(f"  Camera indices: {available_cameras}")
    print(f"\n  Recommended: Use camera index {available_cameras[0]}")
else:
    print("[X] NO WORKING CAMERAS FOUND!")
    print("\nTROUBLESHOOTING STEPS:")
    print("1. Check if camera is connected")
    print("2. Check for privacy shutters")
    print("3. Close other apps causing conflicts (Zoom, Teams, etc.)")
    print("4. Try a different USB port if external")
    print("5. Check Windows Camera Privacy settings")

print("=" * 70)
