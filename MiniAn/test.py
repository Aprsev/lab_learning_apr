import cv2;
try:
    import minian
    print("minian module is imported successfully.")
except ModuleNotFoundError:
    print("minian module is not found.")