import cv2
import matplotlib.pyplot as plt

# Load image
img = cv2.imread(r'D:\cmfd_project\forged\test_images\money.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Initialize SURF
surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400)  # Lower threshold → more points

# Find keypoints and descriptors
keypoints, descriptors = surf.detectAndCompute(gray, None)

# Draw keypoints
img_keypoints = cv2.drawKeypoints(img, keypoints, None, (255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Display
plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(img_keypoints, cv2.COLOR_BGR2RGB))
plt.title("SURF Keypoints")
plt.show()
