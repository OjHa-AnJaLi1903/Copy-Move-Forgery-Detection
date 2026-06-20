import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load suspect image
image = cv2.imread(r'C:\Users\Anjali Ojha\OneDrive\Desktop\Academics\cmfd 2\cmfd_project\forged\7.tiff')
if image is None:
    print("Image not found!")
    exit()

print(image.shape)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

# Detect SIFT keypoints and descriptors
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray, None)

# Draw keypoints
keypoint_img = cv2.drawKeypoints(image_rgb, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.figure(figsize=(10, 7))
plt.title('SIFT Keypoints')
plt.imshow(keypoint_img)
plt.axis('off')
plt.show()


# Match descriptors with themselves
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
matches = bf.match(descriptors, descriptors)

# Filter out matches where keypoints are too close (to avoid trivial matches)
good_matches = []
for m in matches:
    if m.queryIdx != m.trainIdx:  # not the same point
        pt1 = keypoints[m.queryIdx].pt
        pt2 = keypoints[m.trainIdx].pt
        distance = np.linalg.norm(np.array(pt1) - np.array(pt2))
        if distance > 10:  # Set a threshold
            good_matches.append(m)
      


# Draw good matches
result = cv2.drawMatches(image_rgb, keypoints, image_rgb, keypoints, good_matches, None, flags=2)

plt.figure(figsize=(15, 10))
plt.title('Detected Forged Regions (SIFT Matches)')
plt.imshow(result)
plt.show()

print("Number of potential forgery matches detected: ", len(good_matches))
