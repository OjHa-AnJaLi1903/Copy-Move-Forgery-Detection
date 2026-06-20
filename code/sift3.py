import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread(r'C:\Users\Anjali Ojha\OneDrive\Desktop\Academics\cmfd 2\cmfd_project\forged\8.tiff')
if image is None:
    print("Image not found!")
    exit()

print(image.shape)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

# Step 1: Detect keypoints and compute descriptors using SIFT
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray, None)

# Step 2: FLANN parameters and matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

# Step 3: Match descriptors with k-NN (k=2 for Lowe's ratio test)
matches = flann.match(descriptors, descriptors)

# Step 4: Apply Lowe’s ratio test and filter trivial matches
good_matches = []
for m in matches:
    if m.queryIdx != m.trainIdx:
        pt1 = keypoints[m.queryIdx].pt
        pt2 = keypoints[m.trainIdx].pt
        spatial_dist = np.linalg.norm(np.array(pt1) - np.array(pt2))
        if spatial_dist > 20:
            good_matches.append(m)

# Step 5: Draw good matches (forged regions may appear as repeated patterns)
result = cv2.drawMatches(image_rgb, keypoints, image_rgb, keypoints, good_matches, None, flags=2)

# Show result
plt.figure(figsize=(15, 10))
plt.title('Detected Forged Regions (SIFT + FLANN Only)')
plt.imshow(result)
plt.axis('off')
plt.show()

# Print summary
print("Total keypoints detected:", len(keypoints))
print("Good matches found (potential copy-move pairs):", len(good_matches))
