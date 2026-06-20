import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Load suspect image
image = cv2.imread(r'D:\cmfd_project\forged\7.tiff')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

# Detect SIFT keypoints and descriptors
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray, None)

# Draw keypoints (optional visualization)
keypoint_img = cv2.drawKeypoints(image_rgb, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.figure(figsize=(10, 7))
plt.title('SIFT Keypoints')
plt.imshow(keypoint_img)
plt.axis('off')
plt.show()

# Match descriptors with themselves
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
matches = bf.match(descriptors, descriptors)

# Filter out trivial or very close matches
good_matches = []
for m in matches:
    if m.queryIdx != m.trainIdx:
        pt1 = keypoints[m.queryIdx].pt
        pt2 = keypoints[m.trainIdx].pt
        distance = np.linalg.norm(np.array(pt1) - np.array(pt2))
        if distance > 10:
            good_matches.append(m)

# Draw good matches (for visualization)
match_img = cv2.drawMatches(image_rgb, keypoints, image_rgb, keypoints, good_matches, None, flags=2)
plt.figure(figsize=(15, 10))
plt.title('Detected Forged Regions (SIFT Matches)')
plt.imshow(match_img)
plt.axis('off')
plt.show()

print("Number of potential forgery matches detected:", len(good_matches))

# --- CLUSTERING and BOUNDING BOX DETECTION ---

# Collect all matched keypoint coordinates
points = []
for m in good_matches:
    pt1 = keypoints[m.queryIdx].pt
    pt2 = keypoints[m.trainIdx].pt
    points.append(pt1)
    points.append(pt2)

points = np.array(points)

# Cluster using DBSCAN
if len(points) > 0:
    clustering = DBSCAN(eps=20, min_samples=5).fit(points)
    labels = clustering.labels_

    # Draw rectangles around each cluster
    output_img = image_rgb.copy()
    unique_labels = set(labels)

    for label in unique_labels:
        if label == -1:
            continue  # -1 means noise
        cluster_points = points[labels == label]
        x, y, w, h = cv2.boundingRect(np.int32(cluster_points))
        cv2.rectangle(output_img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Show final result with bounding boxes
    plt.figure(figsize=(12, 8))
    plt.title('Detected Forged Regions with Bounding Boxes')
    plt.imshow(output_img)
    plt.axis('off')
    plt.show()
else:
    print("No significant matching keypoints found.")
