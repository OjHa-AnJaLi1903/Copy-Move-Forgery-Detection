import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image and convert
image = cv2.imread(r'D:\cmfd_project\forged\19.tiff')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

# Detect SIFT keypoints and descriptors
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray, None)

# Match descriptors with themselves using BFMatcher
bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors, descriptors, k=2)

# Lowe's ratio test
good = []
for m, n in matches:
    if m.queryIdx != m.trainIdx and m.distance < 0.75 * n.distance:
        good.append(m)

# Extract matching keypoints
src_pts = np.float32([keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

# Estimate homography using RANSAC
if len(src_pts) >= 4:
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    # Draw inliers only
    ransac_matches = [good[i] for i in range(len(good)) if matchesMask[i]]
    result_img = cv2.drawMatches(image_rgb, keypoints, image_rgb, keypoints, ransac_matches, None, flags=2)

    plt.figure(figsize=(15, 10))
    plt.title('Detected Forged Regions using RANSAC')
    plt.imshow(result_img)
    plt.show()

    print("Number of inlier matches after RANSAC: ", len(ransac_matches))
else:
    print("Not enough matches to compute homography.")