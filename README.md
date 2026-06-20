# 🖼️ Copy-Move Forgery Detection

A **Python-based image forgery detection system** that identifies **copy-move forgeries** using **SIFT (Scale-Invariant Feature Transform)** feature extraction and feature matching techniques. The project detects duplicated regions within an image by matching similar keypoints and estimating geometric transformations using **RANSAC** and **DBSCAN**.

---

## 🚀 Features

* 🔍 Detects copy-move image forgery using SIFT feature extraction
* 📌 Robust keypoint detection and descriptor matching
* 🔄 Outlier removal using RANSAC
* 📊 Clustering of matched keypoints using DBSCAN
* 🖼️ Supports multiple forged and original test images
* ⚡ Fast and efficient implementation using OpenCV
* 📈 Visualizes detected forged regions and feature matches

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge\&logo=python\&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge\&logo=opencv\&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge\&logo=numpy\&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge)
![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge\&logo=scikitlearn\&logoColor=white)

---

## 📦 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/OjHa-AnJaLi1903/Copy-Move-Forgery-Detection.git
cd Copy-Move-Forgery-Detection
```

### 2️⃣ Create a Virtual Environment (Optional)

```bash
python -m venv myenv
```

Activate the virtual environment:

**Windows**

```bash
myenv\Scripts\activate
```

**Linux / macOS**

```bash
source myenv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is unavailable:

```bash
pip install opencv-python numpy matplotlib scikit-learn
```

### 4️⃣ Run the Project

Navigate to the code directory:

```bash
cd code
```

Run any implementation file, for example:

```bash
python sift.py
```

or

```bash
python shift_ransac.py
```

---

## 📂 Dataset

The repository contains:

* **orignal/** – Original (authentic) images
* **forged/** – Copy-move forged images
* **forged/test_images/** – Additional test images for evaluation

---

## 📁 Folder Structure

```text
Copy-Move-Forgery-Detection/
├── code/
│   ├── sift.py
│   ├── sift2.py
│   ├── sift3.py
│   ├── shift_ransac.py
│   ├── shift_dbsacn.py
│   ├── test_plot.py
│   └── 1.py
│
├── forged/
│   ├── test_images/
│   └── *.tiff
│
├── orignal/
│   ├── *.jpg
│   ├── *.png
│   ├── *.webp
│   └── *.jpeg
│
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🧠 Detection Pipeline

1. Read input image
2. Detect SIFT keypoints and descriptors
3. Match descriptors between image regions
4. Filter false matches using RANSAC
5. Cluster matched points using DBSCAN
6. Highlight duplicated (forged) regions
7. Display the detection result

---

## 📸 Sample Results

You can add screenshots of:

* Original Image
* Forged Image
* SIFT Keypoints
* Matched Features
* Final Forgery Detection Output

Example:

```text
results/
├── original.png
├── forged.png
├── keypoints.png
└── detection.png
```

---

## 📚 Future Improvements

* Deep Learning-based copy-move forgery detection
* GUI using Streamlit or Tkinter
* Batch image processing
* Performance benchmarking
* Support for additional image formats
* Automatic evaluation metrics (Precision, Recall, F1-Score)

---

## ✍️ Author

**Anjali Ojha**

*B.Tech Computer Science & Engineering*
*National Institute of Technology Meghalaya*

GitHub: https://github.com/OjHa-AnJaLi1903
