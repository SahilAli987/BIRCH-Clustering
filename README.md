# 🌿 BIRCH Clustering Analysis App

An interactive **Streamlit** application for exploring **customer segmentation** using the **BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies)** algorithm.  
This app visualizes how different customer types group naturally based on their spending patterns in the **Wholesale Customers Dataset**.

---

## 🚀 Features

- 📊 **Interactive Clustering Visualization** – Tune parameters like threshold, branching factor, and number of clusters in real-time.
- 🧠 **Automatic Insight Generation** – Get meaningful interpretations of each cluster.
- 🧭 **PCA Visualization** – Reduces dimensionality to 2D for easy visualization of clusters.
- 🧾 **Dataset Exploration** – View raw data and statistical summaries directly inside the app.
- ⚙️ **Parameter Customization** – Adjust clustering behavior dynamically without any coding.

---

## 🧩 Algorithms and Concepts Used

### 🔹 BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies)
BIRCH efficiently handles large datasets by incrementally and dynamically clustering data points using a hierarchical data structure called a **CF Tree (Clustering Feature Tree)**.

Each node stores:
- **N**: Number of data points  
- **LS**: Linear sum of data points  
- **SS**: Square sum of data points  

It computes:
- **Centroid (μ)** = LS / N  
- **Radius (R)** = √((SS / N) - μ²)

These help in identifying how far new data points are from existing subclusters, ensuring scalability and efficiency.

### 🔹 PCA (Principal Component Analysis)
Used to project high-dimensional data (6 features) into **2D space** for visualization while preserving variance and cluster separability.

---

## 📁 Dataset Information

**Dataset:** `Wholesale_customers_data.csv`

| Feature | Description |
|----------|--------------|
| Fresh | Annual spending on fresh products (fruits, vegetables, etc.) |
| Milk | Annual spending on milk products |
| Grocery | Annual spending on grocery items |
| Frozen | Annual spending on frozen foods |
| Detergents_Paper | Spending on cleaning and paper products |
| Delicassen | Spending on delicatessen (fine foods) |

**Source:** UCI Machine Learning Repository – Wholesale Customers Data Set

---

## ⚙️ Installation

### 🔧 Prerequisites
Make sure you have **Python 3.8+** and **pip** installed.
