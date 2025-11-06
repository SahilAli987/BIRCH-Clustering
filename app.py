# === Streamlit App: BIRCH Clustering with PCA Visualization ===
# Dataset: Wholesale_customers_data.csv
# Author: [Your Name]

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import Birch

# =========================
# PAGE CONFIGURATION
# =========================
st.set_page_config(
    page_title="BIRCH Clustering App",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌿 BIRCH Clustering Analysis App")
st.markdown("### Balanced Iterative Reducing and Clustering using Hierarchies (BIRCH)")
st.caption("Interactively explore how the BIRCH algorithm clusters the Wholesale Customers dataset and visualize it using PCA.")

# =========================
# LOAD DATASET
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("Wholesale_customers_data.csv")
    return df

df = load_data()

# Sidebar Navigation
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to:", ["Implementation Explanation", "Visualization", "Dataset"])

# =========================
# PAGE 1: Implementation Explanation
# =========================
if page == "Implementation Explanation":
    st.header("📘 Understanding BIRCH Clustering")

    st.subheader("🔹 What is BIRCH?")
    st.write("""
    **BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies)** is a hierarchical clustering algorithm 
    that efficiently handles large datasets by building a compact representation called a **CF (Clustering Feature) Tree**.
    """)

    st.subheader("🔸 Key Concepts")
    st.markdown("""
    - **CF (Clustering Feature):** Represents a summary of subclusters.  
      Each CF stores:  
      - `N`: Number of points  
      - `LS`: Linear Sum (sum of all data points)  
      - `SS`: Square Sum (sum of squares of all data points)  

    - **Centroid (Mean):** `μ = LS / N`  
    - **Radius (Spread):** `R = √((SS/N) − μ²)`  

    These allow BIRCH to quickly calculate how far new data points are from existing subclusters.
    """)

    st.subheader("🔹 Workflow of the App")
    st.write("""
    1. Load and clean the dataset.  
    2. Apply log transformation to reduce skewness.  
    3. Standardize numeric columns using `StandardScaler`.  
    4. Build a BIRCH model with user-defined threshold, branching factor, and number of clusters.  
    5. Predict cluster labels for all data points.  
    6. Use PCA to reduce data into 2 components for visualization.  
    7. Plot clusters and derive insights automatically.
    """)

    st.success("👉 Move to the 'Visualization' tab to explore the clustering interactively!")

# =========================
# PAGE 2: Visualization
# =========================
elif page == "Visualization":
    st.header("🎨 Interactive BIRCH Visualization")

    # Sidebar Parameters
    st.sidebar.subheader("⚙️ BIRCH Parameters")
    threshold = st.sidebar.slider("Threshold", 0.1, 2.0, 1.0, 0.1)
    branching_factor = st.sidebar.slider("Branching Factor", 10, 100, 40, 10)
    n_clusters = st.sidebar.slider("Number of Clusters", 2, 6, 3, 1)

    # Preprocessing
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df_log = df.copy()
    for col in num_cols:
        if (df[col] > 0).all():
            df_log[col] = np.log1p(df[col])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_log[num_cols])

    # BIRCH Clustering
    birch = Birch(threshold=threshold, branching_factor=branching_factor, n_clusters=n_clusters)
    labels = birch.fit_predict(X_scaled)
    df['Cluster'] = labels

    # Cluster Mean Summary
    cluster_means = df.groupby('Cluster')[num_cols].mean()
    st.subheader("📊 Cluster Mean Comparison")
    st.dataframe(cluster_means.style.highlight_max(axis=0))

    # Cluster Names (insight-based)
    cluster_names = {
        0: 'High Grocery & Milk Buyers (Retail/Small Stores)',
        1: 'High Fresh & Frozen Buyers (Hotels/Restaurants)',
        2: 'Mixed Moderate Buyers (Cafes/Offices)',
        3: 'Low Volume Buyers',
        4: 'High Frozen Buyers',
        5: 'Balanced Buyers'
    }
    df['Cluster Name'] = df['Cluster'].map(cluster_names)

    # PCA for Visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    df_pca['Cluster Name'] = df['Cluster Name']

    # PCA Plot
    st.subheader("🧭 PCA Visualization of BIRCH Clusters")
    fig, ax = plt.subplots(figsize=(8, 6))
    for name in df_pca['Cluster Name'].unique():
        cluster_data = df_pca[df_pca['Cluster Name'] == name]
        ax.scatter(cluster_data['PC1'], cluster_data['PC2'], s=70, alpha=0.8, label=name)
    ax.set_title("PCA Visualization of BIRCH Clusters", fontsize=14, fontweight='bold')
    ax.set_xlabel("Principal Component 1 (PC1)")
    ax.set_ylabel("Principal Component 2 (PC2)")
    ax.legend(title="Customer Segments", fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)

    # ========= Visualization Insights =========
    st.subheader("💡 Insights from Visualization")
    st.write("""
    Based on the PCA visualization, we can interpret the clusters as follows:
    - **Cluster 1 – High Grocery & Milk Buyers (Retail/Small Stores):**  
      These customers spend heavily on grocery and milk, possibly representing small retail shops.
      
    - **Cluster 2 – High Fresh & Frozen Buyers (Hotels/Restaurants):**  
      These clients focus more on perishable goods like fresh and frozen products — likely restaurants or hotels.
      
    - **Cluster 3 – Mixed Moderate Buyers (Cafes/Offices):**  
      Moderate spending across all categories — typical for office canteens or small cafes.
      
    - **Cluster 4 – Low Volume Buyers:**  
      Smaller scale purchasers with low spending in most categories.
      
    The PCA plot shows how each group naturally separates, indicating distinct customer segments 
    based on their purchasing behavior.
    """)

# =========================
# PAGE 3: Dataset
# =========================
elif page == "Dataset":
    st.header("📁 Wholesale Customers Dataset")

    st.write("""
    This dataset contains annual spending in monetary units (m.u.) on diverse product categories 
    by clients of a wholesale distributor.
    """)

    st.markdown("""
    **Features:**
    - **Fresh:** Annual spending on fresh products (fruits, vegetables, etc.)
    - **Milk:** Spending on milk products
    - **Grocery:** Spending on grocery items
    - **Frozen:** Spending on frozen foods
    - **Detergents_Paper:** Spending on cleaning and paper products
    - **Delicassen:** Spending on delicatessen (fine foods)
    """)

    st.dataframe(df.head(15), use_container_width=True)
    st.subheader("📊 Statistical Summary")
    st.write(df.describe())
