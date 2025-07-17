# streamlit_customer_segmentation.py

# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
# import seaborn as sns

# st.set_page_config(page_title="Customer Segmentation App", layout="centered")

# st.title("🧠 Customer Segmentation using KMeans")

# # Upload section
# uploaded_file = st.file_uploader("📂 Upload your customer CSV file", type=["csv"])

# if uploaded_file:
#     df = pd.read_csv(uploaded_file)
#     st.success("✅ File uploaded successfully!")
    
#     # Show dataset preview
#     st.subheader("📊 Data Preview")
#     st.dataframe(df.head())

#     # Show summary
#     st.subheader("📌 Summary Statistics")
#     st.write(df.describe())

#     # Select features
#     st.subheader("🔧 Select Features for Clustering")
#     features = st.multiselect("Choose numeric columns", df.select_dtypes(include='number').columns.tolist(), default=["Annual Income (k$)", "Spending Score (1-100)"])
    
#     if len(features) >= 2:
#         X = df[features]
        
#         # Standardize
#         scaler = StandardScaler()
#         X_scaled = scaler.fit_transform(X)

#         # Number of clusters
#         st.subheader("🔢 Choose Number of Clusters (K)")
#         k = st.slider("Select K", min_value=2, max_value=10, value=3)

#         # Run KMeans
#         model = KMeans(n_clusters=k, random_state=42, n_init=10)
#         labels = model.fit_predict(X_scaled)

#         df['Cluster'] = labels

#         # Visualize Clusters
#         st.subheader("🧩 Cluster Visualization")
#         fig, ax = plt.subplots()
#         sns.scatterplot(x=X[features[0]], y=X[features[1]], hue=labels, palette="Set2", ax=ax)
#         ax.set_xlabel(features[0])
#         ax.set_ylabel(features[1])
#         ax.set_title("Customer Clusters")
#         st.pyplot(fig)

#         # Show cluster-wise data
#         st.subheader("📁 Clustered Data Preview")
#         st.dataframe(df.head())

#         # Download segmented data
#         csv = df.to_csv(index=False).encode('utf-8')
#         st.download_button("⬇️ Download Clustered Data as CSV", data=csv, file_name="clustered_customers.csv", mime="text/csv")
    
#     else:
#         st.warning("Please select at least 2 numeric features.")
# else:
#     st.info("👆 Upload a CSV file to get started.")


# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
# from plotly.express import scatter_3d

# st.set_page_config(page_title="Customer Segmentation", layout="centered")
# st.title("🧠 Customer Segmentation using KMeans")

# # Upload section
# uploaded_file = st.file_uploader("📂 Upload your customer CSV file", type=["csv"])

# if uploaded_file:
#     df = pd.read_csv(uploaded_file)
#     st.success("✅ File uploaded successfully!")

#     # Show dataset preview
#     st.subheader("📊 Data Preview")
#     st.dataframe(df.head())

#     # Show summary
#     st.subheader("📌 Summary Statistics")
#     st.write(df.describe())

#     # Feature selection
#     st.subheader("🔧 Select Features for Clustering")
#     numeric_cols = df.select_dtypes(include='number').columns.tolist()
#     features = st.multiselect("Choose numeric columns", numeric_cols, default=["Annual Income (k$)", "Spending Score (1-100)"])

#     if len(features) >= 2:
#         X = df[features]

#         # Standardization
#         scaler = StandardScaler()
#         X_scaled = scaler.fit_transform(X)

#         # Choose number of clusters
#         st.subheader("🔢 Choose Number of Clusters (K)")
#         k = st.slider("Select K", min_value=2, max_value=10, value=3)

#         # Elbow Method
#         with st.expander("📈 Show Elbow Method"):
#             inertia = []
#             for i in range(1, 11):
#                 km = KMeans(n_clusters=i, random_state=42, n_init=10)
#                 km.fit(X_scaled)
#                 inertia.append(km.inertia_)
#             fig, ax = plt.subplots()
#             ax.plot(range(1, 11), inertia, marker='o')
#             ax.set_title("Elbow Method")
#             ax.set_xlabel("Number of clusters")
#             ax.set_ylabel("WCSS")
#             st.pyplot(fig)

#         # Run KMeans
#         model = KMeans(n_clusters=k, random_state=42, n_init=10)
#         labels = model.fit_predict(X_scaled)
#         df['Cluster'] = labels

#         # Cluster summary
#         st.subheader("🧾 Cluster Summary Statistics")
#         st.dataframe(df.groupby('Cluster')[features].mean().round(2))

#         # Pie chart of cluster size
#         st.subheader("📊 Cluster Size Distribution")
#         cluster_counts = df['Cluster'].value_counts()
#         fig1, ax1 = plt.subplots()
#         ax1.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%', startangle=90)
#         ax1.axis('equal')
#         st.pyplot(fig1)

#         # 2D Cluster Visualization
#         st.subheader("🧩 Cluster Visualization (2D)")
#         fig2, ax2 = plt.subplots()
#         sns.scatterplot(x=X[features[0]], y=X[features[1]], hue=labels, palette="Set2", ax=ax2)
#         ax2.set_xlabel(features[0])
#         ax2.set_ylabel(features[1])
#         st.pyplot(fig2)

#         # Optional 3D Visualization
#         if len(features) >= 3:
#             st.subheader("🌐 3D Cluster Visualization")
#             fig3 = scatter_3d(df, x=features[0], y=features[1], z=features[2], color='Cluster', symbol='Cluster')
#             st.plotly_chart(fig3)

#         # Show data with clusters
#         st.subheader("📁 Clustered Data Preview")
#         st.dataframe(df.head())

#         # Download option
#         csv = df.to_csv(index=False).encode('utf-8')
#         st.download_button("⬇️ Download Clustered Data as CSV", data=csv, file_name="clustered_customers.csv", mime="text/csv")

#     else:
#         st.warning("Please select at least 2 numeric features for clustering.")

# else:
#     st.info("👆 Upload a CSV file to get started.")


# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from plotly.express import scatter_3d

st.set_page_config(page_title="Customer Segmentation", layout="centered")
st.title("🧠 Customer Segmentation using KMeans")

# Upload section
uploaded_file = st.file_uploader("📂 Upload your customer CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    st.subheader("📌 Summary Statistics")
    st.write(df.describe())

    st.subheader("🧮 Feature Correlation Heatmap")
    fig_corr, ax_corr = plt.subplots()
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax_corr)
    st.pyplot(fig_corr)

    st.subheader("🔧 Select Features for Clustering")
    features = st.multiselect("Choose numeric columns", numeric_cols, default=["Annual Income (k$)", "Spending Score (1-100)"])

    if len(features) >= 2:
        X = df[features]

        # Standardization
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Number of clusters
        st.subheader("🔢 Choose Number of Clusters (K)")
        k = st.slider("Select K", min_value=2, max_value=10, value=3)

        # Elbow Method
        with st.expander("📈 Show Elbow Method"):
            inertia = []
            for i in range(1, 11):
                km = KMeans(n_clusters=i, random_state=42, n_init=10)
                km.fit(X_scaled)
                inertia.append(km.inertia_)
            fig_elbow, ax_elbow = plt.subplots()
            ax_elbow.plot(range(1, 11), inertia, marker='o')
            ax_elbow.set_title("Elbow Method")
            ax_elbow.set_xlabel("Number of clusters")
            ax_elbow.set_ylabel("WCSS")
            st.pyplot(fig_elbow)

        # KMeans clustering
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        df['Cluster'] = labels

        # Rename clusters
        st.subheader("✍️ Rename Cluster Labels")
        cluster_names = {}
        for i in range(k):
            new_name = st.text_input(f"Name for Cluster {i}", value=f"Cluster {i}")
            cluster_names[i] = new_name
        df['Cluster Name'] = df['Cluster'].map(cluster_names)

        # Cluster summary
        st.subheader("🧾 Cluster Summary Statistics")
        st.dataframe(df.groupby('Cluster Name')[features].mean().round(2))

        st.subheader("🧠 Cluster Profiles")
        for i in range(k):
            st.markdown(f"**{cluster_names[i]}**")
            profile = df[df['Cluster'] == i][features].mean().round(2).to_dict()
            for key, val in profile.items():
                st.markdown(f"- {key}: {val}")

        # Cluster size pie chart
        st.subheader("📊 Cluster Size Distribution")
        cluster_counts = df['Cluster Name'].value_counts()
        fig1, ax1 = plt.subplots()
        ax1.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')
        st.pyplot(fig1)

        # Custom color palette
        st.subheader("🎨 Choose a Color Palette for Visualizations")
        palette_option = st.selectbox("Select palette", ["Set1", "Set2", "Set3", "coolwarm", "viridis"])

        # 2D visualization
        st.subheader("🧩 Cluster Visualization (2D)")
        fig2, ax2 = plt.subplots()
        sns.scatterplot(x=X[features[0]], y=X[features[1]], hue=df['Cluster Name'], palette=palette_option, ax=ax2)
        ax2.set_xlabel(features[0])
        ax2.set_ylabel(features[1])
        st.pyplot(fig2)

        # 3D visualization
        if len(features) >= 3:
            st.subheader("🌐 3D Cluster Visualization")
            fig3 = scatter_3d(df, x=features[0], y=features[1], z=features[2], color='Cluster Name', symbol='Cluster Name')
            st.plotly_chart(fig3)

        # PCA visualization
        if len(features) > 3:
            st.subheader("📉 PCA - 2D Visualization")
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            fig_pca, ax_pca = plt.subplots()
            sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['Cluster Name'], palette=palette_option, ax=ax_pca)
            ax_pca.set_title("PCA - 2D Cluster View")
            st.pyplot(fig_pca)

        # Cluster trends over time
        date_cols = df.select_dtypes(include='datetime').columns.tolist()
        if not date_cols:
            potential_dates = [col for col in df.columns if "date" in col.lower()]
            for col in potential_dates:
                try:
                    df[col] = pd.to_datetime(df[col])
                    date_cols.append(col)
                except:
                    continue

        if date_cols:
            date_col = st.selectbox("📆 Select Date Column for Trend Analysis", date_cols)
            df['Month'] = df[date_col].dt.to_period("M")
            trend_data = df.groupby(['Month', 'Cluster Name']).size().unstack(fill_value=0)
            st.subheader("📈 Cluster Trends Over Time")
            st.line_chart(trend_data)

        # Preview with clusters
        st.subheader("📁 Clustered Data Preview")
        st.dataframe(df.head())

        # Download option
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Clustered Data as CSV", data=csv, file_name="clustered_customers.csv", mime="text/csv")

    else:
        st.warning("⚠️ Please select at least 2 numeric features for clustering.")

else:
    st.info("👆 Upload a CSV file to get started.")
