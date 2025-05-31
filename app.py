import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt


st.title("🧑‍💼 Сегментування клієнтів")
st.write("Інтерактивне дослідження кластерів клієнтів за допомогою K-Means")

# Бокова панель з описом
with st.sidebar:
    st.markdown("### ℹ️ Про проєкт:")
    st.info("""
    Цей застосунок візуалізує результати кластеризації клієнтів на основі їх доходу та витрат.

    Використано алгоритм **K-Means** для виділення груп споживачів.

    **Навігація:**
    - Перегляньте графік кластерів за доходом та витратами.
    - Наведіть на точку для опису сегмента (характеристика клієнта).
    """)

# Завантаження даних
df = pd.read_csv("Mall_Customers_With_Clustering.csv")
cluster_profiles = pd.read_csv("cluster_profiles.csv")

# Додавання опису до основної таблиці
df = df.merge(cluster_profiles[["KMeans_Cluster", "Description"]], on="KMeans_Cluster", how="left")

# Візуалізація кластерів
st.subheader("📊 Кластери клієнтів (K-Means)")

fig = px.scatter(
    df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    color=df["KMeans_Cluster"].astype(str),
    hover_data=["Description"],
    title="Кластери клієнтів (K-Means)"
)
st.plotly_chart(fig, use_container_width=True)

# Попередній перегляд таблиці
st.subheader("📋 Попередній перегляд даних")
st.dataframe(df[["Annual Income (k$)", "Spending Score (1-100)", "KMeans_Cluster", "Description"]])

# Профілі кластерів
st.subheader("📑 Профілі кластерів (середні значення ознак)")
st.dataframe(cluster_profiles)

# Візуалізація середніх показників по кластерам
st.subheader("📈 Візуалізація середніх показників по кластерам")

sns.set_style("whitegrid")
palette = sns.color_palette("tab10")

fig_bar, ax = plt.subplots(1, 2, figsize=(12, 5))

sns.barplot(data=cluster_profiles, x="KMeans_Cluster", y="Annual Income (k$)",
            hue="KMeans_Cluster", palette=palette, legend=False, ax=ax[0])
ax[0].set_title("Середній дохід по кластерах")
ax[0].set_xlabel("Кластер")
ax[0].set_ylabel("Annual Income (k$)")

sns.barplot(data=cluster_profiles, x="KMeans_Cluster", y="Spending Score (1-100)",
            hue="KMeans_Cluster", palette=palette, legend=False, ax=ax[1])
ax[1].set_title("Середні витрати по кластерах")
ax[1].set_xlabel("Кластер")
ax[1].set_ylabel("Spending Score (1-100)")

st.pyplot(fig_bar)
