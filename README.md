# 🍽️ Zomato Bangalore Restaurant EDA Dashboard

An interactive Exploratory Data Analysis dashboard built with **Streamlit** and **Plotly** for analyzing Zomato Bangalore restaurant data as part of a Data Analytics project.

---

## 📌 Project Overview

This project analyzes the Zomato Bangalore restaurant dataset to help a prospective **cloud kitchen startup** make data-driven decisions around:
- Cuisine selection
- Pricing strategy
- Location targeting
- Online delivery adoption

---

## 📊 Dashboard Sections

| Section | Description |
|---|---|
| **Overview** | KPI summary cards, rating distribution, restaurant type breakdown |
| **Ratings Analysis** | Rating vs cost, online order impact on ratings, top rated restaurants |
| **Location Insights** | Restaurant density, average rating and cost by area |
| **Cuisine Deep Dive** | Most popular cuisines, highest rated cuisines, treemap |
| **Cost & Value** | Spend distribution, cost by restaurant type, votes vs cost |
| **Delivery & Booking** | Online order adoption, table booking trends, violin plots |

---

## 🚀 How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/AaryaWadje/zomato-dashboard.git
cd zomato-dashboard
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run zomato_eda_dashboard.py
```

**4.** Open your browser at `http://localhost:8501` and upload your `zomato_cleaned.csv`

---

## 📁 Files in this Repo

```
├── zomato_eda_dashboard.py   # Main Streamlit dashboard
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 🗂️ Dataset

- **Source:** [Zomato Bangalore Restaurants — Kaggle](https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants)
- **Size:** ~51,000 rows, 17 columns
- **Domain:** Food & Restaurant Analytics

---

## 🛠️ Tech Stack

- **Python** — Core language
- **Streamlit** — Dashboard framework
- **Plotly** — Interactive charts
- **Pandas / NumPy** — Data manipulation

---

## 📚 Assignment Context

This dashboard was built as part of a **Data Analytics (MGB) Project Based Learning** assignment. The objective is to demonstrate end-to-end data analytics using a real-world dataset with a business use case.

---

## ⚠️ Note

This project is for academic purposes. Please do not copy or redistribute without permission.
