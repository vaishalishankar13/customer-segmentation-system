# Customer Segmentation using K-Means & PCA

## Project Overview
This project performs customer segmentation using machine learning techniques.

## Problem Statement

Modern retail businesses generate large amounts of customer behavioral data, but many companies struggle to convert this raw data into actionable business insights. Without proper customer segmentation, marketing campaigns become generalized, customer retention decreases, and businesses fail to identify high-value or at-risk customers effectively.

The objective of this project is to build an intelligent customer segmentation system that groups customers based on purchasing behavior, spending patterns, visit frequency, and engagement metrics. The system should help businesses understand different customer types and support data-driven marketing and retention strategies.

## Features
- K-Means Clustering
- PCA Visualization
- Elbow Method
- Silhouette Score
- Interactive Streamlit Dashboard

## Tech Stack
- Python
- Pandas
- Scikit-learn
- Streamlit
- Seaborn
- Matplotlib

## Results
- Identified 3 customer clusters
- PCA retained ~88% variance
- Interactive cluster prediction app

## Business Impact
Enables data-driven customer targeting
Helps identify valuable but disengaged customers
Supports personalized marketing strategies
Improves decision-making using behavioral analytics
Demonstrates practical application of unsupervised machine learning in retail analytics

## How to Run
```bash
pip install -r requirements.txt
python main.py
streamlit run app/app.py
---

# STEP 13 — Push to GitHub

Commands:
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/vaishalishankar13/customer-segmentation-system.git
git push -u origin main