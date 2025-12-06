# Spotify Music Popularity Analysis 🎧

This repository contains the final project for the **Spotify Music Data Analysis** assignment.  
The project explores how audio features relate to a song’s popularity and builds machine learning models to predict whether a track is **“High”** or **“Low”** popularity.

The entire workflow (EDA → feature engineering → clustering → classification → evaluation) is implemented in a single Jupyter notebook:

> `Final_Project.ipynb`

---

## 🎯 Project Goal

The main goal of this project is to:

> **Understand which audio features drive track popularity and build models that can predict whether a song will be “High” or “Low” popularity.**

More specifically:

- Explore the distribution and relationships of Spotify audio features.
- Identify which features are most correlated with popularity.
- Group songs into distinct **clusters** based on their sound profile.
- Train and compare **classification models** to predict popularity labels.

---

## 📂 Dataset

The analysis is based on two CSV files derived from a Spotify tracks dataset:

- `high_popularity_spotify_data.csv` – tracks labeled as **High** popularity  
- `low_popularity_spotify_data.csv` – tracks labeled as **Low** popularity  

In the notebook, these are combined into a single DataFrame with a target column:

- `popularity` ∈ {`"High"`, `"Low"`}

Several identifier / URL columns that are not useful for modeling are dropped, such as:

- `track_album_id`, `track_id`, `id`, `playlist_id`, `track_href`, `uri`, `analysis_url`

---

## 🧪 Methods & Workflow

All steps are implemented and documented inside `Final_Project.ipynb`.

### 1. Exploratory Data Analysis (EDA)

- Checked for:
  - Missing values (NaNs)
  - Duplicate rows
  - Suspicious “non-NaN” placeholders (e.g. `"?"`, `"N/A"`, etc.)
- Removed rows with missing values.
- Computed **summary statistics** for all numerical features.
- Calculated **skewness** and **kurtosis** to understand distribution shapes.
- Plotted **histograms and KDEs** for numerical features to:
  - See how features are distributed.
  - Detect outliers and heavy tails.

### 2. Feature Engineering & Transformations

To improve data quality and make features more suitable for modeling:

- Applied **log transforms** (`log1p`) to reduce right skew:
  - `acousticness → acousticness_log`
  - `liveness → liveness_log`
  - `speechiness → speechiness_log`
- Applied **Yeo–Johnson transformation** to handle skewed variables that include negative values:
  - `loudness → loudness_yj`
  - `instrumentalness → instrumentalness_yj`
- Engineered a better duration feature:
  - Transformed `duration_ms` with `log1p`, then standardized it → `duration_scaled`
  - Dropped the original `duration_ms` to avoid redundancy.
- Standardized **energy**:
  - `energy → energy_scaled`
- Dropped the original untransformed columns once transformed versions were created.

### 3. Unsupervised Learning – K-Means Clustering

To explore natural groupings of songs:

- Selected features:
  - `energy_scaled`
  - `loudness_yj`
- Used **StandardScaler** where needed and created `X_scaled`.
- Used the **Elbow Method** to choose the number of clusters.
- Applied **K-Means** with `k = 3` and assigned cluster labels to each song.
- Visualized clusters in a 2D scatter plot (`energy_scaled` vs `loudness_yj`) with centroids.
- Interpreted clusters approximately as:
  - **Cluster 0** – low energy, low loudness (calmer, quieter songs)
  - **Cluster 1** – high energy, high loudness (energetic, loud tracks, e.g. party/workout music)
  - **Cluster 2** – moderate energy and loudness (mid-tempo songs)

### 4. Supervised Learning – Popularity Classification

The target variable is the **popularity label**:

- `popularity = "High"` or `"Low"`

For modeling, the most relevant (and transformed) features were selected:

- `energy_scaled`
- `danceability`
- `loudness_yj`
- `instrumentalness_yj`
- `acousticness_log`

The data is split into **train/test** sets using stratified sampling:

- `test_size = 0.2`
- `random_state = 42`
- `stratify = y` (to keep class balance)

Two classification models are trained and evaluated:

1. **Logistic Regression**
   - Baseline linear model.
   - Trained with `max_iter=1000`.
   - Evaluated with:
     - Accuracy
     - Precision (for `"High"` class)
     - Recall (for `"High"` class)
     - F1-score
     - Confusion matrix
     - Classification report

2. **Random Forest Classifier**
   - Non-linear ensemble model.
   - Configuration (from the notebook):
     - `n_estimators = 300`
     - `random_state = 42`
     - `min_samples_split = 2`
   - Evaluated with the same metrics as Logistic Regression.
   - Feature importances extracted to understand which inputs drive predictions.

**Model comparison (summary):**

- The **Random Forest** model achieves better performance than Logistic Regression on the test set (higher F1-score and better classification report for the `"High"` class), making it the preferred model for this task.
