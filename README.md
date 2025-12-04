**Project Structure**
>Data/
>>Country_Data.csv

>Others/
>>Clustering Countries for Strategic Aid Allocation_v2024.1

>>Clustering_Countries_for_Strategic_Aid_Allocation.ipynb

>templates/
>>input_form.html

>>result.html

>app.py

>Clustering Countries for Strategic Aid Allocation.pkl

>requirements.txt

>similar_countries.json(currently for Algeria)


**Notebook: EDA, hypothesis testing, and modeling**
File: Clustering_Countries_for_Strategic_Aid_Allocation.ipynb
> Introduction
>> Objective: Segment countries into development/need tiers to prioritize aid.
- Data: Socioeconomic and health indicators (income, gdpp, life_expec, child_mort, total_fer, health, inflation, imports, exports, import_export_ratio, regions one-hot).
- EDA
- Distributions & outliers: Histograms, boxplots; winsorization/log-transform for skewed gdpp.
- Feature engineering: import_export_ratio; gdpp_log.
- Correlations: Pearson heatmap; highlight meaningful pairs.
- Hypothesis testing
- Inflation vs GDP per capita: Pearson r\approx -0.33, p\approx 1.1\times 10^{-5} → significant inverse association.
- Income vs child mortality: Negative correlation; higher income linked to lower child mortality.
- Fertility vs income: Negative correlation; higher fertility associated with lower income.
- Modeling (unsupervised)
- K-Means: Elbow → k=3; silhouette ≈ 0.1898.
- Hierarchical (Ward): Dendrogram → 4 clusters; silhouette ≈ 0.1846.
- DBSCAN: Baseline failed (<2 clusters), tune via k-distance; document rationale.
- Validation & interpretation
- PCA (2D): Visual separation of clusters; interpret principal components.
- Cluster centroids: Profile development tiers (high/mid/low).
- India’s peer group: Countries similar to India; implications for policy.
- Insights & recommendations
- Policy: Health efficiency, fertility reduction, inflation control, trade diversification.
- Strategic: India’s transitional leverage; regional cooperation.
Tip: Export final tables (cluster assignments, centroids) as CSV for stakeholders.

Deployment code and files
requirements.txt
Flask==2.2.5
scikit-learn==1.3.0
pandas==2.0.3
numpy==1.25.0
joblib==1.3.2
gunicorn==21.2.0


run.py
from app.routes import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


app/init.py
from flask import Flask
from .model_loader import load_model, load_scaler

def create_app():
    app = Flask(__name__)
    app.config["MODEL"] = load_model()
    app.config["SCALER"] = load_scaler()

    from .routes import bp
    app.register_blueprint(bp)
    return app


app/model_loader.py
import joblib
from pathlib import Path

MODEL_PATH = Path("models/aid_kmeans.pkl")
SCALER_PATH = Path("models/scaler.pkl")

def load_model():
    return joblib.load(MODEL_PATH)

def load_scaler():
    return joblib.load(SCALER_PATH)


app/preprocess.py
import numpy as np
import pandas as pd

FEATURE_ORDER = [
    "child_mort", "total_fer", "life_expec", "income", "health", "inflation",
    "imports", "exports", "import_export_ratio", "gdpp", "gdpp_log",
    "regions_East Asia", "regions_Europe", "regions_Middle East",
    "regions_Central Asia", "regions_Americas", "regions_Other",
    "regions_South Asia", "regions_Oceania"
]

def to_dataframe(payload: dict) -> pd.DataFrame:
    df = pd.DataFrame([payload])
    for col in FEATURE_ORDER:
        if col not in df.columns:
            df[col] = 0
    return df[FEATURE_ORDER]

def transform(df: pd.DataFrame, scaler):
    if "import_export_ratio" not in df.columns:
        df["import_export_ratio"] = (df["imports"] + 1e-9) / (df["exports"] + 1e-9)
    if "gdpp_log" not in df.columns:
        df["gdpp_log"] = np.log1p(df["gdpp"])
    X = df.values
    X_scaled = scaler.transform(X)
    return X_scaled


app/routes.py
from flask import Blueprint, current_app, request, jsonify
from .preprocess import to_dataframe, transform

bp = Blueprint("api", __name__)

@bp.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)
        df = to_dataframe(payload)
        X = transform(df, current_app.config["SCALER"])
        model = current_app.config["MODEL"]
        cluster = int(model.predict(X)[0])

        response = {
            "cluster": cluster,
            "rationale": [
                "Higher child mortality and fertility elevate need",
                "Lower income and life expectancy increase priority",
                "Trade imbalance (imports > exports) indicates vulnerability"
            ],
            "note": "Use with country context; clustering is comparative, not prescriptive."
        }
        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


Dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=5000
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "run:app"]


Training and serialization snippet (put in your notebook)
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# X: final feature matrix (engineered features + region one-hot in FEATURE_ORDER)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
kmeans.fit(X_scaled)

joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(kmeans, "models/aid_kmeans.pkl")



README.md
Title
HELP International — Country Clustering API for Aid Prioritization
Problem statement
HELP International needs a robust, data-driven way to prioritize aid across countries with diverse socioeconomic and health profiles. We segment countries via unsupervised learning and expose an API to classify new country inputs into need-based clusters.
Target metric
- Internal clustering quality via the Silhouette coefficient (higher is better).
- Visual validation via PCA cluster separability and interpretability of centroids.
Approach
- EDA: Distributions, outliers, log-transform for skewed gdpp, feature creation (import_export_ratio), correlation heatmap.
- Hypothesis testing:
- Inflation vs GDP per capita: r\approx -0.33, p\approx 1.1\times 10^{-5} → significant inverse relationship.
- Fertility vs income: negative correlation.
- Child mortality vs income: negative correlation.
- Modeling:
- K-Means (elbow → k=3) with silhouette ≈ 0.1898.
- Hierarchical clustering (Ward, 4 clusters) with silhouette ≈ 0.1846.
- DBSCAN baseline failed to form ≥2 clusters; documented parameter sensitivity.
- Validation: PCA confirms coherent structure; clusters map to development tiers (high/mid/low).
- Insights & recommendations: Health efficiency, fertility reduction, inflation control, trade diversification; India clusters with developing economies yet shows transitional potential.
Final scores
- K-Means silhouette: 0.1898 (selected model).
- Hierarchical silhouette: 0.1846.
- DBSCAN: Not computable in baseline (single cluster/noise).
Deployment steps
- API: Flask app with /predict endpoint; consistent preprocessing and scaling; returns cluster with rationale.
- Containerization: Docker + Gunicorn for production serving.
- AWS hosting (options):
- Elastic Beanstalk for rapid deployment (managed EC2 + ALB).
- ECS Fargate for container orchestration, autoscaling, and HTTPS via ACM.
Example request
POST /predict
Content-Type: application/json

{
  "child_mort": 55.2,
  "total_fer": 3.7,
  "life_expec": 68.5,
  "income": 4500,
  "health": 5.2,
  "inflation": 7.8,
  "imports": 22.0,
  "exports": 15.4,
  "gdpp": 2100,
  "regions_South Asia": 1
}


Example response
{
  "cluster": 2,
  "rationale": [
    "Higher child mortality and fertility elevate need",
    "Lower income and life expectancy increase priority",
    "Trade imbalance (imports > exports) indicates vulnerability"
  ],
  "note": "Use with country context; clustering is comparative, not prescriptive."
}


Governance and maintenance
- Clustering offers comparative insights, not causal prescriptions.
- Combine with expert judgment and up-to-date country context.
- Retrain quarterly with new data; version models and scalers; monitor drift.

**Blog**
For a detailed analysis and insights from the project, see the Technical Blog at https://medium.com/@ayushivaishya199/creating-a-clustering-algorithm-for-stratergic-aid-allocation-70781108508d?postPublishedType=initial

**Tableau Dashboard**
For the Tableau Dashboard it can be accessed/viewed at https://github.com/AyushiVai123/DS-Portfolio-Project-Submission-Clustering-Countries-for-Strategic-Aid-Allocation/blob/main/Clustering%20Countries%20for%20Strategic%20Aid%20Allocation.twbx
