[![Python](https://img.shields.io/badge/python-3.13.7-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Polars](https://img.shields.io/badge/polars-1.33.1-orange?logo=polars&logoColor=white)](https://www.pola.rs/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-green?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/xgboost-3.0.5-red?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)

# Heart Disease Classification
Classify Heart Disease risk using [Kaggle's Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction). Quick workflow from basic data exploration to model training, selection and interpretation.

## Worklflow
1. **Explore and Compare models.** Start with [./initial_modelling.ipynb](initial_modelling.ipynb). Two classifiers are compared via nested cross-validation — Logistic Regression and XGBoost. 


2. **Train the Final Model**. Train Logistic Regression in [./train_model.py](train_model.py), saved as `./model/model.joblib`. Run the script in CL: `python ./train_model.py `

3. **Inspect Features.** See an overview of final model feature importance in [./feature_importance.ipynb](feature_importance.ipynb).

# Set Up
## Data
The default data path is `./data/heart.csv`. The data set is **not provided**, it can be downloaded from kaggle. 

## Environment

The project environment was managed via Conda. There are 3 main files:
- `env.lock.yaml` — pinned, explicit dependencies, recommended for reproducibility
- `requirements.txt` — pinned, explicit Python dependencies only
- `env.yaml` — high-level dependencies

For Conda users:

```bash
conda env create -n $ENV_NAME -f ./env.lock.yaml
```

For pip users:

```bash
pip install -r ./requirements.txt
```
## Images
Static images for `feature_importance.ipynb` are provided in [`./images/`](images/) for convenience and to allow display on GitHub. If you wish to generate them on your own, you'll need to install [`kaleido`](https://github.com/plotly/Kaleido) in addition to the environment set up listed above. Note that as of version 1.0.0, `kaleido` requires a separate installation of Chromium, which is user-specific, hence not included in project. Either way, interactive Plotly plots are available directly in the notebook.