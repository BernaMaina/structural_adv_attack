#!/usr/bin/env python
# coding: utf-8

import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import metrics
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
#from scikeras.wrappers import KerasClassifier
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV


def load_and_prepare_data():
    """
    Load and prepare data.
    
    """
    ben = pd.read_csv('./csv_files/hybrid_ben_data.csv')
    mal = pd.read_csv('./csv_files/hybrid_mal_data.csv')
    data = pd.concat([ben, mal], ignore_index=True)
    data.drop(columns=['arch'], inplace=True)
    Feature = np.array(data.iloc[:, 2:])
    Label = np.array(data.iloc[:, 1])
    Feature_names = data.columns[2:]
    Label = np.where(Label != 0, 1, 0)
    return Feature, Label, Feature_names

def split_and_scale_data(Feature, Label):
    
    """
    Split data into training and test sets, then scale features.
    
    """
    X_train, X_test, y_train, y_test = train_test_split(Feature, Label, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, '../models/scaler.joblib')
    return X_train_scaled, X_test_scaled, y_train, y_test
   

def train_random_forest(X_train, y_train, X_test, y_test, feature_names):
    
    """Train Random Forest model with hyperparameter tuning and evaluate."""
    
    rf = RandomForestClassifier(random_state=42)
    param_dist_rf = {
        'n_estimators': [100, 200, 300, 400],
        'max_features': ['sqrt', 'log2', None],
        'max_depth': [10, 20, 30, 40, None],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 6],
        'bootstrap': [True, False]
    }
    random_search_rf = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist_rf,
        n_iter=50,
        cv=5,
        verbose=2,
        random_state=42,
        n_jobs=-1,
        error_score='raise'
    )
    random_search_rf.fit(X_train, y_train)
    joblib.dump(random_search_rf.best_estimator_, '../models/RF.joblib')
    y_pred_rf = random_search_rf.best_estimator_.predict(X_test)
    metrics_rf = compute_metrics(y_test, y_pred_rf, '../models/RF_metrics.txt', feature_names)
    
    perform_shap_analysis(random_search_rf.best_estimator_, X_train, X_test, feature_names, 'RF')
    
    return metrics_rf

def train_xgboost(X_train, y_train, X_test, y_test, feature_names):
    """Train XGBoost model with hyperparameter tuning and evaluate."""
    xgboost_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    param_dist_xgb = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'gamma': [0, 0.1, 0.2, 0.3],
        'reg_alpha': [0, 0.1, 0.5, 1],
        'reg_lambda': [0.1, 0.5, 1, 2]
    }
    random_search_xgb = RandomizedSearchCV(
        estimator=xgboost_model,
        param_distributions=param_dist_xgb,
        n_iter=50,
        cv=5,
        verbose=2,
        random_state=42,
        n_jobs=-1,
        error_score='raise'
    )
    random_search_xgb.fit(X_train, y_train)
    joblib.dump(random_search_xgb.best_estimator_, '../models/xgboost.joblib')
    y_pred_xgb = random_search_xgb.best_estimator_.predict(X_test)
    metrics_xgb = compute_metrics(y_test, y_pred_xgb, '../models/xgboost_metrics.txt', feature_names)

    perform_shap_analysis(random_search_xgb.best_estimator_, X_train, X_test, feature_names, 'XGBoost')
    return metrics_xgb

def train_svm(X_train, y_train, X_test, y_test, feature_names):
    """Train SVM model with hyperparameter tuning and evaluate."""
    svm_model = SVC(probability=False, random_state=42)  # Ensure probability=False
    param_dist_svm = {
        'C': [0.01, 0.1, 1, 10, 100],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': ['scale', 'auto'],
        'degree': [2, 3, 4, 5]  # Only applicable for 'poly' kernel
    }
    random_search_svm = RandomizedSearchCV(
        estimator=svm_model,
        param_distributions=param_dist_svm,
        n_iter=50,
        cv=5,
        verbose=2,
        random_state=42,
        n_jobs=-1,
        error_score='raise'
    )
    random_search_svm.fit(X_train, y_train)
    joblib.dump(random_search_svm.best_estimator_, '../models/svm.joblib')
    y_pred_svm = random_search_svm.best_estimator_.predict(X_test)
    metrics_svm = compute_metrics(y_test, y_pred_svm, '../models/svm_metrics.txt', feature_names)

    
    return metrics_svm


def build_dnn(input_dim, optimizer='adam', units=64):
    """Build DNN model with Keras."""
    model = Sequential([
        Dense(units, input_dim=input_dim, activation='relu'),
        Dense(units, activation='relu'),
        Dense(units, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_dnn(X_train, y_train, X_test, y_test, feature_names):
    """Train DNN model with Keras, perform hyperparameter tuning, and evaluate."""
    input_dim = X_train.shape[1]
    model = KerasClassifier(build_fn=build_dnn, verbose=0, input_dim=input_dim)

    param_dist = {
        'batch_size': [16, 32, 64, 128],   
        'epochs': [50, 100, 150, 200],     
        'optimizer': ['Adam', 'Nadam', 'RMSprop', 'SGD'],
        'units': [32, 64, 128, 256]        
    }

    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, 
                                       n_iter=50, cv=5, verbose=2, n_jobs=-1)
    
    random_search_result = random_search.fit(X_train, y_train)

    best_model = random_search_result.best_estimator_.model
    best_model.save('../models/dnn_model.h5')

    y_pred_dnn = (best_model.predict(X_test) > 0.5).astype('int32')
    metrics_dnn = compute_metrics(y_test, y_pred_dnn, '../models/dnn_metrics.txt', feature_names)
    #
    #perform_shap_analysis(best_model, X_train, X_test, feature_names, 'DNN')
    
    return metrics_dnn

def compute_metrics(y_test, y_pred, metrics_file, feature_names):
    """Compute and save evaluation metrics."""
    
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    with open(metrics_file, 'w') as f:
        f.write(f'Accuracy: {accuracy}\n')
        f.write(f'Precision: {precision}\n')
        f.write(f'Recall: {recall}\n')
        f.write(f'F1 Score: {f1}\n')
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'Feature_names': feature_names}

def perform_shap_analysis(model, X_train, X_test, feature_names, model_name):
    
    """Perform SHAP analysis and save plots."""
    
    if model_name == 'RF':
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        shap_values = shap_values[1]
    elif model_name == 'XGBoost':
        explainer = shap.Explainer(model)
        shap_values = explainer(X_test)
    elif model_name == 'SVM':
        explainer = shap.KernelExplainer(model.predict, X_train)
        shap_values = explainer.shap_values(X_test)
    elif model_name == 'DNN':
        explainer = shap.KernelExplainer(model.predict, X_train)
        shap_values = explainer.shap_values(X_test)
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_size=(10, 7), max_display=20, show=False)
    plt.savefig(f'../models/{model_name}_SHAP.png', dpi=300, bbox_inches='tight')
    plt.show()




def main():
    
    """Main function to run the model training and evaluation."""
    
    Feature, Label, Feature_names = load_and_prepare_data()
    X_train, X_test, y_train, y_test = split_and_scale_data(Feature, Label)
    
    # Train and evaluate models
    metrics_rf = train_random_forest(X_train, y_train, X_test, y_test, Feature_names)
    metrics_xgb = train_xgboost(X_train, y_train, X_test, y_test, Feature_names)
    metrics_svm = train_svm(X_train, y_train, X_test, y_test, Feature_names)
    metrics_dnn = train_dnn(X_train, y_train, X_test, y_test, Feature_names)
    
    # Print metrics
    print(f"Random Forest Metrics: {metrics_rf}")
    print(f"XGBoost Metrics: {metrics_xgb}")
    print(f"SVM Metrics: {metrics_svm}")
    print(f"DNN Metrics: {metrics_dnn}")

if __name__ == '__main__':
    main()