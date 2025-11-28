"""
Model Training Module
Contains functions for training and evaluating machine learning models
"""

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    test_size : float
        Proportion of test set
    random_state : int
        Random seed
        
    Returns:
    --------
    X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Training set: {X_train.shape[0]} properties")
    print(f"Test set: {X_test.shape[0]} properties")
    print(f"Train/Test split: {len(X_train)/(len(X_train)+len(X_test))*100:.1f}% / {len(X_test)/(len(X_train)+len(X_test))*100:.1f}%")
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """
    Scale features using StandardScaler
    
    Parameters:
    -----------
    X_train, X_test : pd.DataFrame
        Training and test features
        
    Returns:
    --------
    X_train_scaled, X_test_scaled, scaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Features scaled using StandardScaler")
    
    return X_train_scaled, X_test_scaled, scaler


def get_models():
    """
    Get dictionary of all models to train
    
    Returns:
    --------
    dict
        Dictionary of model names and model objects
    """
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=15),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=20, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=200, 
            random_state=42, 
            max_depth=5, 
            learning_rate=0.05,
            min_samples_split=5,
            min_samples_leaf=2
        ),
        'XGBoost': XGBRegressor(n_estimators=200, random_state=42, max_depth=7, learning_rate=0.1, n_jobs=-1),
        'LightGBM': LGBMRegressor(n_estimators=200, random_state=42, max_depth=7, learning_rate=0.1, n_jobs=-1, verbose=-1)
    }
    return models


def tune_best_models(X_train, y_train, X_test, y_test):
    """
    Tune hyperparameters for top 3 models using GridSearchCV
    
    Parameters:
    -----------
    X_train, X_test : array-like
        Training and test features
    y_train, y_test : array-like
        Training and test targets
        
    Returns:
    --------
    dict
        Dictionary of tuned models with their scores
    """
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING (GridSearchCV)")
    print("="*80)
    print("Tuning top 3 models: Gradient Boosting, LightGBM, Random Forest")
    print("This may take 5-10 minutes...\n")
    
    tuned_results = {}
    
    # 1. Gradient Boosting
    print("1. Tuning Gradient Boosting...")
    gb_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 7, 10],
        'learning_rate': [0.05, 0.1, 0.2],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    gb_grid = GridSearchCV(
        GradientBoostingRegressor(random_state=42),
        gb_params,
        cv=3,
        scoring='r2',
        n_jobs=-1,
        verbose=0
    )
    gb_grid.fit(X_train, y_train)
    
    gb_pred = gb_grid.predict(X_test)
    gb_r2 = r2_score(y_test, gb_pred)
    gb_mae = mean_absolute_error(y_test, gb_pred)
    
    print(f"   Best params: {gb_grid.best_params_}")
    print(f"   R² Score: {gb_r2:.4f}")
    print(f"   MAE: Rs.{gb_mae:.4f} Cr\n")
    
    tuned_results['Gradient Boosting'] = {
        'model': gb_grid.best_estimator_,
        'r2': gb_r2,
        'mae': gb_mae,
        'params': gb_grid.best_params_
    }
    
    # 2. LightGBM
    print("2. Tuning LightGBM...")
    lgb_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 7, 10, -1],
        'learning_rate': [0.05, 0.1, 0.2],
        'num_leaves': [31, 50, 70],
        'min_child_samples': [20, 30]
    }
    lgb_grid = GridSearchCV(
        LGBMRegressor(random_state=42, verbose=-1),
        lgb_params,
        cv=3,
        scoring='r2',
        n_jobs=-1,
        verbose=0
    )
    lgb_grid.fit(X_train, y_train)
    
    lgb_pred = lgb_grid.predict(X_test)
    lgb_r2 = r2_score(y_test, lgb_pred)
    lgb_mae = mean_absolute_error(y_test, lgb_pred)
    
    print(f"   Best params: {lgb_grid.best_params_}")
    print(f"   R² Score: {lgb_r2:.4f}")
    print(f"   MAE: Rs.{lgb_mae:.4f} Cr\n")
    
    tuned_results['LightGBM'] = {
        'model': lgb_grid.best_estimator_,
        'r2': lgb_r2,
        'mae': lgb_mae,
        'params': lgb_grid.best_params_
    }
    
    # 3. Random Forest
    print("3. Tuning Random Forest...")
    rf_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [15, 20, 25, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }
    rf_grid = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        rf_params,
        cv=3,
        scoring='r2',
        n_jobs=-1,
        verbose=0
    )
    rf_grid.fit(X_train, y_train)
    
    rf_pred = rf_grid.predict(X_test)
    rf_r2 = r2_score(y_test, rf_pred)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    
    print(f"   Best params: {rf_grid.best_params_}")
    print(f"   R² Score: {rf_r2:.4f}")
    print(f"   MAE: Rs.{rf_mae:.4f} Cr\n")
    
    tuned_results['Random Forest'] = {
        'model': rf_grid.best_estimator_,
        'r2': rf_r2,
        'mae': rf_mae,
        'params': rf_grid.best_params_
    }
    
    print("="*80)
    print("TUNING COMPLETE!")
    print("="*80)
    
    # Find best tuned model
    best_model_name = max(tuned_results, key=lambda x: tuned_results[x]['r2'])
    print(f"\nBest Tuned Model: {best_model_name}")
    print(f"R² Score: {tuned_results[best_model_name]['r2']:.4f}")
    print(f"MAE: Rs.{tuned_results[best_model_name]['mae']:.4f} Cr")
    print(f"Best Parameters: {tuned_results[best_model_name]['params']}")
    print("="*80 + "\n")
    
    return tuned_results


def train_single_model(name, model, X_train, X_test, y_train, y_test, 
                       X_train_scaled=None, X_test_scaled=None):
    """
    Train and evaluate a single model
    
    Parameters:
    -----------
    name : str
        Model name
    model : sklearn model
        Model instance
    X_train, X_test : array-like
        Training and test features
    y_train, y_test : array-like
        Training and test targets
    X_train_scaled, X_test_scaled : array-like, optional
        Scaled features for linear models
        
    Returns:
    --------
    dict
        Dictionary containing model and metrics
    """
    print(f"\nTraining {name}...")
    
    # Train
    if 'Linear' in name or 'Ridge' in name or 'Lasso' in name:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE: Rs.{rmse/10000000:.4f} Cr")
    print(f"  MAE: Rs.{mae/10000000:.4f} Cr")
    print(f"  CV R² (mean): {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    return {
        'model': model,
        'predictions': y_pred,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'CV_R2_mean': cv_scores.mean(),
        'CV_R2_std': cv_scores.std()
    }


def train_all_models(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled):
    """
    Train all models and store results
    
    Parameters:
    -----------
    X_train, X_test : array-like
        Training and test features
    y_train, y_test : array-like
        Training and test targets
    X_train_scaled, X_test_scaled : array-like
        Scaled features
        
    Returns:
    --------
    dict
        Dictionary of model results
    """
    models = get_models()
    results = {}
    
    for name, model in models.items():
        results[name] = train_single_model(
            name, model, X_train, X_test, y_train, y_test,
            X_train_scaled, X_test_scaled
        )
    
    print("\n" + "="*80)
    print("All models trained successfully!")
    print("="*80)
    
    return results


def calculate_train_test_comparison(results, X_train, X_test, y_train, y_test,
                                    X_train_scaled, X_test_scaled):
    """
    Calculate training vs test performance comparison
    
    Parameters:
    -----------
    results : dict
        Dictionary of model results
    X_train, X_test : array-like
        Training and test features
    y_train, y_test : array-like
        Training and test targets
    X_train_scaled, X_test_scaled : array-like
        Scaled features
        
    Returns:
    --------
    list
        List of comparison dictionaries
    """
    train_test_comparison = []
    
    for name, model_info in results.items():
        model = model_info['model']
        
        # Get predictions for training and test sets
        if 'Linear' in name or 'Ridge' in name or 'Lasso' in name:
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)
        else:
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
        
        # Calculate R² scores
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Calculate accuracy (within ±10%)
        train_accuracy = np.mean(np.abs((y_train - y_train_pred) / y_train) <= 0.10) * 100
        test_accuracy = np.mean(np.abs((y_test - y_test_pred) / y_test) <= 0.10) * 100
        
        # Store results
        train_test_comparison.append({
            'Model': name,
            'Train R²': train_r2,
            'Test R²': test_r2,
            'R² Difference': train_r2 - test_r2,
            'Train Accuracy (±10%)': train_accuracy,
            'Test Accuracy (±10%)': test_accuracy,
            'Accuracy Difference': train_accuracy - test_accuracy
        })
    
    return train_test_comparison


def get_best_model(results):
    """
    Get the best performing model
    
    Parameters:
    -----------
    results : dict
        Dictionary of model results
        
    Returns:
    --------
    str, dict
        Best model name and its info
    """
    best_model_name = max(results.keys(), key=lambda k: results[k]['R2'])
    best_model_info = results[best_model_name]
    
    return best_model_name, best_model_info


def calculate_additional_metrics(y_test, predictions):
    """
    Calculate additional performance metrics
    
    Parameters:
    -----------
    y_test : array-like
        True values
    predictions : array-like
        Predicted values
        
    Returns:
    --------
    dict
        Dictionary of metrics
    """
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
    accuracy_within_10_pct = np.mean(np.abs((y_test - predictions) / y_test) <= 0.10) * 100
    accuracy_within_20_pct = np.mean(np.abs((y_test - predictions) / y_test) <= 0.20) * 100
    
    return {
        'MAPE': mape,
        'Accuracy_10pct': accuracy_within_10_pct,
        'Accuracy_20pct': accuracy_within_20_pct
    }
