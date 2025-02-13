import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

def optimize_random_forest(X, y, test_size=0.2):
    """
    Optimize Random Forest classifier with train/test metrics.
    
    Parameters:
    X: Feature matrix with 3 parameters
    y: Target variable (binary classification)
    test_size: Proportion of dataset to use for testing
    
    Returns:
    best_model: Optimized Random Forest model
    scaler: Fitted StandardScaler
    """
    # Split data first
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Parameter grid optimized for large dataset
    param_grid = {
        'n_estimators': [100, 200],  # More trees for large dataset
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'max_features': ['sqrt']  # Good default for classification
    }
    
    # Create and train model
    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,  # Use all CPU cores
        class_weight='balanced'  # Handle any class imbalance
    )
    
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit the model
    print("Training model...")
    grid_search.fit(X_train_scaled, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Print comprehensive results
    print("\nBest parameters:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)
    
    # Calculate and print train accuracy
    train_pred = best_model.predict(X_train_scaled)
    train_accuracy = accuracy_score(y_train, train_pred)
    print(f"\nTrain Accuracy: {train_accuracy:.4f}")
    
    # Calculate and print test accuracy
    test_pred = best_model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, test_pred)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Print feature importance
    importance = best_model.feature_importances_
    print("\nFeature Importances:")
    for i, imp in enumerate(importance):
        print(f"Feature {i+1}: {imp:.4f}")
    
    # Print detailed classification report
    print("\nClassification Report on Test Set:")
    print(classification_report(y_test, test_pred))
    
    return best_model, scaler

# Example usage
"""
# Assuming your data is in X and y
model, scaler = optimize_random_forest(X, y)

# To make predictions on new data:
X_new_scaled = scaler.transform(X_new)
predictions = model.predict(X_new_scaled)
"""
data = pd.read_csv('WildFires_DataSet.csv')
X = data[['NDVI', 'LST', 'BURNED_AREA']]
y = data['CLASS']
model, scaler = optimize_random_forest(X, y)