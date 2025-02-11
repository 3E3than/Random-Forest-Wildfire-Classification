import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

def trainClassifier():
    # Load dataset
    data = pd.read_csv('WildFires_DataSet.csv')
    x = data[['NDVI', 'LST', 'BURNED_AREA']]
    y = data['CLASS']

    # Split data into train and test sets
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=1)

    # Train the Random Forest model
    rf = RandomForestClassifier()
    rf.fit(xtrain, ytrain)

    # Make predictions
    testPredictions = rf.predict(xtest)

    # Save classification report to a text file
    report = classification_report(ytest, testPredictions)
    print(report)
    with open("classification_report.txt", "w") as f:
        f.write(report)

    # Save feature importance
    feature_importance = pd.DataFrame({
        'feature': x.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    print(feature_importance)
    feature_importance.to_csv("feature_importance.csv", index=False)

    # Create and save confusion matrix plot
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(ytest, testPredictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['no_fire', 'fire'],
                yticklabels=['no_fire', 'fire'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # Save the confusion matrix image
    plt.savefig("confusion_matrix.png")
    plt.show()

trainClassifier()