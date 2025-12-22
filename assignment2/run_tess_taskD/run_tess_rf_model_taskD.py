## Task D: Experimenting with Random Forest Hyperparameters
#%%
import sys
sys.path.append('..')

import numpy as np
from Tess_models import TESSModels
from sklearn.metrics import precision_score, confusion_matrix, roc_auc_score, average_precision_score, classification_report


# Define the four combinations of n_estimators and max_depth
hyperparams_list = [
    {'n_estimators': 100, 'max_depth': 5},
    {'n_estimators': 100, 'max_depth': 10},
    {'n_estimators': 200, 'max_depth': 5},
    {'n_estimators': 200, 'max_depth': 10},
]

# Initialize the class
rf_pipeline = TESSModels(csv_path='../tess_data.csv', samples_per_class=350)

X_train, X_test, y_train, y_test, metadata_test, X_test_std_copy, X_err_test, scaler = rf_pipeline.load_data()

report_lines = []

# dataset summary 
report_lines.append("Dataset Summary:")
report_lines.append("-" * 40)
report_lines.append(f"Total samples: {len(y_train) + len(y_test)}")
report_lines.append(f"Number of features (bins): {X_train.shape[1]}")
report_lines.append(f"Train/Test split: Train={len(y_train)}, Test={len(y_test)}")
report_lines.append(f"Original class distribution (before balancing): "
                    f"Class 0={np.sum(y_train==0)+np.sum(y_test==0)}, "
                    f"Class 1={np.sum(y_train==1)+np.sum(y_test==1)}")
report_lines.append(f"Balanced training set: Class 0={np.sum(y_train==0)}, Class 1={np.sum(y_train==1)}")
report_lines.append("-" * 40 + "\n")

for i, params in enumerate(hyperparams_list, 1):
    print(f"\nRunning model {i}: n_estimators={params['n_estimators']}, max_depth={params['max_depth']}")

    rf_model, y_pred, proba, threshold, _ = rf_pipeline.run_random_forest(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        preview_samples=0
    )
    
    # Compute confusion matrix and precision
    cm = confusion_matrix(rf_pipeline.y_test, y_pred)
    precision = precision_score(rf_pipeline.y_test, y_pred)
    # auc = roc_auc_score(rf_pipeline.y_test, proba)
    # ap = average_precision_score(rf_pipeline.y_test, proba)
    # class_report = classification_report(rf_pipeline.y_test, y_pred, target_names=['Non-Planet','Planet'], digits=4)

    
    report_lines.append(f"Model {i} - n_estimators={params['n_estimators']}, max_depth={params['max_depth']}")
    report_lines.append(f"Optimal Threshold used: {rf_pipeline.best_thresh:.4f}\n")
    report_lines.append(f"Confusion Matrix:\n{cm}")
    report_lines.append(f"Precision: {precision:.4f}\n")
    # report_lines.append(f"AUC-ROC: {auc:.4f}")
    # report_lines.append(f"Average Precision (AP): {ap:.4f}")
    # report_lines.append("\nClassification Report:\n" + class_report)
    # report_lines.append(f"Saved plots and model artifacts: tess_model{i}_RF_model.joblib, model{i}_RF_optimal_threshold.npy, PNG figures\n")
    report_lines.append("-" * 80)
    
# Save report
with open('report_random_forest_assignment2_taskD.txt', 'w') as f:
    f.write("\n".join(report_lines))

print("\nTask D completed! Report saved as 'report_random_forest_assignment2_taskD.txt'.")
# %%
