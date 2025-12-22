# mahdis 
#  Task E: Linear Model Evaluation and Reporting
#%%
import sys
sys.path.append('.')

import numpy as np
from Tess_models import TESSModels
from sklearn.metrics import precision_score, confusion_matrix, classification_report, roc_auc_score

# Initialize the linear model pipeline
lm_pipeline = TESSModels(csv_path='./tess_data.csv', samples_per_class=350, use_scaler=True)

# Load data to get summary info
X_train, X_test, y_train, y_test, metadata_test, X_test_std_copy, X_err_test, scaler = lm_pipeline.load_data()

report_lines = []

# Dataset Summary
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

# Run the linear model (this will also generate plots)
lm_model, y_pred_opt, proba_test, best_thresh, roc_tuple = lm_pipeline.run_linear_model(
    n_bins=1000,
    use_scaler=True,
    samples_per_class=350,
    random_state=42
)

# Get the true labels (y_test)
_, _, _, y_test, _, _, _, _ = lm_pipeline.load_data()

# Compute confusion matrix and precision
cm = confusion_matrix(y_test, y_pred_opt)
precision = precision_score(y_test, y_pred_opt)
auc = roc_auc_score(y_test, proba_test)

# Prepare report
report_lines.append("Model 1 - Logistic Regression")
report_lines.append(f"Optimal Threshold used: {best_thresh:.4f}")
report_lines.append(f"AUC-ROC: {auc:.4f}\n")

# Confusion matrix & precision
report_lines.append("Confusion Matrix:")
report_lines.append(f"{cm.tolist()}")
report_lines.append(f"Precision: {precision:.4f}\n")
report_lines.append("Classification report:")
report_lines.append(classification_report(y_test, y_pred_opt, target_names=['Non-Planet', 'Planet'], digits=4))
report_lines.append("-" * 40)

# Save report
report_filename = 'report_linear_model_assignment2_taskE.txt'
with open(report_filename, 'w') as f:
    f.write("\n".join(report_lines))

print(f"\nTask E completed! Report saved as '{report_filename}'.")
# %%
