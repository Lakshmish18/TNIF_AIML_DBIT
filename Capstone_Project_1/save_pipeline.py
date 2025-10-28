# Capstone_Project_1/save_pipeline.py  (run from repo root or notebook)
import os, joblib, json, sklearn
ARTIFACTS_DIR = "Capstone_Project_1/artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Assume `best_model`, `feature_columns_list`, and metrics variables exist in your notebook.
# If you run as a script, re-run training or load your pipeline object into `best_model`.
# Example: if best_model variable exists in the notebook, use:
final_pipeline = best_model  # replace with your pipeline variable

# Save pipeline (binary)
model_path = os.path.join(ARTIFACTS_DIR, "final_ridge_pipeline.joblib")
joblib.dump(final_pipeline, model_path, compress=3)
print("Saved pipeline to:", model_path)

# Save feature list (must be exact order used at training)
feature_columns_list = feature_columns  # or however you computed the list
with open(os.path.join(ARTIFACTS_DIR, "feature_columns.json"), "w") as f:
    json.dump(feature_columns_list, f, indent=2)
print("Saved feature_columns.json")

# Save basic metadata
import datetime, platform
meta = {
    "model_file": os.path.basename(model_path),
    "timestamp": datetime.datetime.now().isoformat(),
    "sklearn_version": sklearn.__version__
}
with open(os.path.join(ARTIFACTS_DIR, "model_metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)
print("Saved metadata")
