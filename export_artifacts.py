# Run this cell in your notebook RIGHT AFTER training (Cell 11 from before) —
# it saves everything the API needs: model weights, the fitted scaler, and the
# exact column order/schema so inference-time data lines up with training-time data.

import torch
import joblib
import json

# 1. Save the trained PyTorch model (architecture + weights via TorchScript,
#    so the API doesn't need your FailureClassifier class definition at all)
model.eval()
example_input = torch.randn(1, X_tr.shape[1]).to(device)
traced_model = torch.jit.trace(model, example_input)
traced_model.save("battery_failure_model.pt")

# 2. Save the fitted StandardScaler (needed to scale new raw readings the same way)
joblib.dump(scaler, "scaler.pkl")

# 3. Save the exact column order + dtypes the model expects, plus fallback values
#    for handling missing/incomplete sensor payloads at inference time (mirrors
#    the imputation logic from the cleaning pipeline).
schema = {
    "feature_columns": list(X_train.columns),
    "categorical_source_cols": categorical_cols,   # pre-one-hot-encoding names, for reference
    "numeric_source_cols": numeric_cols,
    "numeric_medians": {col: float(df_clean[col].median()) for col in numeric_cols},
    "categorical_fill_value": "Unknown",
    "decision_threshold": 0.5   # tune this later based on precision/recall trade-off Voltrix wants
}
with open("model_schema.json", "w") as f:
    json.dump(schema, f, indent=2)

print("Saved: battery_failure_model.pt, scaler.pkl, model_schema.json")
print(f"Model expects {len(schema['feature_columns'])} input features in this exact order.")
