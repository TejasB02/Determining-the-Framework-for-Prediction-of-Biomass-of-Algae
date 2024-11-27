import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Load and preprocess your dataset
new_data = pd.read_csv("new_data.csv")
X = new_data.drop(columns=['density'])
y = new_data['density']

# Train the Random Forest model if not pre-trained
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X, y)

# Load the best model identified from cross-validation
try:
    best_model = joblib.load("best_model_CatBoost.pkl")
    st.success("Model loaded successfully.")
except FileNotFoundError:
    st.warning("Model file not found. Using only Random Forest model.")
    best_model = None

# Calculate the optimal threshold for density (using 75th percentile as an example)
optimal_density_threshold = y.quantile(0.75)
st.write(f"Optimal Density Threshold : {optimal_density_threshold:.4f}")


# Function to make ensemble predictions
def ensemble_predictions(models, X, weights=None):
    predictions = []
    for model in models:
        if model is not None:
            predictions.append(model.predict(X))

    if weights and len(weights) == len(predictions):
        weighted_preds = sum(w * pred for w, pred in zip(weights, predictions))
        return weighted_preds / sum(weights)
    else:
        return sum(predictions) / len(predictions)


# Define weights (optional, based on model performance; here we assume equal weighting)
weights = [0.5, 0.5] if best_model else [1.0]  # Adjust based on model performance
models = [rf_model]
if best_model is not None:
    models.append(best_model)

# Streamlit inputs
st.title("Spirulina Growth Rate Predictor")

# User input form
st.header("Input Environmental Factors")

user_input = {
    "temperature": st.number_input("Temperature (°C)", value=0.0, format="%.10f"),
    "irradiance": st.write("0 to ~2000 µmol m-2 s-1") or
                  st.number_input("Irradiance (µmol m-2 s-1)", value=0.0, format="%.10f"),
    "ph": st.write("0 to 14 (Neutral is 7)") or
           st.number_input("pH Level", value=0.0, format="%.10f"),
    "oxygen": st.write("0 to ~14 mg/L (Dissolved oxygen)") or
              st.number_input("Oxygen (mg/L)", value=0.0, format="%.10f"),
    "nitrate": st.write("Suggested range: 0 to 10 mg/L (Drinking water standards)") or
               st.number_input("Nitrate (mg/L)", value=0.0, format="%.10f"),
    "phosphate": st.write("Suggested range: 0 to 0.1 mg/L (Eutrophication concern)") or
                 st.number_input("Phosphate (mg/L)", value=0.0, format="%.10f"),
    "chlorophyll": st.write("Suggested range: 0 to ~10 mg/L (Natural productivity)") or
                   st.number_input("Chlorophyll (mg/L)", value=0.0, format="%.10f"),
    "salinity": st.write("Suggested range: 0 ppt (pure water) to ~35 ppt (seawater)") or
                st.number_input("Salinity (ppt)", value=0.0, format="%.10f")
}
user_input_df = pd.DataFrame([user_input])
user_input_df = user_input_df[X.columns]  # Reorder to match X

# Prediction
if st.button("Predict Density"):
    user_input_density = ensemble_predictions(models, user_input_df, weights)
    st.write(f"Predicted Density for User Input: {user_input_density[0]:.4f}" if isinstance(user_input_density,
                                                                                            np.ndarray) else f"Predicted Density for User Input: {user_input_density:.4f}")

    if user_input_density >= optimal_density_threshold:
        st.success("The predicted density is within the optimal range for Spirulina growth.")
    else:
        st.warning("The predicted density is below the optimal threshold. Consider adjustments for optimal growth.")

        # Calculate feature importance
        feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)

        # Filter optimal data for recommendations
        new_data['density'] = ensemble_predictions(models, X, weights)
        optimal_data = new_data[new_data['density'] >= optimal_density_threshold]

        # Recommend adjustments
        def recommend_adjustments_with_importance(user_input, optimal_data, feature_importance):
            recommendations = {}
            for feature in feature_importance.index:
                optimal_min = optimal_data[feature].min()
                optimal_max = optimal_data[feature].max()
                value = user_input.get(feature, None)

                if value is not None:
                    if value < optimal_min:
                        recommendations[
                            feature] = f"Increase {feature} to be within the range {optimal_min:.2f} to {optimal_max:.2f}"
                    elif value > optimal_max:
                        recommendations[
                            feature] = f"Decrease {feature} to be within the range {optimal_min:.2f} to {optimal_max:.2f}"
                    else:
                        recommendations[feature] = f"{feature} is within the optimal range."
                else:
                    recommendations[feature] = f"{feature} value is missing in user input."
            return recommendations

        recommendations = recommend_adjustments_with_importance(user_input, optimal_data, feature_importance)

        # Display recommendations
        st.header("Recommendations for Optimal Growth")
        for feature, recommendation in recommendations.items():
            st.write(f"{feature}: {recommendation}")
