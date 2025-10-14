import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

st.set_page_config(page_title="ML Regression App", layout="wide")

st.title("📊 Machine Learning Regression App")
st.write("Upload a CSV file, choose features and target, and train a regression model easily!")

# File uploader
uploaded_file = st.file_uploader("📂 Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    st.write("### Dataset Summary")
    st.write(df.describe())

    # Feature and target selection
    all_columns = df.columns.tolist()
    x_cols = st.multiselect("Select Feature Columns (X)", options=all_columns)
    y_col = st.selectbox("Select Target Column (y)", options=all_columns)

    if x_cols and y_col:
        X = df[x_cols]
        y = df[y_col]

        # Train-test split
        test_size = st.slider("Test Size (as fraction)", 0.1, 0.5, 0.2)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Model selection
        st.write("### Choose a Regression Model")
        model_choice = st.radio(
            "Select Model",
            ("Linear Regression", "Random Forest Regressor")
        )

        # Train model
        if st.button("🚀 Train Model"):
            if model_choice == "Linear Regression":
                model = LinearRegression()
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Evaluation
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            st.subheader("📈 Model Performance")
            st.write(f"**R² Score:** {r2:.4f}")
            st.write(f"**MAE:** {mae:.4f}")
            st.write(f"**RMSE:** {rmse:.4f}")

            # Plot actual vs predicted
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, color='blue')
            ax.plot([y_test.min(), y_test.max()],
                    [y_test.min(), y_test.max()],
                    color='red', linestyle='--')
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Predicted Values")
            ax.set_title("Actual vs Predicted")
            st.pyplot(fig)

            # Show feature importance (for Random Forest)
            if model_choice == "Random Forest Regressor":
                importances = pd.Series(model.feature_importances_, index=x_cols)
                st.bar_chart(importances)
else:
    st.info("👆 Upload a CSV file to get started.")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit and Scikit-learn")
