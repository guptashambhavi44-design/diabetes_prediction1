
import streamlit as st
import pandas as pd
import pickle

# ================= PAGE SETTINGS =================
st.set_page_config(
    page_title="Diabetes AI Dashboard",
    page_icon="🧬",
    layout="wide"
)

# ================= DARK THEME =================
st.markdown("""
<style>
.stApp {
    background-color: #0b1020;
    color: #e5e7eb;
}

[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #263249;
}

h1, h2, h3 {
    color: #f8fafc !important;
}

p, label {
    color: #cbd5e1 !important;
}

div[data-testid="stMetric"] {
    background: #151e31;
    border: 1px solid #2b3852;
    padding: 18px;
    border-radius: 16px;
}

div[data-testid="stMetric"] label {
    color: #94a3b8 !important;
}

div[data-testid="stMetricValue"] {
    color: #a78bfa !important;
}

.stButton button {
    background: linear-gradient(90deg, #7c3aed, #4f46e5);
    color: white;
    border: none;
    border-radius: 12px;
    height: 48px;
    font-weight: bold;
}

.stButton button:hover {
    border: 1px solid #a78bfa;
    color: white;
}

div[data-testid="stForm"] {
    background-color: #111827;
    border: 1px solid #29364f;
    border-radius: 18px;
    padding: 20px;
}

div[data-testid="stExpander"] {
    background-color: #111827;
    border: 1px solid #29364f;
    border-radius: 12px;
}

hr {
    border-color: #29364f;
}

</style>
""", unsafe_allow_html=True)


# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    with open("diabetes_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()


# ================= SIDEBAR =================
with st.sidebar:

    st.title("🧬 Diabetes AI")

    st.caption("MACHINE LEARNING DASHBOARD")

    st.divider()

    page = st.radio(
        "Navigation",
        ["Dashboard", "About Project"]
    )

    st.divider()

    st.markdown("### System Status")

    st.success("Model loaded successfully")

    st.info("Prediction system ready")


# ================= DASHBOARD =================
if page == "Dashboard":

    st.title("🩺 Diabetes Prediction")
    st.caption("AI-powered classification dashboard")

    st.divider()

    # Top information cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("System", "ONLINE")

    with col2:
        st.metric("Model Type", "ML Pipeline")

    with col3:
        st.metric("Prediction", "Binary Classification")

    st.divider()

    # Get columns from model preprocessing
    preprocessor = model.named_steps["preprocessing"]

    num_cols = list(preprocessor.transformers_[0][2])
    cat_cols = list(preprocessor.transformers_[1][2])

    cat_pipeline = preprocessor.named_transformers_["cat"]
    encoder = cat_pipeline.named_steps["encoder"]
    categories = encoder.categories_

    st.subheader("👤 Patient Information")
    st.write("Enter the information below and generate a prediction.")

    input_values = {}

    # Input form
    with st.form("prediction_form"):

        left, right = st.columns(2)

        for i, col in enumerate(num_cols):

            name = col.lower()

            if "age" in name:
                default, minimum, maximum = 25.0, 1.0, 100.0

            elif "bmi" in name:
                default, minimum, maximum = 25.0, 10.0, 60.0

            elif "glucose" in name:
                default, minimum, maximum = 100.0, 40.0, 300.0

            elif "pressure" in name:
                default, minimum, maximum = 80.0, 30.0, 200.0

            elif "insulin" in name:
                default, minimum, maximum = 100.0, 0.0, 1000.0

            elif "skin" in name:
                default, minimum, maximum = 20.0, 0.0, 100.0

            elif "pregnan" in name:
                default, minimum, maximum = 0.0, 0.0, 20.0

            else:
                default, minimum, maximum = 0.0, 0.0, 1000.0

            with left if i % 2 == 0 else right:

                input_values[col] = st.slider(
                    col,
                    min_value=float(minimum),
                    max_value=float(maximum),
                    value=float(default)
                )

        for i, col in enumerate(cat_cols):

            with left if i % 2 == 0 else right:

                input_values[col] = st.selectbox(
                    col,
                    list(categories[i])
                )

        submitted = st.form_submit_button(
            "🔮 Generate Prediction",
            use_container_width=True
        )

    # ================= PREDICTION =================

    if submitted:

        input_data = pd.DataFrame([input_values])

        prediction = model.predict(input_data)[0]

        st.divider()
        st.subheader("📊 Prediction Results")

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            if prediction == 1:
                st.error("Model Classification: Class 1")
                st.warning("Higher likelihood predicted by the model.")
            else:
                st.success("Model Classification: Class 0")
                st.info("Lower likelihood predicted by the model.")

        with result_col2:

            if hasattr(model, "predict_proba"):

                probability = model.predict_proba(input_data)[0][1]

                st.metric(
                    "Class 1 Probability",
                    f"{probability * 100:.2f}%"
                )

                st.progress(float(probability))

        # Patient details
        with st.expander("📋 View Input Summary", expanded=True):

            summary = pd.DataFrame(
                list(input_values.items()),
                columns=["Feature", "Value"]
            )

            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "⬇ Download Report",
                summary.to_csv(index=False),
                file_name="patient_report.csv",
                mime="text/csv"
            )

        st.warning(
            "Educational project only. This prediction is not a medical diagnosis."
        )


# ================= ABOUT PAGE =================
elif page == "About Project":

    st.title("📚 About The Project")

    st.caption("Machine Learning Based Diabetes Prediction")

    st.divider()

    st.subheader("🎯 Objective")

    st.write(
        "This project explores machine learning classification "
        "to predict diabetes-related outcomes using input features."
    )

    st.subheader("⚙️ Project Workflow")

    st.write("""
    1. Data collection
    2. Data cleaning
    3. Missing value treatment
    4. Exploratory data analysis
    5. Feature preprocessing
    6. Model training
    7. Model evaluation
    8. Prediction interface
    """)

    st.subheader("🤖 Machine Learning Models")

    st.write("""
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - Gradient Boosting
    """)

    st.subheader("💻 Technologies Used")

    st.write("""
    Python • Pandas • NumPy • Scikit-learn • Matplotlib • Streamlit
    """)

    st.info(
        "This application was created for educational purposes."
    )
