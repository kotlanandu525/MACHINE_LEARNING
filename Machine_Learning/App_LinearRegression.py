import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(page_title="Linear Regression", layout="centered")

# --------------------------------------------------
# Load CSS
# --------------------------------------------------
def load_css(file):
    try:
        with open(file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass  # avoids crash during cloud deployment

load_css("style.css")

# --------------------------------------------------
# Title
# --------------------------------------------------
st.markdown(
    """
    <div class="card">
        <h1>Linear Regression</h1>
        <p>Predict <b>Tip Amount</b> from <b>Total Bill</b> using Linear Regression</p>
    </div>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# Load data
# --------------------------------------------------
@st.cache_data
def load_data():
    return sns.load_dataset("tips")

df = load_data()

# --------------------------------------------------
# Dataset preview
# --------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df.head())
st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# Prepare data
# --------------------------------------------------
X = df[["total_bill"]]
y = df["tip"]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# --------------------------------------------------
# Train model
# --------------------------------------------------
model = LinearRegression()
model.fit(x_train_scaled, y_train)

y_pred = model.predict(x_test_scaled)

# --------------------------------------------------
# Metrics
# --------------------------------------------------
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# --------------------------------------------------
# Visualization
# --------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Total Bill vs Tip")

fig, ax = plt.subplots()
ax.scatter(df["total_bill"], df["tip"], alpha=0.6)

x_line = np.linspace(df["total_bill"].min(), df["total_bill"].max(), 100).reshape(-1, 1)
y_line = model.predict(scaler.transform(x_line))

ax.plot(x_line, y_line, color="red")
ax.set_xlabel("Total Bill ($)")
ax.set_ylabel("Tip ($)")

st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# Performance
# --------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Model Performance")

c1, c2, c3 = st.columns(3)
c1.metric("MAE", f"{mae:.2f}")
c2.metric("RMSE", f"{rmse:.2f}")
c3.metric("R²", f"{r2:.2f}")

st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# Model interpretation
# --------------------------------------------------
st.markdown(
    f"""
    <div class="card">
        <h3>Model Interpretation</h3>
        <p>
            <b>Coefficient:</b> {model.coef_[0]:.3f}<br>
            <b>Intercept:</b> {model.intercept_:.3f}
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

bill = st.slider(
    "Enter Total Bill ($)",
    min_value=float(df["total_bill"].min()),
    max_value=float(df["total_bill"].max()),
    value=30.0
)

tip = model.predict(scaler.transform([[bill]]))[0]

st.markdown(
    f'<div class="prediction-box">Predicted Tip: ${tip:.2f}</div>',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)
