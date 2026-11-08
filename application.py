import streamlit as st
import joblib

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="Spam Mail Classifier",
    page_icon="📧",
    layout="centered"
)

# =========================
# Load Models
# =========================
@st.cache_resource
def load_models():
    lr_model = joblib.load("models/best_lr.pkl")
    nb_model = joblib.load("models/best_nb.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

    return lr_model, nb_model, vectorizer


lr_model, nb_model, vectorizer = load_models()

# =========================
# Title
# =========================
st.title("📧 Spam Mail Classifier")
st.write(
    "Detect whether a message is Spam or Legitimate using "
    "Logistic Regression and Naive Bayes."
)

# =========================
# Input
# =========================
message = st.text_area(
    "Enter Message",
    placeholder="Type your email or SMS message here..."
)

# =========================
# Prediction
# =========================
if st.button("Predict"):

    if not message.strip():
        st.warning("⚠️ Please enter a message.")

    else:

        # Transform text
        message_vector = vectorizer.transform([message])

        # Logistic Regression
        lr_pred = lr_model.predict(message_vector)[0]
        lr_prob = lr_model.predict_proba(message_vector).max() * 100

        # Naive Bayes
        nb_pred = nb_model.predict(message_vector)[0]
        nb_prob = nb_model.predict_proba(message_vector).max() * 100

        # ==================================
        # Individual Model Results
        # ==================================
        st.subheader("🤖 Individual Model Predictions")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Logistic Regression")

            if lr_pred == "Spam Mail":
                st.error(f"🚨 Spam Mail ({lr_prob:.2f}%)")
            else:
                st.success(f"✅ Legitimate Mail ({lr_prob:.2f}%)")

        with col2:
            st.markdown("### Naive Bayes")

            if nb_pred == "Spam Mail":
                st.error(f"🚨 Spam Mail ({nb_prob:.2f}%)")
            else:
                st.success(f"✅ Legitimate Mail ({nb_prob:.2f}%)")

        st.divider()

        # ==================================
        # Combined Verdict
        # ==================================
        st.subheader("🎯 Combined Model Verdict")

        if lr_pred == nb_pred:

            final_prediction = lr_pred

            st.info("✅ Both models agree on the prediction.")

        else:

            st.warning(
                "⚠️ Models disagree. Choosing prediction with higher confidence."
            )

            final_prediction = (
                lr_pred if lr_prob >= nb_prob else nb_pred
            )

        if final_prediction == "Spam Mail":
            st.error("🚨 FINAL VERDICT: SPAM MAIL")
        else:
            st.success("✅ FINAL VERDICT: LEGITIMATE MAIL")

        st.divider()

        # ==================================
        # Confidence Scores
        # ==================================
        st.subheader("📊 Confidence Scores")

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "Logistic Regression",
                f"{lr_prob:.2f}%"
            )

        with c2:
            st.metric(
                "Naive Bayes",
                f"{nb_prob:.2f}%"
            )

        st.metric(
            "Average Confidence",
            f"{(lr_prob + nb_prob)/2:.2f}%"
        )

        # ==================================
        # Debug (Optional)
        # ==================================
        with st.expander("Debug Predictions"):
            st.write("Logistic Regression:", lr_pred)
            st.write("Naive Bayes:", nb_pred)