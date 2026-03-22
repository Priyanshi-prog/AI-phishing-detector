import streamlit as st
import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
import joblib

# --- 1. FEATURE EXTRACTION ENGINE ---
def extract_features(url):
    """Extracts numerical features from a URL for the AI model."""
    features = {}
    
    # URL length (Phishing URLs are often long)
    features['url_length'] = len(url)
    
    # Presence of @ symbol (Used to hide real domain)
    features['have_at'] = 1 if "@" in url else 0
    
    # Presence of IP address instead of domain
    features['is_ip'] = 1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0
    
    # Prefix/Suffix in domain (using '-')
    domain = urlparse(url).netloc
    features['have_dash'] = 1 if "-" in domain else 0
    
    # Subdomain count (Phishing often uses many subdomains)
    features['dots_count'] = url.count('.')
    
    # Redirection using //
    features['redirection'] = 1 if url.rfind('//') > 7 else 0
    
    # HTTPS Check (Note: Modern phishers use HTTPS too, but lack of it is a red flag)
    features['is_https'] = 1 if url.startswith('https') else 0

    return pd.DataFrame([features])

# --- 2. MODEL BOILERPLATE (Mock Training Data) ---
@st.cache_resource
def train_initial_model():
    """Trains a simple Random Forest model on sample data for demo purposes."""
    # Data: [Length, @, IP, -, Dots, //, HTTPS] -> Label (1=Phishing, 0=Safe)
    data = [
        [15, 0, 0, 0, 1, 0, 1, 0], # google.com
        [120, 1, 0, 1, 5, 1, 0, 1], # suspicious-login-verify-account.com
        [22, 0, 0, 0, 2, 0, 1, 0], # github.com
        [85, 0, 1, 0, 4, 0, 0, 1], # 192.168.1.1/login.php
        [18, 0, 0, 0, 1, 0, 1, 0], # apple.com
        [95, 1, 0, 1, 3, 1, 1, 1], # secure-appleid-login.com/redirect
    ]
    columns = ['url_length', 'have_at', 'is_ip', 'have_dash', 'dots_count', 'redirection', 'is_https', 'target']
    df = pd.DataFrame(data, columns=columns)
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# --- 3. STREAMLIT UI ---
def main():
    st.set_page_config(page_title="AI Phishing Detector", page_icon="🛡️")
    
    st.title("🛡️ AI Phishing Website Detector")
    st.write("Enter a URL below to check if it's likely a phishing attempt.")

    # Input
    url_input = st.text_input("Enter URL (e.g., https://secure-login.example.com)", placeholder="https://...")

    if st.button("Analyze URL"):
        if url_input:
            with st.spinner('Analyzing patterns...'):
                # 1. Train/Load Model
                model = train_initial_model()
                
                # 2. Extract Features
                features_df = extract_features(url_input)
                
                # 3. Predict
                prediction = model.predict(features_df)
                probability = model.predict_proba(features_df)[0][1]
                
                # Display Results
                st.divider()
                if prediction[0] == 1:
                    st.error(f"⚠️ **Warning: This looks like a Phishing site!**")
                    st.metric("Risk Score", f"{probability*100:.1f}%")
                else:
                    st.success(f"✅ **This URL appears to be Safe.**")
                    st.metric("Risk Score", f"{probability*100:.1f}%")
                
                # Show Feature Breakdown
                with st.expander("View AI Feature Breakdown"):
                    st.write("The AI analyzed the following characteristics:")
                    st.dataframe(features_df)
        else:
            st.warning("Please enter a URL first.")

    st.sidebar.info(
        "### How it works\n"
        "This app uses a **Random Forest Classifier** to analyze URL structure. "
        "It looks for common phishing traits like unusual length, suspicious symbols (@), "
        "and IP-based hosting."
    )

if __name__ == "__main__":
    main()
