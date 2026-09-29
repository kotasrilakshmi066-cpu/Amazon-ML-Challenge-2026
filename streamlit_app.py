import streamlit as st
import pandas as pd
from rapidfuzz import fuzz
from xgboost import XGBClassifier

st.set_page_config(page_title="Entity Resolution AI", page_icon="🏢")

# Load the saved AI Brain safely
@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model("xgb_model.json")
    return model

model = load_model()
THRESHOLD = 0.65

st.title("🏢 Business Entity Resolution AI")
st.markdown("**Top 10% Solution - Amazon ML Challenge 2026**")
st.write("Type in two businesses below to see if my Machine Learning model thinks they are the same entity!")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Business 1")
    n1 = st.text_input("Business Name", placeholder="e.g. Amazon Inc", key="n1")
    a1 = st.text_input("Address", placeholder="e.g. 410 Terry Ave N, Seattle", key="a1")

with col2:
    st.subheader("Business 2")
    n2 = st.text_input("Business Name", placeholder="e.g. Amazon.com", key="n2")
    a2 = st.text_input("Address", placeholder="e.g. Terry Avenue North", key="a2")

if st.button("Compare Businesses", type="primary"):
    if n1 and n2:
        str_n1, str_n2 = str(n1).lower(), str(n2).lower()
        str_a1, str_a2 = str(a1).lower(), str(a2).lower()
        
        # Calculate features
        name_ratio = fuzz.ratio(str_n1, str_n2)
        addr_ratio = fuzz.ratio(str_a1, str_a2)
        name_set_ratio = fuzz.token_set_ratio(str_n1, str_n2)
        addr_set_ratio = fuzz.token_set_ratio(str_a1, str_a2)
        name_partial = fuzz.partial_ratio(str_n1, str_n2)
        
        features = pd.DataFrame([[name_ratio, addr_ratio, name_set_ratio, addr_set_ratio, name_partial]], 
                                columns=['name_ratio', 'addr_ratio', 'name_set_ratio', 'addr_set_ratio', 'name_partial'])
        
        prob = model.predict_proba(features)[0][1]
        is_match = prob > THRESHOLD
        
        st.divider()
        if is_match:
            st.success(f"### ✅ MATCH\n**AI Confidence:** {prob * 100:.2f}%")
        else:
            st.error(f"### ❌ NO MATCH\n**AI Confidence:** {prob * 100:.2f}%")
            
        st.markdown(f"""
        **Math Scores calculated by the AI:**
        - Name Match: `{name_ratio}%`
        - Name Subset Match: `{name_partial}%`
        - Address Match: `{addr_ratio}%`
        """)
    else:
        st.warning("Please enter at least the business names to compare!")
