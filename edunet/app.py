import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score

# Page config
st.set_page_config(page_title="Phoenix Fertility Engine", layout="centered")

# Theme and language toggles
theme = st.sidebar.radio("🌗 Choose Theme", ["Dark", "Light"])
language = st.sidebar.selectbox("🌐 Language", ["English", "தமிழ்"])

# Translations
translations = {
    "English": {
        "title": "Phoenix Fertility Engine",
        "objective": "Objective",
        "problem": "Problem Statement",
        "algorithm": "Algorithm Used",
        "start": "Start Prediction",
        "recommend": "Fertilizer Recommendations",
        "sensor": "Sensor Metrics Overview",
        "adjust": "Adjust Chemical Values",
        "graph": "Interactive Metric Graph",
        "manual": "Manual Toxicity Check",
        "predict": "Predict Toxicity",
        "average": "Average of Inputs",
        "safe": "Fertilizer is GOOD for plants.",
        "bad": "Fertilizer is BAD for plants.",
        "model": "Model Used",
        "accuracy": "Accuracy",
        "precision": "Precision"
    },
    "தமிழ்": {
        "title": "பீனிக்ஸ் உரம் இயந்திரம்",
        "objective": "நோக்கம்",
        "problem": "சிக்கல் விளக்கம்",
        "algorithm": "பயன்படுத்தப்படும் الگورிதம்",
        "start": "முன்னறிதலை தொடங்கு",
        "recommend": "உர பரிந்துரைகள்",
        "sensor": "சென்சார் அளவீட்டு மேடைகள்",
        "adjust": "வேதியியல் மதிப்புகளை மாற்றவும்",
        "graph": "மெட்ரிக் வரைபடம்",
        "manual": "கைமுறை நச்சுத்தன்மை கணிப்பு",
        "predict": "நச்சுத்தன்மையை கணிக்கவும்",
        "average": "உள்ளீடுகளின் சராசரி",
        "safe": "உரம் செடிகளுக்கு நல்லது.",
        "bad": "உரம் செடிகளுக்கு தீங்கு விளைவிக்கிறது.",
        "model": "பயன்படுத்தப்பட்ட மாதிரி",
        "accuracy": "துல்லியம்",
        "precision": "நிகர்த்தன்மை"
    }
}
t = translations[language]

# Styling
bg_color = "#1e1e1e" if theme == "Dark" else "#f5f5f5"
text_color = "#ffffff" if theme == "Dark" else "#000000"
accent = "#00ff88" if theme == "Dark" else "#ff6600"

st.markdown(f"""
    <style>
    body, .stApp {{ background-color: {bg_color}; color: {text_color}; }}
    .circle {{ width: 80px; height: 80px; border-radius: 50%; margin: 20px auto; }}
    .green {{ background-color: #00ff88; box-shadow: 0 0 25px #00ff88; }}
    .red {{ background-color: #ff4444; box-shadow: 0 0 25px #ff4444; }}
    .fade-in {{ animation: fadeIn 1s ease-in; }}
    @keyframes fadeIn {{ from {{opacity: 0;}} to {{opacity: 1;}} }}
    .mythic-panel {{ background-color: rgba(255,255,255,0.05); border: 1px solid {accent}; border-radius: 10px; padding: 15px; margin-bottom: 20px; }}
    .phoenix-logo {{ animation: pulse 2s infinite; margin: auto; display: block; }}
    @keyframes pulse {{ 0% {{ transform: scale(1); }} 50% {{ transform: scale(1.05); filter: drop-shadow(0 0 10px {accent}); }} 100% {{ transform: scale(1); }} }}
    </style>
""", unsafe_allow_html=True)

# Navigation
st.sidebar.markdown("### 🔀 Navigate")
for label, page in [("🏠 Home", "home"), ("📡 Sensor Metrics", "sensor"), ("🎛️ Adjust Values", "adjust"), ("📊 Show Graph", "graph"), ("🧪 Manual Toxicity Check", "manual")]:
    if st.sidebar.button(label):
        st.session_state.page = page
if "page" not in st.session_state:
    st.session_state.page = "home"

# Load data
data = pd.read_csv(r"edunet/fertilizer_ph_data.csv")
le = LabelEncoder()
data['Toxicity'] = le.fit_transform(data['Toxicity'])
X = data.drop('Toxicity', axis=1)
y = data['Toxicity']
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
y_pred = model.predict(X)
accuracy = round(accuracy_score(y, y_pred) * 100, 2)
precision = round(precision_score(y, y_pred, average='macro') * 100, 2)
default = X.iloc[0]
fertilizer_image = "edunet/fert.jpg"

# Recommendation logic
def recommend_fertilizer(pH, N, P, K, OM, SM, PMR, PHI):
    recs = []
    if pH < 5.5: recs.append("🧪 Add lime to reduce acidity.")
    elif pH > 7.5: recs.append("🧪 Add sulfur or compost to lower alkalinity.")
    if N < 1.5: recs.append("🌬️ Use urea or ammonium sulfate.")
    if P < 1.0: recs.append("🔥 Apply single super phosphate.")
    if K < 1.5: recs.append("🪨 Use muriate of potash or composted banana peels.")
    if OM < 3.0: recs.append("🌿 Add organic manure or vermicompost.")
    if SM < 40: recs.append("💧 Improve irrigation or add mulch.")
    if PMR < 75: recs.append("⚔️ Use neem-based biopesticides.")
    if PHI < 80: recs.append("🌟 Apply balanced NPK and monitor stress.")
    return recs

# Prediction block
def show_prediction_block(values):
    sample = [values]
    prediction = model.predict(sample)
    result = le.inverse_transform(prediction)[0]
    average = round(sum(values) / len(values), 2)
    st.markdown(f"**📊 {t['average']}:** `{average}`")
    if result == "Safe":
        st.markdown('<div class="circle green"></div>', unsafe_allow_html=True)
        st.success(f"✅ {t['safe']}")
    else:
        st.markdown('<div class="circle red"></div>', unsafe_allow_html=True)
        st.error(f"❌ {t['bad']}")
    st.markdown(f"**{t['model']}:** Random Forest Classifier")
    st.markdown(f"**{t['accuracy']}:** `{accuracy}%`  |  **{t['precision']}:** `{precision}%`")
    recs = recommend_fertilizer(*values)
    st.markdown(f"### 🌿 {t['recommend']}:")
    for r in recs:
        st.markdown(f"- {r}")

if st.session_state.page == "home":
    st.image("edunet/fert.jpg", width=120)
    st.markdown(f"<h1 style='text-align:center;'>{t['title']}</h1>", unsafe_allow_html=True)
    st.image(fertilizer_image, caption="Applying fertilizer to tomato plant 🌱", use_container_width=True)

    st.markdown(f"### 🧭 {t['objective']}")
    st.markdown("Design a machine learning system to predict fertilizer safety based on soil and chemical parameters.")
    st.markdown(f"### 🧪 {t['problem']}")
    st.markdown("Farmers struggle to find safe fertilizer mixes. This system helps predict toxicity and improve crop yield.")
    st.markdown(f"### 🧠 {t['algorithm']}")
    st.markdown("Random Forest Classifier — ensemble of decision trees with majority voting.")

    if st.button(f"🚀 {t['start']}"):
        st.session_state.page = "manual"

    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == "sensor":
    st.header(f"📡 {t['sensor']}")
    st.image(fertilizer_image, caption="Applying fertilizer to tomato plant 🌱", use_container_width=True)

    chemicals = {
        "🌊 pH": (default["pH"], 9.0),
        "🌬️ Nitrogen": (default["Nitrogen"], 5.0),
        "🔥 Phosphorus": (default["Phosphorus"], 5.0),
        "🪨 Potassium": (default["Potassium"], 5.0),
        "🌿 Organic Matter": (default["OrganicMatter"], 10.0),
        "💧 Soil Moisture": (default["SoilMoisture"], 100.0),
        "⚔️ Pest Mortality": (default["PestMortalityRate"], 100.0),
        "🌟 Plant Health": (default["PlantHealthIndex"], 100.0)
    }

    for chem, (value, max_val) in chemicals.items():
        fig = go.Figure(go.Pie(
            labels=[chem, 'Remaining'],
            values=[value, max_val - value],
            hole=0.5,
            marker=dict(colors=[accent, '#2e2e2e']),
            hoverinfo='label+percent',
            textinfo='value'
        ))
        fig.update_layout(title=f"{chem} Level", template="plotly_dark" if theme == "Dark" else "plotly_white", height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    if st.button(f"🔍 {t['predict']}"):
        values = [
            st.session_state.get("pH", default["pH"]),
            st.session_state.get("N", default["Nitrogen"]),
            st.session_state.get("P", default["Phosphorus"]),
            st.session_state.get("K", default["Potassium"]),
            st.session_state.get("OM", default["OrganicMatter"]),
            st.session_state.get("SM", default["SoilMoisture"]),
            st.session_state.get("PMR", default["PestMortalityRate"]),
            st.session_state.get("PHI", default["PlantHealthIndex"])
        ]
        show_prediction_block(values)

    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == "adjust":
    st.header(f"🎛️ {t['adjust']}")
    st.image(fertilizer_image, caption="Applying fertilizer to tomato plant 🌱", use_container_width=True)

    st.session_state.pH = st.slider("🌊 pH", 3.0, 9.0, float(default["pH"]))
    st.session_state.N = st.slider("🌬️ Nitrogen", 0.0, 5.0, float(default["Nitrogen"]))
    st.session_state.P = st.slider("🔥 Phosphorus", 0.0, 5.0, float(default["Phosphorus"]))
    st.session_state.K = st.slider("🪨 Potassium", 0.0, 5.0, float(default["Potassium"]))
    st.session_state.OM = st.slider("🌿 Organic Matter", 0.0, 10.0, float(default["OrganicMatter"]))
    st.session_state.SM = st.slider("💧 Soil Moisture (%)", 0, 100, int(default["SoilMoisture"]))
    st.session_state.PMR = st.slider("⚔️ Pest Mortality Rate (%)", 0, 100, int(default["PestMortalityRate"]))
    st.session_state.PHI = st.slider("🌟 Plant Health Index", 0, 100, int(default["PlantHealthIndex"]))

    if st.button(f"🔍 {t['predict']}"):
        values = [
            st.session_state.pH,
            st.session_state.N,
            st.session_state.P,
            st.session_state.K,
            st.session_state.OM,
            st.session_state.SM,
            st.session_state.PMR,
            st.session_state.PHI
        ]
        show_prediction_block(values)

    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == "graph":
    st.header(f"📊 {t['graph']}")
    st.image(fertilizer_image, caption="Applying fertilizer to tomato plant 🌱", use_container_width=True)

    fig = go.Figure(data=[
        go.Bar(
            x=[
                '🌊 pH', '🌬️ Nitrogen', '🔥 Phosphorus', '🪨 Potassium',
                '🌿 Organic Matter', '💧 Soil Moisture', '⚔️ Pest Mortality', '🌟 Plant Health'
            ],
            y=[
                st.session_state.get("pH", default["pH"]),
                st.session_state.get("N", default["Nitrogen"]),
                st.session_state.get("P", default["Phosphorus"]),
                st.session_state.get("K", default["Potassium"]),
                st.session_state.get("OM", default["OrganicMatter"]),
                st.session_state.get("SM", default["SoilMoisture"]),
                st.session_state.get("PMR", default["PestMortalityRate"]),
                st.session_state.get("PHI", default["PlantHealthIndex"])
            ],
            marker_color=[accent] * 8,
            hovertemplate='%{x}: %{y}<extra></extra>'
        )
    ])
    fig.update_layout(template="plotly_dark" if theme == "Dark" else "plotly_white", yaxis=dict(range=[0, 100]), height=450)
    st.plotly_chart(fig, use_container_width=True)

    if st.button(f"🔍 {t['predict']}"):
        values = [
            st.session_state.get("pH", default["pH"]),
            st.session_state.get("N", default["Nitrogen"]),
            st.session_state.get("P", default["Phosphorus"]),
            st.session_state.get("K", default["Potassium"]),
            st.session_state.get("OM", default["OrganicMatter"]),
            st.session_state.get("SM", default["SoilMoisture"]),
            st.session_state.get("PMR", default["PestMortalityRate"]),
            st.session_state.get("PHI", default["PlantHealthIndex"])
        ]
        show_prediction_block(values)

    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == "manual":
    st.header(f"🧪 {t['manual']}")
    st.image(fertilizer_image, caption="Applying fertilizer to tomato plant 🌱", use_container_width=True)

    manual_pH = st.number_input("🌊 pH", 3.0, 9.0, value=6.5)
    manual_N = st.number_input("🌬️ Nitrogen", 0.0, 5.0, value=2.5)
    manual_P = st.number_input("🔥 Phosphorus", 0.0, 5.0, value=2.0)
    manual_K = st.number_input("🪨 Potassium", 0.0, 5.0, value=2.5)
    manual_OM = st.number_input("🌿 Organic Matter", 0.0, 10.0, value=5.0)
    manual_SM = st.number_input("💧 Soil Moisture (%)", 0, 100, value=60)
    manual_PMR = st.number_input("⚔️ Pest Mortality Rate (%)", 0, 100, value=80)
    manual_PHI = st.number_input("🌟 Plant Health Index", 0, 100, value=85)

    if st.button(f"🔍 {t['predict']}"):
        values = [
            manual_pH, manual_N, manual_P, manual_K,
            manual_OM, manual_SM, manual_PMR, manual_PHI
        ]
        show_prediction_block(values)


    st.markdown('</div>', unsafe_allow_html=True)

