import streamlit as st
import re
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import google.generativeai as genai
import os
import logging

from google.generativeai.types import GenerationConfigDict
from google.api_core.exceptions import InvalidArgument, ResourceExhausted

# -------------------------------------------------------------------
# Inline CSS (inspired by Streamly layout and minimal design approach)
# -------------------------------------------------------------------
custom_css = """
<style>
/* GENERAL APP CONTAINER */
body {
    margin: 0;
    padding: 0;
    background-color: #F8FBFD; /* Light background akin to Streamly */
    font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
}
.main, .stApp {
    background-color: #F8FBFD !important;
    color: #333;
}

/* HEADER STYLING */
.app-title {
    font-weight: 900;
    font-size: 2.4rem;
    color: #2c3e50;
    margin-bottom: 0.3rem;
}
.app-subtitle {
    font-weight: 400;
    font-size: 1.1rem;
    color: #8898aa;
}
.logo-container {
    font-size: 2.4rem;
    font-weight: 900;
    text-align: center;
    color: #2c3e50;
    margin-bottom: 0.5rem;
}

/* SIDEBAR STYLING */
.block-container {
    padding-top: 1rem !important;
}
.sidebar .sidebar-content {
    background-color: #ffffff !important;
}
.sidebar-header {
    font-weight: 700;
    font-size: 1.4rem;
    margin-bottom: 0.5rem;
    color: #2c3e50;
}
.sidebar-subheader {
    font-weight: 600;
    font-size: 1rem;
    margin-top: 1rem;
    margin-bottom: 0.3rem;
    color: #2c3e50;
}
.sidebar-about {
    font-size: 0.9rem;
    color: #444;
    margin-bottom: 1rem;
}
.sidebar-footer {
    margin-top: 2rem;
    font-size: 0.8rem;
    color: #666;
}

/* BUTTON STYLING */
.stButton button {
    background-color: #3182CE !important;
    color: #fff !important;
    border-radius: 6px;
    border: none;
    font-weight: 500;
    transition: background-color 0.2s ease;
}
.stButton button:hover {
    background-color: #2b6cb0 !important;
}

/* FEATURE CARDS */
.feature-card {
    background-color: #ffffff;
    border-radius: 8px;
    border: 1px solid #e2e8f0;
    padding: 1.2rem;
    margin-bottom: 1.5rem;
    text-align: center;
}
.feature-icon {
    font-size: 2rem;
    margin-bottom: 0.5rem;
}
.feature-title {
    font-weight: 700;
    font-size: 1rem;
    margin-bottom: 0.2rem;
    color: #2c3e50;
}
.feature-description {
    font-size: 0.9rem;
    color: #555;
}

/* ITEM CARD */
.item-card {
    display: flex;
    align-items: center;
    background-color: #ffffff;
    border-radius: 8px;
    border: 1px solid #e2e8f0;
    padding: 1rem;
    margin-bottom: 1rem;
    width: 100%;
}
.item-icon {
    font-size: 2rem;
    margin-right: 1rem;
}
.item-name {
    font-weight: 600;
    font-size: 1.2rem;
    color: #2c3e50;
}
.item-cost {
    margin-left: auto;
    font-weight: 600;
    font-size: 1.1rem;
    color: #2c3e50;
}

/* FACTOR CARDS */
.factor-card {
    background-color: #ffffff;
    border-radius: 8px;
    border: 1px solid #e2e8f0;
    padding: 1rem;
    margin-bottom: 1rem;
    display: flex;
    flex-direction: row;
    align-items: center;
}
.factor-letter {
    width: 40px;
    height: 40px;
    background-color: #3182CE;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #fff;
    font-weight: 700;
    font-size: 1.2rem;
    margin-right: 1rem;
}
.factor-description {
    font-weight: 600;
    font-size: 0.95rem;
    margin-bottom: 0.2rem;
    color: #2c3e50;
}
.factor-explanation {
    font-size: 0.85rem;
    color: #555;
}
.factor-value {
    margin-left: auto;
    font-weight: 600;
    font-size: 1rem;
}
.factor-value.positive { color: #48bb78; }
.factor-value.neutral { color: #ed8936; }
.factor-value.negative { color: #f56565; }

/* RECOMMENDATION BOX */
.recommendation-box {
    background-color: #ffffff;
    border-radius: 8px;
    border: 1px solid #e2e8f0;
    padding: 1rem;
    margin-bottom: 1rem;
    display: flex;
    flex-direction: row;
    align-items: center;
    justify-content: space-between;
}
.recommendation-score {
    text-align: left;
}
.score-label {
    font-size: 0.8rem;
    color: #8898aa;
    margin-bottom: 0.2rem;
}
.score-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #2c3e50;
}
.recommendation-text {
    display: flex;
    align-items: center;
    font-weight: 600;
    color: #2c3e50;
}
.recommendation-icon {
    font-size: 1.5rem;
    margin-right: 0.5rem;
}
.positive .recommendation-icon { color: #48bb78; }
.negative .recommendation-icon { color: #f56565; }
.neutral .recommendation-icon { color: #ed8936; }

/* INSIGHTS SECTION */
.insights-header {
    font-weight: 600;
    font-size: 1rem;
    margin-top: 1rem;
    margin-bottom: 0.5rem;
    color: #2c3e50;
}
.insight-item {
    font-size: 0.9rem;
    color: #555;
    margin-bottom: 0.2rem;
}

/* SAMPLE CARDS */
.sample-card {
    background-color: #ffffff;
    border-radius: 8px;
    border: 1px solid #e2e8f0;
    padding: 1rem;
    margin-bottom: 1.5rem;
}
.sample-title {
    font-weight: 600;
    font-size: 1rem;
    margin-bottom: 0.3rem;
    color: #2c3e50;
}
.sample-recommendation {
    font-size: 0.9rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}
.sample-recommendation.negative {
    color: #f56565;
}
.sample-recommendation.positive {
    color: #48bb78;
}
.sample-insight {
    font-size: 0.85rem;
    color: #555;
}

/* PAGE TITLES */
.page-title {
    font-size: 1.5rem;
    font-weight: 700;
    margin: 1rem 0 1rem;
    color: #2c3e50;
}

/* FORM SECTIONS */
.form-section {
    font-weight: 600;
    font-size: 1rem;
    margin-top: 1rem;
    margin-bottom: 0.5rem;
    color: #2c3e50;
}

/* CTA STYLING */
.cta-container {
    margin-top:1.5rem;
    margin-bottom:1rem;
    text-align:center;
}
.cta-text {
    font-weight:600; 
    font-size:1.2rem; 
    color: #2c3e50;
}
</style>
"""

# -------------------------------------------------------------------
# Apply the custom CSS to the app
# -------------------------------------------------------------------
st.markdown(custom_css, unsafe_allow_html=True)


# -------------------------------------------------------------------
# STREAMLIT PAGE CONFIG
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Munger AI - Should You Buy It?",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": "https://github.com/yourusername/munger-ai",
        "Report a bug": "https://github.com/yourusername/munger-ai/issues",
        "About": """
            ## Munger AI - Purchase Decision Assistant
            
            Powered by Google's Gemini AI, this tool helps you make smarter 
            purchase decisions using Charlie Munger's mental models 
            and AI-powered analysis.
            
            **GitHub**: https://github.com/yourusername/munger-ai
        """
    }
)

# -------------------------------------------------------------------
# SET YOUR GOOGLE GENERATIVE AI KEY
# -------------------------------------------------------------------
API_KEY = os.environ.get("GOOGLE_API_KEY", "my_test_api_key_123")  # test key as fallback
if not API_KEY:
    st.error("Please set your Google API key in your environment or secrets.")
    st.stop()

try:
    genai.configure(api_key=API_KEY)
except Exception as e:
    st.error(f"Failed to configure Google Generative AI: {str(e)}")
    st.stop()

# -------------------------------------------------------------------
# LOGGING
# -------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)

# -------------------------------------------------------------------
# UI HELPER FUNCTIONS
# -------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def display_header():
    """Display the app header with logo and title."""
    col1, col2 = st.columns([1, 5])
    with col1:
        st.markdown('<div class="logo-container">M</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<h1 class="app-title">Munger AI</h1>', unsafe_allow_html=True)
        st.markdown(
            '<p class="app-subtitle">Should you buy it? Get AI-powered purchase advice in seconds.</p>',
            unsafe_allow_html=True
        )


@st.cache_data(show_spinner=False)
def create_radar_chart(factors):
    """Create a radar chart for the factor analysis."""
    categories = ["Discretionary Income", "Opportunity Cost", "Goal Alignment",
                  "Long-Term Impact", "Behavioral"]
    vals = [factors["D"], factors["O"], factors["G"], factors["L"], factors["B"]]
    # Close the radar shape
    vals.append(vals[0])
    categories.append(categories[0])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals,
        theta=categories,
        fill='toself',
        fillcolor='rgba(49, 130, 206, 0.2)',
        line=dict(color='#3182CE', width=2),
        name='Factors'
    ))

    # Reference lines
    for i in [-2, -1, 0, 1, 2]:
        fig.add_trace(go.Scatterpolar(
            r=[i]*len(categories),
            theta=categories,
            line=dict(color='rgba(200,200,200,0.5)', width=1, dash='dash'),
            showlegend=False
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[-3, 3],
                tickvals=[-2, -1, 0, 1, 2],
                gridcolor='rgba(200,200,200,0.3)'
            ),
            angularaxis=dict(gridcolor='rgba(200,200,200,0.3)'),
            bgcolor='rgba(255,255,255,0.9)'
        ),
        showlegend=False,
        margin=dict(l=60, r=60, t=20, b=20),
        height=350,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


@st.cache_data(show_spinner=False)
def create_pds_gauge(pds):
    """Create a gauge chart for the PDS score."""
    if pds >= 5:
        color = "#48bb78"  # Green
    elif pds < 0:
        color = "#f56565"  # Red
    else:
        color = "#ed8936"  # Orange

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pds,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [-10, 10]},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e2e8f0",
            'steps': [
                {'range': [-10, 0], 'color': '#fed7d7'},
                {'range': [0, 5], 'color': '#feebc8'},
                {'range': [5, 10], 'color': '#c6f6d5'}
            ],
        }
    ))

    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "#2d3748", 'family': "Inter, sans-serif"}
    )
    return fig

# -------------------------------------------------------------------
# GEMINI SERVICE CLASS
# -------------------------------------------------------------------
class EnhancedGeminiService:
    """
    Enhanced Gemini service for purchase decision analysis with improved
    prompting and error handling.
    """

    def __init__(self, model_name="gemini-1.5-pro"):
        self.model_name = model_name
        try:
            self.model = genai.GenerativeModel(model_name)
        except Exception as e:
            logging.error(f"Error initializing Gemini model: {str(e)}")
            st.error(f"Error initializing Gemini model: {str(e)}")
            self.model = None

    def analyze_purchase(self, context):
        """Analyze a purchase decision using enhanced prompting."""
        if not self.model:
            return self._create_fallback_response("Gemini model not initialized")

        prompt = self._create_enhanced_prompt(context)
        with st.spinner("Analyzing your purchase..."):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.2,
                        max_output_tokens=1024,
                        top_p=0.95,
                    )
                )
                if response:
                    return self._process_response(response.text)
                else:
                    return self._create_fallback_response("Empty response")
            except ResourceExhausted as e:
                st.warning("API rate limit exceeded. Please try again in a few moments.")
                return self._create_fallback_response(f"API rate limit: {str(e)}")
            except InvalidArgument as e:
                st.error(f"Invalid argument error: {str(e)}")
                return self._create_fallback_response(str(e))
            except Exception as e:
                st.error(f"Error calling Gemini API: {str(e)}")
                return self._create_fallback_response(str(e))

    def _create_enhanced_prompt(self, context):
        """Create an enhanced prompt with clear instructions."""
        item_name = context.get("item_name", "Unknown Item")
        item_cost = context.get("item_cost", 0)
        leftover_income = context.get("leftover_income", 0)
        has_debt = context.get("has_high_interest_debt", "Unknown")
        financial_goal = context.get("main_financial_goal", "")
        urgency = context.get("purchase_urgency", "Mixed")
        extra_context = context.get("extra_context", "")

        # Ratio
        income_cost_ratio = leftover_income / max(1, item_cost)

        # Prompt
        prompt = f"""
        You are a financial decision assistant based on Charlie Munger's mental models.

        # PURCHASE CONTEXT
        - Item Name: {item_name}
        - Item Cost: ${item_cost:.2f}
        - Monthly Leftover Income: ${leftover_income:.2f}
        - Income-to-Cost Ratio: {income_cost_ratio:.2f}
        - Has High-Interest Debt: {has_debt}
        - Financial Goal: {financial_goal}
        - Purchase Urgency: {urgency}
        - Additional Context: {extra_context}

        # TASK
        Evaluate this purchase using our 5-factor Purchase Decision Score (PDS) system:
        PDS = D + O + G + L + B, where each factor ranges from -2 to +2.

        # EVALUATION FRAMEWORK
        For each factor, provide:
        1. A score from -2 to +2
        2. A brief but specific explanation for the score

        ## D: DISCRETIONARY INCOME FACTOR
        - +2: Excellent affordability (cost < 10% of monthly leftover income)
        - +1: Good affordability (cost 10-25%)
        - 0: Moderate (25-50%)
        - -1: Challenging (50-100%)
        - -2: Poor (>100%)

        ## O: OPPORTUNITY COST FACTOR
        - +2: Exceptional value vs. alternatives
        - +1: Good value
        - 0: Equivalent
        - -1: Better alternatives exist
        - -2: Significantly better alternatives exist

        ## G: GOAL ALIGNMENT FACTOR
        - +2: Directly accelerates primary financial goal
        - +1: Somewhat supports
        - 0: Neutral
        - -1: Slightly delays
        - -2: Directly contradicts

        ## L: LONG-TERM IMPACT FACTOR
        - +2: Delivers long-term value or income
        - +1: Durable
        - 0: Average lifespan
        - -1: Rapid depreciation or ongoing costs
        - -2: Minimal long-term value

        ## B: BEHAVIORAL FACTOR
        - +2: Essential need with immediate utility
        - +1: Practical need
        - 0: Mix of need and want
        - -1: Primarily want-based
        - -2: Pure impulse purchase

        # OUTPUT FORMAT
        Return a valid JSON object with this exact structure:
        {{
          "D": score,
          "O": score,
          "G": score,
          "L": score,
          "B": score,
          "D_explanation": "Your explanation here",
          "O_explanation": "Your explanation here",
          "G_explanation": "Your explanation here",
          "L_explanation": "Your explanation here",
          "B_explanation": "Your explanation here",
          "insights": ["Insight 1", "Insight 2"]
        }}

        Return ONLY the JSON object, no other text.
        """
        return prompt

    def _process_response(self, text):
        """Process JSON from Gemini with robust fallback parsing."""
        logging.info(f"Processing Gemini response: {text[:80]}...")

        # 1. Attempt direct JSON
        try:
            data = json.loads(text)
            if self._validate_factors(data):
                return data
        except json.JSONDecodeError:
            pass

        # 2. Attempt regex-based extraction
        json_matches = re.findall(r"(\{[\s\S]*?\})", text)
        for json_str in json_matches:
            try:
                data = json.loads(json_str)
                if self._validate_factors(data):
                    return data
            except json.JSONDecodeError:
                continue

        # 3. Fallback
        return self._extract_structured_data(text)

    def _validate_factors(self, data):
        """Validate factor presence and ranges."""
        required_factors = ["D", "O", "G", "L", "B"]
        if not all(f in data for f in required_factors):
            return False
        for f in required_factors:
            val = data[f]
            if not isinstance(val, int) or val < -2 or val > 2:
                return False
        return True

    def _extract_structured_data(self, text):
        """Fallback extraction if no valid JSON found."""
        logging.info("Falling back to structured data extraction.")
        result = {
            "D": 0, "O": 0, "G": 0, "L": 0, "B": 0,
            "D_explanation": "",
            "O_explanation": "",
            "G_explanation": "",
            "L_explanation": "",
            "B_explanation": "",
            "insights": []
        }

        # Factor patterns
        for f in ["D", "O", "G", "L", "B"]:
            match = re.search(rf"{f}:\s*([+-]?\d+)", text)
            if match:
                try:
                    val = int(match.group(1))
                    result[f] = max(-2, min(2, val))
                except:
                    pass

        # Explanations
        for f in ["D", "O", "G", "L", "B"]:
            expl_pattern = rf"{f}_explanation['\"]?\s*:\s*['\"]([^\"']+)"
            match = re.search(expl_pattern, text)
            if match:
                result[f"{f}_explanation"] = match.group(1).strip()

        return result

    def _create_fallback_response(self, msg):
        logging.error(f"Analysis error: {msg}")
        return {
            "D": 0, "O": 0, "G": 0, "L": 0, "B": 0,
            "D_explanation": "Unable to analyze.",
            "O_explanation": "Unable to analyze.",
            "G_explanation": "Unable to analyze.",
            "L_explanation": "Unable to analyze.",
            "B_explanation": "Unable to analyze.",
            "insights": [
                "Consider your budget constraints first.",
                "Technical error occurred."
            ],
            "error": msg
        }

# -------------------------------------------------------------------
# UTILITY
# -------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def compute_pds(factors):
    return sum(factors.get(f, 0) for f in ["D", "O", "G", "L", "B"])

def get_recommendation(pds):
    if pds >= 5:
        return "Buy it.", "positive"
    elif pds < 0:
        return "Don't buy it.", "negative"
    else:
        return "Consider carefully.", "neutral"

def render_factor_card(factor, value, description, explanation):
    if value > 0:
        val_class = "positive"
        icon = "↑"
    elif value < 0:
        val_class = "negative"
        icon = "↓"
    else:
        val_class = "neutral"
        icon = "→"

    html = f"""
    <div class="factor-card">
        <div class="factor-letter">{factor}</div>
        <div class="factor-info">
            <div class="factor-description">{description}</div>
            <div class="factor-explanation">{explanation}</div>
        </div>
        <div class="factor-value {val_class}">{icon} {value:+d}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_item_card(item_name, cost):
    icon = "💼" if cost >= 1000 else "🛍️"
    html = f"""
    <div class="item-card">
        <div class="item-icon">{icon}</div>
        <div class="item-details">
            <div class="item-name">{item_name}</div>
        </div>
        <div class="item-cost">${cost:,.2f}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_recommendation_box(pds, recommendation, rec_class):
    icon = "✅" if rec_class == "positive" else "❌" if rec_class == "negative" else "⚠️"
    html = f"""
    <div class="recommendation-box {rec_class}">
        <div class="recommendation-score">
            <div class="score-label">PURCHASE DECISION SCORE</div>
            <div class="score-value">{pds}</div>
        </div>
        <div class="recommendation-text">
            <div class="recommendation-icon">{icon}</div>
            <div class="recommendation-message">{recommendation}</div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def display_insights(insights):
    st.markdown('<div class="insights-header">INSIGHTS</div>', unsafe_allow_html=True)
    for insight in insights:
        st.markdown(f'<div class="insight-item">💡 {insight}</div>', unsafe_allow_html=True)

# Factor descriptions
factor_descriptions = {
    "D": "Discretionary Income",
    "O": "Opportunity Cost",
    "G": "Goal Alignment",
    "L": "Long-Term Impact",
    "B": "Behavioral"
}

# -------------------------------------------------------------------
# SESSION STATE INIT
# -------------------------------------------------------------------
def initialize_session_state():
    if 'page' not in st.session_state:
        st.session_state.page = 'home'
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'purchase_info' not in st.session_state:
        st.session_state.purchase_info = None

def display_analysis_results():
    if not st.session_state.analysis_results or not st.session_state.purchase_info:
        st.info("No analysis results available.")
        return

    factors = st.session_state.analysis_results
    purchase_info = st.session_state.purchase_info

    # PDS
    pds = compute_pds(factors)
    recommendation, rec_class = get_recommendation(pds)

    st.markdown("## Analysis Results")
    st.write("---")

    # Item card
    render_item_card(purchase_info["item_name"], purchase_info["item_cost"])

    # Recommendation
    render_recommendation_box(pds, recommendation, rec_class)

    # Factor breakdown
    st.markdown("### Factor Breakdown")
    col1, col2 = st.columns(2)
    with col1:
        render_factor_card("D", factors["D"], factor_descriptions["D"], factors["D_explanation"])
        render_factor_card("O", factors["O"], factor_descriptions["O"], factors["O_explanation"])
    with col2:
        render_factor_card("G", factors["G"], factor_descriptions["G"], factors["G_explanation"])
        render_factor_card("L", factors["L"], factor_descriptions["L"], factors["L_explanation"])

    render_factor_card("B", factors["B"], factor_descriptions["B"], factors["B_explanation"])

    # Radar
    st.markdown("### Factor Radar Chart")
    radar_fig = create_radar_chart(factors)
    st.plotly_chart(radar_fig, use_container_width=True)

    # Gauge
    st.markdown("### Purchase Decision Score Gauge")
    gauge_fig = create_pds_gauge(pds)
    st.plotly_chart(gauge_fig, use_container_width=True)

    # Insights
    if factors.get("insights"):
        display_insights(factors["insights"])


# -------------------------------------------------------------------
# SIDEBAR NAV
# -------------------------------------------------------------------
gemini_service = None

def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="sidebar-header">Munger AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-subheader">NAVIGATION</div>', unsafe_allow_html=True)

        if st.button("🏠 Home", key="nav_home"):
            st.session_state.page = 'home'
        if st.button("🔍 Decision Tool", key="nav_decision_tool"):
            st.session_state.page = 'decision_tool'
        if st.button("⚙️ Advanced Tool", key="nav_advanced_tool"):
            st.session_state.page = 'advanced_tool'

        st.markdown('<div class="sidebar-subheader">ABOUT</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="sidebar-about">
            Munger AI uses Charlie Munger's mental models to analyze purchase decisions. 
            It evaluates five key factors to give you personalized recommendations.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown('<div class="sidebar-subheader">FACTORS</div>', unsafe_allow_html=True)
        with st.expander("D: Discretionary Income"):
            st.markdown(" - +2: Cost < 10% leftover\n - -2: Cost > 100% leftover")
        with st.expander("O: Opportunity Cost"):
            st.markdown(" - +2: Exceptional value\n - -2: Much better alternatives")
        with st.expander("G: Goal Alignment"):
            st.markdown(" - +2: Directly supports goals\n - -2: Directly contradicts")
        with st.expander("L: Long-Term Impact"):
            st.markdown(" - +2: Long-term value or income\n - -2: Minimal or negative value")
        with st.expander("B: Behavioral Factor"):
            st.markdown(" - +2: Essential need\n - -2: Pure impulse purchase")

        st.markdown('<div class="sidebar-footer">© 2025 Munger AI</div>', unsafe_allow_html=True)


# -------------------------------------------------------------------
# PAGES
# -------------------------------------------------------------------
def render_home_page():
    display_header()
    st.markdown(
        """
        <div class="home-intro" style="background:#ffffff; padding:1rem; border-radius:8px; margin-bottom:2rem;">
            <div class="home-title" style="font-weight:700; font-size:1.5rem;">Make smarter purchase decisions</div>
            <div class="home-description" style="font-size:0.95rem; color:#444;">
                Powered by AI and inspired by Charlie Munger's mental models, 
                our tool helps you decide if a purchase is right for you.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Feature grid
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🧠</div>
                <div class="feature-title">AI-Powered Analysis</div>
                <div class="feature-description">
                    Get purchase recommendations backed by deep financial intelligence and mental models.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">📊</div>
                <div class="feature-title">5-Factor Evaluation</div>
                <div class="feature-description">
                    We analyze discretionary income, opportunity cost, goal alignment, long-term impact, and behavioral factors.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">⚡</div>
                <div class="feature-title">Instant Results</div>
                <div class="feature-description">
                    Get clear, actionable recommendations in seconds to help you make better decisions.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # CTA
    st.markdown(
        """
        <div class="cta-container">
            <div class="cta-text">Ready to make a purchase decision?</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    colA, colB = st.columns(2)
    with colA:
        if st.button("🛍️ Basic Analysis", use_container_width=True):
            st.session_state.page = 'decision_tool'
    with colB:
        if st.button("🔍 Detailed Analysis", use_container_width=True):
            st.session_state.page = 'advanced_tool'

    # Sample
    st.markdown('<div class="sample-header" style="margin-top:2rem; font-weight:600; font-size:1rem;">SAMPLE INSIGHTS</div>', unsafe_allow_html=True)
    colA, colB = st.columns(2)
    with colA:
        st.markdown(
            """
            <div class="sample-card">
                <div class="sample-title">New Smartphone ($1,200)</div>
                <div class="sample-recommendation negative">Don't buy it</div>
                <div class="sample-insight">
                    "With high-interest debt, paying it down would yield a better financial return than a new smartphone."
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with colB:
        st.markdown(
            """
            <div class="sample-card">
                <div class="sample-title">Professional Course ($800)</div>
                <div class="sample-recommendation positive">Buy it</div>
                <div class="sample-insight">
                    "This investment directly aligns with your career goals and has strong potential for positive ROI."
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


def render_decision_tool():
    display_header()
    st.markdown('<div class="page-title">Basic Purchase Decision Tool</div>', unsafe_allow_html=True)

    with st.form("basic_form"):
        col1, col2 = st.columns([3, 1])
        with col1:
            item_name = st.text_input("What are you buying?", value="New Laptop")
        with col2:
            cost = st.number_input("Cost ($)", min_value=1.0, value=1200.0, step=50.0)

        col1, col2 = st.columns(2)
        with col1:
            leftover_income = st.number_input("Monthly leftover income ($)", min_value=0.0, value=2000.0, step=100.0)
        with col2:
            has_debt = st.selectbox("Do you have high-interest debt?", ["No", "Yes"])

        main_goal = st.text_input("Your main financial goal", value="Build an emergency fund")
        col1, col2 = st.columns(2)
        with col1:
            urgency = st.selectbox("Purchase urgency", ["Need", "Mixed", "Want"])
        with col2:
            st.write("")  # filler

        submit_btn = st.form_submit_button("Analyze My Purchase", use_container_width=True)

    if submit_btn:
        purchase_info = {
            "item_name": item_name,
            "item_cost": cost,
            "leftover_income": leftover_income,
            "has_high_interest_debt": has_debt,
            "main_financial_goal": main_goal,
            "purchase_urgency": urgency,
            "extra_context": ""
        }
        st.session_state.purchase_info = purchase_info

        service = EnhancedGeminiService()
        factors = service.analyze_purchase(purchase_info)
        st.session_state.analysis_results = factors

    if st.session_state.analysis_results and st.session_state.purchase_info:
        display_analysis_results()


def render_advanced_tool():
    display_header()
    st.markdown('<div class="page-title">Advanced Purchase Decision Tool</div>', unsafe_allow_html=True)

    with st.form("advanced_form"):
        st.markdown('<div class="form-section">Purchase Details</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([3, 1])
        with col1:
            item_name = st.text_input("What are you buying?", value="High-End Laptop")
        with col2:
            cost = st.number_input("Cost ($)", min_value=1.0, value=2000.0, step=100.0)

        st.markdown('<div class="form-section">Financial Information</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            leftover_income = st.number_input("Monthly leftover income ($)", min_value=0.0, value=3000.0, step=100.0)
            st.caption("Income remaining after essential expenses")
        with col2:
            has_debt = st.selectbox("High-interest debt?", ["No", "Yes"])
            st.caption("Debt above ~8% APR")

        col1, col2 = st.columns(2)
        with col1:
            main_goal = st.text_input("Main financial goal", value="Save for a house down payment")
            st.caption("Your primary financial objective")
        with col2:
            urgency = st.selectbox("Purchase urgency", ["Need", "Mixed", "Want"])
            st.caption("How necessary is this purchase?")

        st.markdown('<div class="form-section">Additional Context</div>', unsafe_allow_html=True)
        extra_context = st.text_area("Any other details to consider?", height=100)
        st.caption("Add relevant considerations or constraints")

        submit_advanced = st.form_submit_button("Get Detailed Analysis", use_container_width=True)

    if submit_advanced:
        purchase_info = {
            "item_name": item_name,
            "item_cost": cost,
            "leftover_income": leftover_income,
            "has_high_interest_debt": has_debt,
            "main_financial_goal": main_goal,
            "purchase_urgency": urgency,
            "extra_context": extra_context
        }
        st.session_state.purchase_info = purchase_info

        service = EnhancedGeminiService()
        factors = service.analyze_purchase(purchase_info)
        st.session_state.analysis_results = factors

    if st.session_state.analysis_results and st.session_state.purchase_info:
        display_analysis_results()


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():
    initialize_session_state()

    global gemini_service
    if gemini_service is None:
        try:
            gemini_service = EnhancedGeminiService()
        except Exception as e:
            st.error(f"Failed to initialize Gemini: {str(e)}")

    render_sidebar()

    if st.session_state.page == 'home':
        render_home_page()
    elif st.session_state.page == 'decision_tool':
        render_decision_tool()
    elif st.session_state.page == 'advanced_tool':
        render_advanced_tool()
    else:
        render_home_page()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        logging.exception("Unexpected error in application:", exc_info=e)
