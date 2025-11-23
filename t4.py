import streamlit as st
import string
import nltk
from nltk.corpus import stopwords
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import numpy as np
from collections import defaultdict
import io

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Smart Email Spam Detector", 
    page_icon="📧", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- SESSION STATE INITIALIZATION ----------------
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []
if 'stats' not in st.session_state:
    st.session_state.stats = {'total': 0, 'ham': 0, 'spam': 0, 'confidence_scores': []}
if 'recent_activity' not in st.session_state:
    st.session_state.recent_activity = []
if 'detection_trends' not in st.session_state:
    st.session_state.detection_trends = []

# ---------------- PREPROCESSING FUNCTION ----------------
def clean_text(text):
    """Cleans the input text by converting to lowercase, removing punctuation,
    and removing stopwords (same process as model training)."""
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()

    try:
        stop_words = stopwords.words('english')
    except LookupError:
        st.info("Downloading necessary NLTK data (stopwords)...")
        nltk.download('stopwords')
        stop_words = stopwords.words('english')
        st.success("Download complete. Please refresh if the app doesn't update.")

    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# ---------------- LOAD MODEL & VECTORIZER ----------------
@st.cache_resource
def load_model_and_vectorizer():
    try:
        model = joblib.load("spam_detector_model.joblib")
        vectorizer = joblib.load("tfidf_vectorizer.joblib")
        return model, vectorizer
    except FileNotFoundError:
        st.error("⚠️ Model or vectorizer not found! Please keep "
                 "`spam_detector_model.joblib` & `tfidf_vectorizer.joblib` "
                 "in the same folder as this script.")
        return None, None

model, vectorizer = load_model_and_vectorizer()

# ---------------- ADVANCED FEATURES ----------------
def get_prediction_with_confidence(message):
    """Get prediction with confidence score"""
    cleaned_message = clean_text(message)
    message_vector = vectorizer.transform([cleaned_message])
    
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(message_vector)[0]
        prediction = model.predict(message_vector)[0]
        confidence = max(probabilities)
    else:
        prediction = model.predict(message_vector)[0]
        confidence = 0.8  # Default confidence for models without probability
    
    return prediction, confidence

def update_stats(prediction, confidence):
    """Update statistics and history"""
    timestamp = datetime.now()
    
    # Update stats
    st.session_state.stats['total'] += 1
    if prediction == 'spam':
        st.session_state.stats['spam'] += 1
    else:
        st.session_state.stats['ham'] += 1
    st.session_state.stats['confidence_scores'].append(confidence)
    
    # Update history
    st.session_state.detection_history.append({
        'timestamp': timestamp,
        'prediction': prediction,
        'confidence': confidence,
        'type': 'manual_input'
    })
    
    # Update trends for line chart
    st.session_state.detection_trends.append({
        'timestamp': timestamp,
        'ham_count': st.session_state.stats['ham'],
        'spam_count': st.session_state.stats['spam'],
        'total': st.session_state.stats['total']
    })
    
    # Update recent activity (keep last 10)
    st.session_state.recent_activity.insert(0, {
        'time': timestamp.strftime("%H:%M:%S"),
        'date': timestamp.strftime("%Y-%m-%d"),
        'type': prediction.upper(),
        'confidence': f"{confidence:.1%}"
    })
    st.session_state.recent_activity = st.session_state.recent_activity[:10]

def analyze_text_features(text):
    """Analyze text features that might indicate spam"""
    features = {
        'length': len(text),
        'word_count': len(text.split()),
        'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(1, len(text)),
        'exclamation_count': text.count('!'),
        'dollar_sign_count': text.count('$'),
        'url_keywords': any(keyword in text.lower() for keyword in ['http', 'www', '.com', 'click']),
        'spam_keywords': any(keyword in text.lower() for keyword in 
                           ['free', 'winner', 'prize', 'urgent', 'money', 'cash', 'guaranteed'])
    }
    return features

# ---------------- ENHANCED CSS ----------------
st.markdown("""
<style>
    /* App Background */
    .stApp {
        background: linear-gradient(135deg, #f0f4f9 0%, #dbeafe 50%, #f0f4f9 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Headers */
    h1, h2, h3 {
        color: #1e3a8a;
        font-weight: 700;
        text-align: center;
    }

    /* Main Container */
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    /* Button Styling */
    .stButton>button {
        background: linear-gradient(90deg, #1e3a8a, #2563eb);
        color: white;
        border-radius: 40px;
        padding: 0.7rem 1.5rem;
        font-weight: 600;
        border: none;
        transition: 0.3s;
        width: 100%;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #2563eb, #1d4ed8);
        transform: scale(1.05);
        box-shadow: 0 8px 25px rgba(37, 99, 235, 0.3);
    }

    /* Text Area */
    .stTextArea textarea {
        border-radius: 15px;
        border: 2px solid #93c5fd;
        background-color: #f9fafb;
        padding: 15px;
        font-size: 14px;
        transition: all 0.3s ease;
    }
    .stTextArea textarea:focus {
        border-color: #2563eb;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
    }

    /* Result Cards */
    .result-box {
        padding: 2rem;
        border-radius: 20px;
        font-weight: bold;
        text-align: center;
        font-size: 1.3rem;
        margin: 1rem 0;
        box-shadow: 0px 8px 25px rgba(0,0,0,0.15);
        transition: transform 0.3s ease;
    }
    .result-box:hover {
        transform: translateY(-5px);
    }
    .spam {
        background: linear-gradient(135deg, #dc2626, #ef4444);
        color: white;
    }
    .ham {
        background: linear-gradient(135deg, #16a34a, #22c55e);
        color: white;
    }

    /* Stats Cards */
    .stats-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
    }

    /* Confidence Meter */
    .confidence-meter {
        background: linear-gradient(90deg, #dc2626, #f59e0b, #16a34a);
        height: 10px;
        border-radius: 10px;
        margin: 1rem 0;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
    }

    /* Tabs */
    .stTabs [role="tablist"] {
        gap: 1rem;
        justify-content: center;
    }
    .stTabs [role="tab"] {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 10px 10px 0 0;
        padding: 0.5rem 1.5rem;
        border: 1px solid #e5e7eb;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1e3a8a, #2563eb);
        color: white;
    }

    /* Feature Analysis Cards */
    .feature-card {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        border-left: 4px solid #2563eb;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    /* Activity Timeline */
    .activity-item {
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 8px;
        background: rgba(37, 99, 235, 0.05);
        border-left: 3px solid #2563eb;
    }
    
    /* Horizontal Stats */
    .horizontal-stats {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        margin: 1rem 0;
    }
    .stat-item {
        flex: 1;
        text-align: center;
        padding: 1rem;
        background: rgba(255, 255, 255, 0.8);
        border-radius: 10px;
        border: 1px solid #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- APP HEADER ----------------
st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <h1 style="font-size: 3rem; margin-bottom: 0.5rem;">📧 Smart Email Spam Detector</h1>
    <h3 style="color: #4b5563; margin-top: 0;">by Yuvraj Kumar Gond</h3>
    <p style="font-size: 1.2rem; color: #374151; max-width: 800px; margin: 0 auto;">
        Advanced AI-powered email classification with real-time analytics, trend analysis, 
        and comprehensive spam detection features 🚀
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------- MAIN APP ----------------
if model and vectorizer:
    # Quick Stats Overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stats-card">
            <h3>📊 Total</h3>
            <h2>{st.session_state.stats['total']}</h2>
            <p>Messages Analyzed</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stats-card" style="background: linear-gradient(135deg, #16a34a, #22c55e);">
            <h3>✅ Ham</h3>
            <h2>{st.session_state.stats['ham']}</h2>
            <p>Legitimate Emails</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stats-card" style="background: linear-gradient(135deg, #dc2626, #ef4444);">
            <h3>🚨 Spam</h3>
            <h2>{st.session_state.stats['spam']}</h2>
            <p>Spam Detected</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_confidence = np.mean(st.session_state.stats['confidence_scores']) if st.session_state.stats['confidence_scores'] else 0
        st.markdown(f"""
        <div class="stats-card" style="background: linear-gradient(135deg, #f59e0b, #eab308);">
            <h3>🎯 Accuracy</h3>
            <h2>{avg_confidence:.1%}</h2>
            <p>Average Confidence</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("" "")

    # Main Tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Analyze Message", "📂 Upload File", "📊 Analytics Dashboard"])

    # ---- Tab 1: Write Message ----
    with tab1:
        
        st.subheader("✍️ Analyze Email Content")
        
        # Message input area
        message_input = st.text_area(
            "Enter the email content below:", 
            height=200, 
            placeholder="Type or paste your email content here...\n\nExample: 'Congratulations! You've won a $1000 prize. Click here to claim your reward.'",
            help="The AI will analyze the text for spam characteristics"
        )
        
        # Analyze button right below the text area
        check_button = st.button("🚀 Analyze Message", use_container_width=True, type="primary")
        
        if check_button and message_input:
            with st.spinner("🔍 Analyzing email content..."):
                # Get prediction with confidence
                prediction, confidence = get_prediction_with_confidence(message_input)
                features = analyze_text_features(message_input)
                
                # Update statistics
                update_stats(prediction, confidence)
                
                # Display the main result FIRST
                st.write("---")
                st.subheader("🎯 Detection Result")
                
                if prediction == 'spam':
                    st.markdown('<div class="result-box spam">🚨 SPAM DETECTED!</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="result-box ham">✅ LEGITIMATE EMAIL (HAM)</div>', unsafe_allow_html=True)
                
                # Confidence score
                st.metric("Confidence Score", f"{confidence:.1%}")
                
                # Confidence meter
                st.write("Confidence Level:")
                confidence_color = "red" if confidence < 0.6 else "orange" if confidence < 0.8 else "green"
                st.markdown(f"""
                <div class="confidence-meter">
                    <div class="confidence-fill" style="width: {confidence*100}%; background: {confidence_color};"></div>
                </div>
                """, unsafe_allow_html=True)
                
                # Horizontal Quick Stats
                st.subheader("📊 Quick Analysis")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Text Length", f"{features['length']} chars")
                with col2:
                    st.metric("Word Count", features['word_count'])
                with col3:
                    st.metric("Spam Indicators", sum([features['url_keywords'], features['spam_keywords']]))
                with col4:
                    risk_level = 'High' if features['spam_keywords'] else 'Medium' if features['url_keywords'] else 'Low'
                    st.metric("Risk Level", risk_level)
                
                # Feature Analysis
                st.subheader("🔬 Detailed Text Analysis")
                feat_col1, feat_col2, feat_col3 = st.columns(3)
                
                with feat_col1:
                    st.markdown(f"""
                    <div class="feature-card">
                        <strong>📝 Text Metrics</strong><br>
                        Length: {features['length']} characters<br>
                        Words: {features['word_count']}<br>
                        Uppercase: {features['uppercase_ratio']:.1%}
                    </div>
                    """, unsafe_allow_html=True)
                
                with feat_col2:
                    st.markdown(f"""
                    <div class="feature-card">
                        <strong>⚠️ Spam Indicators</strong><br>
                        Exclamations: {features['exclamation_count']}<br>
                        Dollar Signs: {features['dollar_sign_count']}<br>
                        URLs: {'Yes' if features['url_keywords'] else 'No'}
                    </div>
                    """, unsafe_allow_html=True)
                
                with feat_col3:
                    st.markdown(f"""
                    <div class="feature-card">
                        <strong>🔍 Keywords</strong><br>
                        Spam Words: {'Yes' if features['spam_keywords'] else 'No'}<br>
                        Suspicious Terms: {features['spam_keywords']}
                    </div>
                    """, unsafe_allow_html=True)
        
        elif check_button and not message_input:
            st.warning("⚠️ Please enter some text to analyze.")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # ---- Tab 2: Upload File ----
    with tab2:
        
        st.subheader("📂 Upload Email File")
        
        uploaded_file = st.file_uploader(
            "Choose a text file (.txt, .eml):", 
            type=['txt', 'eml'],
            help="Upload email files for bulk analysis"
        )
        
        if uploaded_file is not None:
            try:
                file_content = uploaded_file.read().decode("utf-8")
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                file_content = uploaded_file.read().decode("latin-1")
            
            # File preview
            st.subheader("📜 File Preview")
            st.text_area("Content Preview:", file_content[:1500] + ("..." if len(file_content) > 1500 else ""), 
                        height=200, disabled=True, key="file_preview")
            
            if st.button("🔍 Analyze Uploaded File", use_container_width=True):
                with st.spinner("Processing file content..."):
                    prediction, confidence = get_prediction_with_confidence(file_content)
                    update_stats(prediction, confidence)
                    
                    
                    st.markdown("---")
                    st.subheader("🎯 Detection Result")
                    
                    if prediction == 'spam':
                        st.markdown('<div class="result-box spam">🚨 FILE CONTAINS SPAM!</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="result-box ham">✅ FILE IS SAFE (HAM)</div>', unsafe_allow_html=True)
                
                    
                    # Confidence meter
                    st.write("Confidence Level:")
                    confidence_color = "red" if confidence < 0.6 else "orange" if confidence < 0.8 else "green"
                    st.markdown(f"""
                    <div class="confidence-meter">
                        <div class="confidence-fill" style="width: {confidence*100}%; background: {confidence_color};"></div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Analyze text features from the file content
                    features = analyze_text_features(file_content)
                    
                    # Horizontal Quick Stats
                    st.subheader("📊 Quick Analysis")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Text Length", f"{features['length']} chars")
                    with col2:
                        st.metric("Word Count", features['word_count'])
                    with col3:
                        st.metric("Spam Indicators", sum([features['url_keywords'], features['spam_keywords']]))
                    with col4:
                        st.metric("Confidence", f"{confidence:.1%}")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # ---- Tab 3: Analytics Dashboard ----
    with tab3:
        
        st.subheader("📊 Analytics Dashboard")
        
        if st.session_state.stats['total'] > 0:
            # Charts and Graphs in two columns
            col1, col2 = st.columns(2)
            
            with col1:
                # Enhanced Pie chart
                fig_pie = px.pie(
                    values=[st.session_state.stats['ham'], st.session_state.stats['spam']],
                    names=['Ham (Legitimate)', 'Spam'],
                    title='📈 Email Distribution: Ham vs Spam',
                    color=['Ham', 'Spam'],
                    color_discrete_map={'Ham': '#16a34a', 'Spam': '#dc2626'},
                    hole=0.3
                )
                fig_pie.update_traces(
                    textposition='inside', 
                    textinfo='percent+label',
                    marker=dict(line=dict(color='#ffffff', width=2))
                )
                fig_pie.update_layout(
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Line chart for detection trends
                if len(st.session_state.detection_trends) > 1:
                    df_trends = pd.DataFrame(st.session_state.detection_trends)
                    fig_line = go.Figure()
                    
                    fig_line.add_trace(go.Scatter(
                        x=df_trends['timestamp'],
                        y=df_trends['ham_count'],
                        mode='lines+markers',
                        name='Ham Detections',
                        line=dict(color='#16a34a', width=3),
                        marker=dict(size=6)
                    ))
                    
                    fig_line.add_trace(go.Scatter(
                        x=df_trends['timestamp'],
                        y=df_trends['spam_count'],
                        mode='lines+markers',
                        name='Spam Detections',
                        line=dict(color='#dc2626', width=3),
                        marker=dict(size=6)
                    ))
                    
                    fig_line.update_layout(
                        title='📊 Detection Trends Over Time',
                        xaxis_title='Time',
                        yaxis_title='Number of Detections',
                        hovermode='x unified',
                        showlegend=True,
                        height=400
                    )
                    
                    st.plotly_chart(fig_line, use_container_width=True)
                else:
                    # Show confidence distribution if not enough data for trends
                    if st.session_state.stats['confidence_scores']:
                        fig_hist = px.histogram(
                            x=st.session_state.stats['confidence_scores'],
                            title='📊 Confidence Score Distribution',
                            labels={'x': 'Confidence Score', 'y': 'Frequency'},
                            color_discrete_sequence=['#2563eb'],
                            nbins=10
                        )
                        fig_hist.update_layout(
                            bargap=0.1,
                            showlegend=False
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
                    else:
                        st.info("Not enough data for trends yet. Analyze more messages!")
            
            # Additional charts in second row
            col3, col4 = st.columns(2)
            
            with col3:
                # Bar chart for detection comparison
                fig_bar = px.bar(
                    x=['Ham', 'Spam'],
                    y=[st.session_state.stats['ham'], st.session_state.stats['spam']],
                    title='📊 Detection Comparison',
                    color=['Ham', 'Spam'],
                    color_discrete_map={'Ham': '#16a34a', 'Spam': '#dc2626'},
                    labels={'x': 'Category', 'y': 'Count'}
                )
                fig_bar.update_layout(showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with col4:
                # Performance metrics
                st.subheader("🎯 Performance Metrics")
                col_met1, col_met2 = st.columns(2)
                
                with col_met1:
                    st.metric("Total Analysis", st.session_state.stats['total'])
                    st.metric("Spam Rate", 
                             f"{(st.session_state.stats['spam']/st.session_state.stats['total']*100):.1f}%")
                
                with col_met2:
                    st.metric("Ham Rate", 
                             f"{(st.session_state.stats['ham']/st.session_state.stats['total']*100):.1f}%")
                    st.metric("Avg Confidence", f"{avg_confidence:.1%}")
            
            # Recent Activity Timeline
            st.subheader("🕒 Recent Activity")
            if st.session_state.recent_activity:
                for activity in st.session_state.recent_activity:
                    bg_color = "rgba(220, 38, 38, 0.1)" if activity['type'] == 'SPAM' else "rgba(22, 163, 74, 0.1)"
                    border_color = "#dc2626" if activity['type'] == 'SPAM' else "#16a34a"
                    
                    st.markdown(f"""
                    <div class="activity-item" style="border-left-color: {border_color}; background: {bg_color};">
                        <strong>{activity['time']}</strong> | {activity['date']} | 
                        <span style="color: {border_color}; font-weight: bold;">{activity['type']}</span> | 
                        Confidence: {activity['confidence']}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No recent activity to display.")
            
            # Export Data
            st.subheader("📤 Export Data")
            if st.button("Download Analysis Report", use_container_width=True):
                # Create a simple report
                report = f"""
                Spam Detection Analysis Report
                Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                
                Summary:
                - Total Messages: {st.session_state.stats['total']}
                - Ham Messages: {st.session_state.stats['ham']}
                - Spam Messages: {st.session_state.stats['spam']}
                - Spam Rate: {st.session_state.stats['spam']/max(1, st.session_state.stats['total']):.1%}
                - Average Confidence: {np.mean(st.session_state.stats['confidence_scores']):.1% if st.session_state.stats['confidence_scores'] else 'N/A'}
                """
                
                st.download_button(
                    label="📥 Download Report",
                    data=report,
                    file_name=f"spam_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        else:
            st.info("📊 Analytics will appear here after you analyze some emails. Go to the 'Analyze Message' tab to get started!")
        
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.error("""
    ⚠️ Application initialization failed. 
    Please ensure both model files (`spam_detector_model.joblib` and `tfidf_vectorizer.joblib`) 
    are present in the application directory.
    """)

# ---------------- FOOTER ----------------
st.markdown("""
<div style="text-align: center; margin-top: 3rem; padding: 2rem; color: #6b7280;">
    <hr style="border: 1px solid #e5e7eb; margin-bottom: 1rem;">
    <p>Built with ❤️ using Streamlit • Powered by Machine Learning • Smart Email Spam Detector v2.0</p>
</div>
""", unsafe_allow_html=True)