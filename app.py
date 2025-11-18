# app.py - Enhanced Spam Detection App (FIXED DEPRECATION WARNINGS)
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import re
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import base64
from io import StringIO

# Page configuration
st.set_page_config(
    page_title="SpamGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 800;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        background: white;
    }
    .spam-prediction {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        color: white;
    }
    .ham-prediction {
        background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
        color: white;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 0.5rem 0;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .sidebar .sidebar-content {
        background: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load the trained model and preprocessing artifacts"""
    try:
        model = joblib.load("spam_classifier_pipeline.joblib")
        artifacts = joblib.load("preprocessing_artifacts.joblib")
        return model, artifacts
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.info("Please make sure 'spam_classifier_pipeline.joblib' and 'preprocessing_artifacts.joblib' are in the same directory.")
        return None, None

def preprocess_text(text):
    """Text preprocessing function matching training pipeline"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', ' URL ', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', ' EMAIL ', text)
    
    # Remove phone numbers
    text = re.sub(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', ' PHONE ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^a-zA-Z\s\.\?\!]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_text_features(text):
    """Extract features similar to training pipeline"""
    features = {}
    
    # Basic length features
    features['char_count'] = len(text)
    features['word_count'] = len(text.split())
    features['avg_word_length'] = features['char_count'] / max(features['word_count'], 1)
    
    # Special feature indicators
    features['has_url'] = int('URL' in text)
    features['has_email'] = int('EMAIL' in text)
    features['has_phone'] = int('PHONE' in text)
    
    # Punctuation features
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    
    # Spam keywords
    spam_keywords = r'free|win|winner|click|prize|offer|limited|guarantee|cash|money|urgent|congratulations'
    features['spam_keyword_count'] = len(re.findall(spam_keywords, text))
    
    return features

def explain_prediction_enhanced(model, text, top_n=10):
    """Enhanced prediction explanation with visualizations"""
    try:
        if hasattr(model.named_steps['clf'], 'coef_'):
            vectorizer = model.named_steps['tfidf']
            classifier = model.named_steps['clf']
            
            X_vec = vectorizer.transform([text])
            feature_names = vectorizer.get_feature_names_out()
            coefficients = classifier.coef_[0]
            feature_values = X_vec.toarray()[0]
            
            contributions = coefficients * feature_values
            non_zero_idx = np.where(feature_values != 0)[0]
            
            if len(non_zero_idx) > 0:
                # Get top contributing features
                sorted_idx = non_zero_idx[np.argsort(np.abs(contributions[non_zero_idx]))[::-1]]
                
                top_features = []
                for idx in sorted_idx[:top_n]:
                    word = feature_names[idx]
                    contribution = contributions[idx]
                    value = feature_values[idx]
                    top_features.append({
                        'word': word,
                        'contribution': contribution,
                        'absolute_impact': abs(contribution),
                        'direction': 'spam' if contribution > 0 else 'not spam',
                        'value': value
                    })
                
                return top_features
    except Exception as e:
        st.error(f"Error in generating explanation: {str(e)}")
    return None

def main():
    # Header with modern design
    st.markdown('<h1 class="main-header">🛡️ SpamGuard AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Intelligent Email Spam Detection System</p>', unsafe_allow_html=True)
    
    # Load models
    with st.spinner("🔄 Loading AI models..."):
        model, artifacts = load_models()
    
    if model is None:
        st.error("""
        ❌ Models not found! Please ensure:
        1. You've trained the model using spamFinalVersion.ipynb
        2. 'spam_classifier_pipeline.joblib' and 'preprocessing_artifacts.joblib' are in the same directory
        3. Run all cells in the notebook to generate the model files
        """)
        return
    
    # Sidebar with modern design
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        detection_mode = st.radio(
            "**Detection Mode**",
            ["📨 Single Message", "📊 Batch Analysis", "🔬 Model Insights"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("### 🎯 Detection Settings")
        
        threshold = st.slider(
            "**Spam Detection Threshold**",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Adjust sensitivity for spam detection"
        )
        
        st.markdown("---")
        st.markdown("### ℹ️ Model Information")
        
        if artifacts and 'model_info' in artifacts:
            model_info = artifacts['model_info']
            st.metric("🤖 Algorithm", model_info.get('model_name', 'Logistic Regression'))
            st.metric("🎯 Accuracy", f"{model_info.get('accuracy', 0):.1%}")
            st.metric("📈 AUC-ROC", f"{model_info.get('auc_roc', 0):.3f}")
            st.metric("📅 Trained", model_info.get('training_date', 'N/A'))
        else:
            st.info("Model information not available")
    
    # Main content based on mode
    if "Single Message" in detection_mode:
        render_single_message_mode(model, threshold)
    elif "Batch Analysis" in detection_mode:
        render_batch_analysis_mode(model, threshold)
    else:
        render_model_insights_mode(model, artifacts)

def render_single_message_mode(model, threshold):
    """Render single message detection interface"""
    st.header("📨 Single Message Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Message input with examples
        message_input = st.text_area(
            "**Enter your message to analyze:**",
            height=200,
            placeholder="Paste your email or message here...\n\nExample spam indicators:\n• 'You won a prize! Click here to claim'\n• 'Free money guaranteed'\n• 'Urgent: Your account will be closed'\n• 'Congratulations! You've been selected'",
            help="The AI will analyze the text for spam characteristics and provide detailed insights"
        )
        
        analyze_clicked = st.button(
            "🔍 Analyze Message", 
            type="primary", 
            width='stretch',
            disabled=not message_input.strip()
        )
        
        if analyze_clicked:
            with st.spinner("🤖 AI is analyzing your message..."):
                time.sleep(1)  # Simulate processing for better UX
                
                # Preprocess and predict
                cleaned_text = preprocess_text(message_input)
                probability = model.predict_proba([cleaned_text])[0][1]
                prediction = 1 if probability >= threshold else 0
                
                # Display results
                display_prediction_results(message_input, prediction, probability, model, cleaned_text, threshold)

def display_prediction_results(original_text, prediction, probability, model, cleaned_text, threshold):
    """Display prediction results with enhanced visualization"""
    
    # Prediction box with modern design
    prediction_class = "spam-prediction" if prediction == 1 else "ham-prediction"
    prediction_text = "🚫 SPAM DETECTED" if prediction == 1 else "✅ LEGITIMATE MESSAGE"
    prediction_icon = "🚫" if prediction == 1 else "✅"
    confidence = probability if prediction == 1 else (1 - probability)
    
    st.markdown(f"""
    <div class="prediction-box {prediction_class}">
        <h2 style="margin: 0 0 1rem 0; font-size: 2rem;">{prediction_icon} {prediction_text}</h2>
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem;">
            <div style="text-align: center;">
                <h3 style="margin: 0; font-size: 1rem;">SPAM SCORE</h3>
                <p style="margin: 0; font-size: 2rem; font-weight: bold;">{probability:.4f}</p>
            </div>
            <div style="text-align: center;">
                <h3 style="margin: 0; font-size: 1rem;">PREDICTION</h3>
                <p style="margin: 0; font-size: 2rem; font-weight: bold;">{'SPAM' if prediction == 1 else 'NOT SPAM'}</p>
            </div>
            <div style="text-align: center;">
                <h3 style="margin: 0; font-size: 1rem;">CONFIDENCE</h3>
                <p style="margin: 0; font-size: 2rem; font-weight: bold;">{confidence:.1%}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Probability visualization
    st.subheader("📊 Probability Visualization")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Custom progress bar with threshold marker
        fig = go.Figure()
        
        # Add background
        fig.add_shape(
            type="rect",
            x0=0, x1=1, y0=0, y1=1,
            fillcolor="lightgray",
            opacity=0.2,
            layer="below",
            line_width=0,
        )
        
        # Add progress
        fig.add_shape(
            type="rect",
            x0=0, x1=probability, y0=0, y1=1,
            fillcolor="#ff6b6b" if prediction == 1 else "#51cf66",
            opacity=0.8,
            layer="above",
            line_width=0,
        )
        
        # Add threshold line
        fig.add_shape(
            type="line",
            x0=threshold, x1=threshold, y0=0, y1=1,
            line=dict(color="red", width=3, dash="dash"),
        )
        
        fig.update_layout(
            height=60,
            showlegend=False,
            xaxis=dict(showticklabels=False, range=[0, 1]),
            yaxis=dict(showticklabels=False),
            margin=dict(l=0, r=0, t=0, b=0),
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    with col2:
        st.metric("Threshold", f"{threshold:.2f}")
    
    # Feature analysis
    st.subheader("🔍 Feature Analysis")
    
    features = extract_text_features(cleaned_text)
    
    # Display key features
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Word Count", features['word_count'])
    with col2:
        st.metric("Spam Keywords", features['spam_keyword_count'])
    with col3:
        st.metric("Exclamations", features['exclamation_count'])
    with col4:
        st.metric("URLs/Emails", features['has_url'] + features['has_email'])
    
    # Detailed explanation
    explanation = explain_prediction_enhanced(model, cleaned_text)
    
    if explanation:
        st.subheader("🧠 AI Explanation")
        st.info("The chart below shows which words most influenced the prediction:")
        
        df_explanation = pd.DataFrame(explanation)
        
        # Impact chart
        fig = px.bar(
            df_explanation,
            x='absolute_impact',
            y='word',
            color='direction',
            color_discrete_map={'spam': '#ff6b6b', 'not spam': '#51cf66'},
            orientation='h',
            title="",
            labels={'absolute_impact': 'Impact Strength', 'word': 'Feature'}
        )
        
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed table
        with st.expander("📋 View Detailed Feature Analysis"):
            display_df = df_explanation.copy()
            display_df['impact'] = display_df['contribution'].apply(
                lambda x: f"{x:.4f} ({'↑ spam' if x > 0 else '↓ not spam'})"
            )
            display_df = display_df[['word', 'impact', 'value']]
            display_df.columns = ['Feature Word', 'Impact Contribution', 'TF-IDF Value']
            st.dataframe(display_df, width='stretch')
    else:
        st.info("Feature explanation is not available for this model type.")

def render_batch_analysis_mode(model, threshold):
    """Render batch analysis interface"""
    st.header("📊 Batch Message Analysis")
    
    st.info("""
    **Upload a CSV file containing messages to analyze multiple emails at once.**
    Your CSV should have a column named 'text' containing the messages.
    """)
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="CSV file must contain a 'text' column with messages"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded {len(df)} messages")
            
            # Check for required columns
            if 'text' not in df.columns:
                st.error("❌ CSV must contain a 'text' column")
                return
            
            # Show sample data
            with st.expander("👀 Preview Uploaded Data"):
                st.dataframe(df.head(), width='stretch')
            
            # Analyze messages
            if st.button("🚀 Analyze All Messages", type="primary", width='stretch'):
                with st.spinner(f"🤖 Analyzing {len(df)} messages..."):
                    progress_bar = st.progress(0)
                    
                    # Process predictions
                    df['cleaned_text'] = df['text'].apply(preprocess_text)
                    probabilities = model.predict_proba(df['cleaned_text'])[:, 1]
                    predictions = (probabilities >= threshold).astype(int)
                    
                    df['spam_probability'] = probabilities
                    df['prediction'] = predictions
                    df['prediction_label'] = df['prediction'].map({1: 'SPAM', 0: 'NOT SPAM'})
                    
                    progress_bar.progress(100)
                    
                    # Display results
                    display_batch_results(df)
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

def display_batch_results(df):
    """Display batch analysis results"""
    
    # Summary statistics
    spam_count = (df['prediction'] == 1).sum()
    total_count = len(df)
    spam_percentage = (spam_count / total_count) * 100
    
    st.subheader("📈 Analysis Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Messages", total_count)
    with col2:
        st.metric("Spam Detected", spam_count)
    with col3:
        st.metric("Spam Percentage", f"{spam_percentage:.1f}%")
    with col4:
        avg_prob = df['spam_probability'].mean()
        st.metric("Avg Spam Probability", f"{avg_prob:.4f}")
    
    # Distribution chart
    st.subheader("📊 Probability Distribution")
    
    fig = px.histogram(
        df, 
        x='spam_probability',
        nbins=20,
        title="Distribution of Spam Probabilities Across Messages",
        color_discrete_sequence=['#667eea'],
        opacity=0.8
    )
    
    fig.add_vline(
        x=0.5, 
        line_dash="dash", 
        line_color="red", 
        annotation_text="Threshold",
        annotation_position="top right"
    )
    
    fig.update_layout(
        xaxis_title="Spam Probability",
        yaxis_title="Number of Messages",
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Results table
    st.subheader("📋 Detailed Results")
    
    display_columns = ['text', 'spam_probability', 'prediction_label']
    display_df = df[display_columns].copy()
    display_df['text_preview'] = display_df['text'].str[:80] + '...'
    display_df = display_df[['text_preview', 'spam_probability', 'prediction_label']]
    display_df.columns = ['Message Preview', 'Spam Probability', 'Prediction']
    
    st.dataframe(
        display_df,
        width='stretch',
        height=400
    )
    
    # Download results
    st.subheader("💾 Download Results")
    
    csv = df[['text', 'spam_probability', 'prediction_label']].to_csv(index=False)
    
    st.download_button(
        label="📥 Download Full Results as CSV",
        data=csv,
        file_name="spam_analysis_results.csv",
        mime="text/csv",
        width='stretch'
    )

def render_model_insights_mode(model, artifacts):
    """Render model insights and information"""
    st.header("🔬 Model Insights & Performance")
    
    if artifacts and 'model_info' in artifacts:
        model_info = artifacts['model_info']
        
        # Model performance metrics
        st.subheader("📊 Model Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🤖 Algorithm", model_info.get('model_name', 'Logistic Regression'))
        with col2:
            st.metric("🎯 Accuracy", f"{model_info.get('accuracy', 0):.1%}")
        with col3:
            st.metric("📈 AUC-ROC", f"{model_info.get('auc_roc', 0):.3f}")
        with col4:
            st.metric("📅 Training Date", model_info.get('training_date', 'N/A'))
    
    # Feature importance visualization
    if hasattr(model.named_steps['clf'], 'coef_'):
        st.subheader("🔍 Feature Importance")
        st.info("Top words that indicate spam or legitimate messages:")
        
        vectorizer = model.named_steps['tfidf']
        classifier = model.named_steps['clf']
        
        feature_names = vectorizer.get_feature_names_out()
        coefficients = classifier.coef_[0]
        
        # Get top spam and ham indicators
        top_spam_idx = np.argsort(coefficients)[-15:]
        top_ham_idx = np.argsort(coefficients)[:15]
        
        top_spam_words = [feature_names[i] for i in top_spam_idx]
        top_spam_scores = coefficients[top_spam_idx]
        
        top_ham_words = [feature_names[i] for i in top_ham_idx]
        top_ham_scores = coefficients[top_ham_idx]
        
        # Create comparison chart
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('🚫 Top Spam Indicators', '✅ Top Legitimate Indicators'),
            horizontal_spacing=0.2
        )
        
        fig.add_trace(
            go.Bar(
                y=top_spam_words, 
                x=top_spam_scores, 
                orientation='h', 
                marker_color='#ff6b6b', 
                name='Spam Indicators'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                y=top_ham_words, 
                x=top_ham_scores, 
                orientation='h',
                marker_color='#51cf66', 
                name='Legitimate Indicators'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=600, 
            showlegend=False, 
            title_text="Feature Importance Analysis",
            font=dict(size=12)
        )
        
        fig.update_xaxes(title_text="Coefficient Value", row=1, col=1)
        fig.update_xaxes(title_text="Coefficient Value", row=1, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Model explanation
    st.subheader("🤔 How It Works")
    
    st.markdown("""
    <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #667eea;">
    <h4 style="margin-top: 0;">The spam detection system uses advanced machine learning to analyze messages based on:</h4>
    
    - **📝 Text Patterns**: Identifies common spam phrases and patterns
    - **🔢 Word Frequency**: Analyzes how often suspicious words appear (TF-IDF)
    - **🏗️ Message Structure**: Examines the overall composition and formatting
    - **📊 Linguistic Features**: Looks at writing style, punctuation, and formatting
    - **🔗 External Indicators**: Detects URLs, email addresses, and phone numbers
    
    The model was trained on a diverse dataset of spam and legitimate messages to achieve high accuracy 
    in distinguishing between unwanted spam and legitimate communications.
    </div>
    """, unsafe_allow_html=True)
    
    # Usage tips
    st.subheader("💡 Usage Tips")
    
    tip_col1, tip_col2 = st.columns(2)
    
    with tip_col1:
        st.markdown("""
        **For Best Results:**
        - Use the threshold slider to adjust sensitivity
        - Analyze individual messages for detailed insights
        - Upload CSV files for batch processing
        - Review the feature explanations to understand decisions
        """)
    
    with tip_col2:
        st.markdown("""
        **Common Spam Indicators:**
        - Urgent action required
        - Prize/winner announcements
        - Free offers and guarantees
        - Multiple exclamation marks!!!
        - Suspicious URLs and email addresses
        """)

if __name__ == "__main__":
    main()