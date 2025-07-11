import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from PIL import Image
import requests
import json
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM
import torch
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Environmental AI Hub",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #228B22;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #F0FFF0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #32CD32;
        margin: 1rem 0;
    }
    .result-box {
        background-color: #E6F3FF;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1E90FF;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'generated_images' not in st.session_state:
    st.session_state.generated_images = []
if 'ner_results' not in st.session_state:
    st.session_state.ner_results = []

# Load models (with caching for better performance)
@st.cache_resource
def load_classification_model():
    """Load sentence classification model"""
    try:
        classifier = pipeline("zero-shot-classification", 
                            model="facebook/bart-large-mnli")
        return classifier
    except Exception as e:
        st.error(f"Error loading classification model: {e}")
        return None

@st.cache_resource
def load_ner_model():
    """Load NER model"""
    try:
        ner_model = pipeline("ner", 
                           model="dbmdz/bert-large-cased-finetuned-conll03-english",
                           aggregation_strategy="simple")
        return ner_model
    except Exception as e:
        st.error(f"Error loading NER model: {e}")
        return None

@st.cache_resource
def load_fill_mask_model():
    """Load fill mask model"""
    try:
        fill_mask = pipeline("fill-mask", 
                           model="bert-base-uncased")
        return fill_mask
    except Exception as e:
        st.error(f"Error loading fill mask model: {e}")
        return None

# Classification categories
ENVIRONMENTAL_CATEGORIES = [
    "Air Quality and Pollution",
    "Water Resources and Conservation",
    "Climate Change and Global Warming",
    "Renewable Energy and Sustainability",
    "Waste Management and Recycling",
    "Biodiversity and Wildlife Conservation"
]

# Main app
def main():
    st.markdown('<h1 class="main-header">🌱 Environmental AI Hub</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a feature:",
        ["🏠 Home", "📝 Sentence Classification", "🎨 Image Generation", 
         "🔍 NER & Graph Mapping", "📝 Fill the Blank"]
    )
    
    if page == "🏠 Home":
        show_home()
    elif page == "📝 Sentence Classification":
        show_classification()
    elif page == "🎨 Image Generation":
        show_image_generation()
    elif page == "🔍 NER & Graph Mapping":
        show_ner_mapping()
    elif page == "📝 Fill the Blank":
        show_fill_blank()

def show_home():
    st.markdown("""
    <div class="info-box">
    <h2>Welcome to the Environmental AI Hub! 🌍</h2>
    <p>This application provides four powerful AI-driven features for environmental text analysis and content generation:</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📝 Sentence Classification")
        st.write("Classify environmental sentences into 6 categories:")
        st.write("• Air Quality and Pollution")
        st.write("• Water Resources and Conservation")
        st.write("• Climate Change and Global Warming")
        st.write("• Renewable Energy and Sustainability")
        st.write("• Waste Management and Recycling")
        st.write("• Biodiversity and Wildlife Conservation")
        
        st.markdown("### 🎨 Image Generation")
        st.write("Generate environmental images using AI prompts")
        st.write("• Nature scenes")
        st.write("• Environmental concepts")
        st.write("• Sustainability themes")
    
    with col2:
        st.markdown("### 🔍 NER & Graph Mapping")
        st.write("Extract named entities and visualize relationships")
        st.write("• Person, Organization, Location recognition")
        st.write("• Interactive network graphs")
        st.write("• Entity relationship mapping")
        
        st.markdown("### 📝 Fill the Blank")
        st.write("Complete environmental sentences using AI")
        st.write("• Context-aware predictions")
        st.write("• Environmental vocabulary")
        st.write("• Multiple suggestions")

def show_classification():
    st.markdown('<h2 class="sub-header">📝 Environmental Sentence Classification</h2>', unsafe_allow_html=True)
    
    # Load model
    classifier = load_classification_model()
    
    if classifier is None:
        st.error("Failed to load classification model. Please check your internet connection.")
        return
    
    st.markdown("""
    <div class="info-box">
    Enter an environmental sentence below, and the AI will classify it into one of 6 environmental categories.
    </div>
    """, unsafe_allow_html=True)
    
    # Input section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_input = st.text_area(
            "Enter your environmental sentence:",
            placeholder="e.g., Solar panels are becoming more efficient and affordable for homeowners.",
            height=100
        )
    
    with col2:
        st.markdown("### Sample Sentences:")
        sample_sentences = [
            "Solar panels reduce carbon emissions significantly.",
            "Plastic waste is polluting our oceans.",
            "Deforestation threatens endangered species.",
            "Electric vehicles are becoming more popular.",
            "Air pollution causes respiratory problems.",
            "Recycling programs help reduce landfill waste."
        ]
        
        for i, sentence in enumerate(sample_sentences):
            if st.button(f"Use Sample {i+1}", key=f"sample_{i}"):
                user_input = sentence
                st.rerun()
    
    if st.button("Classify Sentence", type="primary"):
        if user_input.strip():
            with st.spinner("Classifying sentence..."):
                try:
                    result = classifier(user_input, ENVIRONMENTAL_CATEGORIES)
                    
                    st.markdown(f"""
                    <div class="result-box">
                    <h3>Classification Results:</h3>
                    <p><strong>Input:</strong> {user_input}</p>
                    <p><strong>Top Category:</strong> {result['labels'][0]}</p>
                    <p><strong>Confidence:</strong> {result['scores'][0]:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Create visualization
                    fig, ax = plt.subplots(figsize=(10, 6))
                    categories = result['labels'][:5]  # Top 5 categories
                    scores = result['scores'][:5]
                    
                    bars = ax.barh(categories, scores, color='lightgreen')
                    ax.set_xlabel('Confidence Score')
                    ax.set_title('Classification Confidence by Category')
                    ax.set_xlim(0, 1)
                    
                    # Add value labels on bars
                    for i, (bar, score) in enumerate(zip(bars, scores)):
                        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                               f'{score:.2%}', va='center')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Show detailed results
                    st.markdown("### Detailed Results:")
                    results_df = pd.DataFrame({
                        'Category': result['labels'],
                        'Confidence': [f"{score:.2%}" for score in result['scores']]
                    })
                    st.dataframe(results_df, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error during classification: {e}")
        else:
            st.warning("Please enter a sentence to classify.")

def show_image_generation():
    st.markdown('<h2 class="sub-header">🎨 Environmental Image Generation</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    Generate environmental images using AI. Note: This is a simulation - in a real application, 
    you would integrate with services like DALL-E, Midjourney, or Stable Diffusion.
    </div>
    """, unsafe_allow_html=True)
    
    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        prompt = st.text_input(
            "Enter your image prompt:",
            placeholder="e.g., A beautiful forest with renewable energy windmills"
        )
        
        style = st.selectbox(
            "Choose image style:",
            ["Realistic", "Artistic", "Cartoon", "Minimalist", "Vintage"]
        )
        
        size = st.selectbox(
            "Image size:",
            ["512x512", "1024x1024", "1920x1080"]
        )
    
    with col2:
        st.markdown("### Suggested Prompts:")
        suggested_prompts = [
            "Solar panel farm in a green valley",
            "Clean ocean with marine life",
            "Sustainable city with green buildings",
            "Wind turbines on rolling hills",
            "Recycling facility with modern design",
            "Electric car charging station"
        ]
        
        for i, suggestion in enumerate(suggested_prompts):
            if st.button(f"Use: {suggestion[:25]}...", key=f"prompt_{i}"):
                prompt = suggestion
                st.rerun()
    
    if st.button("Generate Image", type="primary"):
        if prompt.strip():
            with st.spinner("Generating image..."):
                # Simulate image generation
                st.success("Image generated successfully!")
                
                # Create a placeholder visualization
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Generate random environmental-themed colors
                colors = ['#2E8B57', '#228B22', '#32CD32', '#90EE90', '#98FB98']
                
                # Create abstract environmental visualization
                np.random.seed(hash(prompt) % 1000)
                
                if "solar" in prompt.lower():
                    # Solar panel simulation
                    for i in range(20):
                        x = np.random.uniform(0, 10)
                        y = np.random.uniform(0, 8)
                        ax.add_patch(plt.Rectangle((x, y), 0.5, 0.3, 
                                                 color='darkblue', alpha=0.7))
                    ax.set_title(f"Generated: {prompt}", fontsize=14, color='#2E8B57')
                
                elif "forest" in prompt.lower():
                    # Forest simulation
                    for i in range(30):
                        x = np.random.uniform(0, 10)
                        y = np.random.uniform(0, 8)
                        ax.scatter(x, y, s=np.random.uniform(50, 200), 
                                 c=np.random.choice(colors), alpha=0.8)
                    ax.set_title(f"Generated: {prompt}", fontsize=14, color='#228B22')
                
                else:
                    # Generic environmental visualization
                    x = np.random.uniform(0, 10, 100)
                    y = np.random.uniform(0, 8, 100)
                    colors_random = np.random.choice(colors, 100)
                    ax.scatter(x, y, c=colors_random, s=50, alpha=0.7)
                    ax.set_title(f"Generated: {prompt}", fontsize=14, color='#2E8B57')
                
                ax.set_xlim(0, 10)
                ax.set_ylim(0, 8)
                ax.axis('off')
                
                st.pyplot(fig)
                
                # Image details
                st.markdown(f"""
                <div class="result-box">
                <h3>Image Details:</h3>
                <p><strong>Prompt:</strong> {prompt}</p>
                <p><strong>Style:</strong> {style}</p>
                <p><strong>Size:</strong> {size}</p>
                <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Please enter an image prompt.")

def show_ner_mapping():
    st.markdown('<h2 class="sub-header">🔍 Named Entity Recognition & Graph Mapping</h2>', unsafe_allow_html=True)
    
    # Load NER model
    ner_model = load_ner_model()
    
    if ner_model is None:
        st.error("Failed to load NER model. Please check your internet connection.")
        return
    
    st.markdown("""
    <div class="info-box">
    Extract named entities from environmental text and visualize their relationships in an interactive graph.
    </div>
    """, unsafe_allow_html=True)
    
    # Input section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        text_input = st.text_area(
            "Enter environmental text:",
            placeholder="e.g., The Environmental Protection Agency (EPA) announced that Tesla will build a new solar panel factory in California to combat climate change.",
            height=150
        )
    
    with col2:
        st.markdown("### Sample Texts:")
        sample_texts = [
            "The Environmental Protection Agency (EPA) announced new regulations for carbon emissions in New York.",
            "Greenpeace is working with the United Nations to protect the Amazon rainforest in Brazil.",
            "Tesla CEO Elon Musk unveiled solar panels manufactured in Nevada to reduce pollution in Los Angeles.",
            "The World Wildlife Fund is collaborating with local communities in Kenya to protect elephants."
        ]
        
        for i, text in enumerate(sample_texts):
            if st.button(f"Sample {i+1}", key=f"ner_sample_{i}"):
                text_input = text
                st.rerun()
    
    if st.button("Extract Entities & Create Graph", type="primary"):
        if text_input.strip():
            with st.spinner("Processing text and creating graph..."):
                try:
                    # Extract entities
                    entities = ner_model(text_input)
                    
                    if entities:
                        # Display entities
                        st.markdown("### Extracted Entities:")
                        
                        entity_df = pd.DataFrame(entities)
                        entity_df['confidence'] = entity_df['score'].apply(lambda x: f"{x:.2%}")
                        
                        # Group by entity type
                        entity_types = entity_df['entity_group'].unique()
                        
                        for entity_type in entity_types:
                            st.markdown(f"**{entity_type}:**")
                            type_entities = entity_df[entity_df['entity_group'] == entity_type]
                            st.dataframe(type_entities[['word', 'confidence']], use_container_width=True)
                        
                        # Create network graph
                        st.markdown("### Entity Relationship Graph:")
                        
                        # Create graph
                        G = nx.Graph()
                        
                        # Add nodes
                        for entity in entities:
                            G.add_node(entity['word'], 
                                     type=entity['entity_group'],
                                     confidence=entity['score'])
                        
                        # Add edges (connect entities that appear in the same text)
                        entity_words = [entity['word'] for entity in entities]
                        for i in range(len(entity_words)):
                            for j in range(i+1, len(entity_words)):
                                G.add_edge(entity_words[i], entity_words[j])
                        
                        # Create visualization using Plotly
                        pos = nx.spring_layout(G, k=3, iterations=50)
                        
                        # Extract node and edge information
                        node_x = [pos[node][0] for node in G.nodes()]
                        node_y = [pos[node][1] for node in G.nodes()]
                        node_text = list(G.nodes())
                        node_colors = []
                        
                        color_map = {
                            'PER': '#FF6B6B',  # Person - Red
                            'ORG': '#4ECDC4',  # Organization - Teal
                            'LOC': '#45B7D1',  # Location - Blue
                            'MISC': '#96CEB4'  # Miscellaneous - Green
                        }
                        
                        for node in G.nodes():
                            node_type = G.nodes[node]['type']
                            node_colors.append(color_map.get(node_type, '#95A5A6'))
                        
                        # Create edges
                        edge_x = []
                        edge_y = []
                        
                        for edge in G.edges():
                            x0, y0 = pos[edge[0]]
                            x1, y1 = pos[edge[1]]
                            edge_x.extend([x0, x1, None])
                            edge_y.extend([y0, y1, None])
                        
                        # Create Plotly figure
                        fig = go.Figure()
                        
                        # Add edges
                        fig.add_trace(go.Scatter(
                            x=edge_x, y=edge_y,
                            line=dict(width=2, color='#E0E0E0'),
                            hoverinfo='none',
                            mode='lines',
                            name='Connections'
                        ))
                        
                        # Add nodes
                        fig.add_trace(go.Scatter(
                            x=node_x, y=node_y,
                            mode='markers+text',
                            text=node_text,
                            textposition="middle center",
                            textfont=dict(size=12, color='white'),
                            marker=dict(
                                size=30,
                                color=node_colors,
                                line=dict(width=2, color='white')
                            ),
                            hovertemplate='<b>%{text}</b><br>' +
                                        'Type: %{customdata}<br>' +
                                        '<extra></extra>',
                            customdata=[G.nodes[node]['type'] for node in G.nodes()],
                            name='Entities'
                        ))
                        
                        fig.update_layout(
                            title='Entity Relationship Network',
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40),
                            annotations=[ dict(
                                text="Interactive graph: hover over nodes for details",
                                showarrow=False,
                                xref="paper", yref="paper",
                                x=0.005, y=-0.002,
                                xanchor="left", yanchor="bottom",
                                font=dict(color="#888", size=12)
                            )],
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            height=600
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Entity statistics
                        st.markdown("### Entity Statistics:")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Entities", len(entities))
                        
                        with col2:
                            st.metric("Entity Types", len(entity_types))
                        
                        with col3:
                            avg_confidence = np.mean([entity['score'] for entity in entities])
                            st.metric("Avg Confidence", f"{avg_confidence:.2%}")
                        
                        # Entity type distribution
                        type_counts = entity_df['entity_group'].value_counts()
                        
                        fig_pie = px.pie(
                            values=type_counts.values,
                            names=type_counts.index,
                            title="Entity Type Distribution",
                            color_discrete_map=color_map
                        )
                        
                        st.plotly_chart(fig_pie, use_container_width=True)
                        
                    else:
                        st.warning("No entities found in the text.")
                    
                except Exception as e:
                    st.error(f"Error during NER processing: {e}")
        else:
            st.warning("Please enter some text to analyze.")

def show_fill_blank():
    st.markdown('<h2 class="sub-header">📝 Environmental Fill-in-the-Blank</h2>', unsafe_allow_html=True)
    
    # Load fill mask model
    fill_mask_model = load_fill_mask_model()
    
    if fill_mask_model is None:
        st.error("Failed to load fill mask model. Please check your internet connection.")
        return
    
    st.markdown("""
    <div class="info-box">
    Complete environmental sentences by filling in the blanks. Use [MASK] to indicate where you want the AI to predict words.
    </div>
    """, unsafe_allow_html=True)
    
    # Input section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        masked_text = st.text_area(
            "Enter text with [MASK] tokens:",
            placeholder="e.g., Solar energy is a [MASK] source of renewable power that helps reduce [MASK] emissions.",
            height=120
        )
        
        num_predictions = st.slider(
            "Number of predictions per mask:",
            min_value=1,
            max_value=5,
            value=3
        )
    
    with col2:
        st.markdown("### Sample Sentences:")
        sample_sentences = [
            "Solar energy is a [MASK] source of renewable power.",
            "Plastic pollution is [MASK] the ocean ecosystem.",
            "Electric vehicles help reduce [MASK] emissions.",
            "Deforestation leads to loss of [MASK] habitats.",
            "Recycling helps [MASK] waste and conserve resources.",
            "Climate change is [MASK] global temperatures."
        ]
        
        for i, sentence in enumerate(sample_sentences):
            if st.button(f"Use Sample {i+1}", key=f"mask_sample_{i}"):
                masked_text = sentence
                st.rerun()
    
    if st.button("Fill the Blanks", type="primary"):
        if masked_text.strip() and "[MASK]" in masked_text:
            with st.spinner("Predicting missing words..."):
                try:
                    # Get predictions
                    predictions = fill_mask_model(masked_text, top_k=num_predictions)
                    
                    st.markdown("### Predictions:")
                    
                    if isinstance(predictions[0], list):
                        # Multiple masks
                        for i, mask_predictions in enumerate(predictions):
                            st.markdown(f"**Mask {i+1}:**")
                            
                            pred_df = pd.DataFrame(mask_predictions)
                            pred_df['confidence'] = pred_df['score'].apply(lambda x: f"{x:.2%}")
                            pred_df = pred_df[['token_str', 'confidence']]
                            pred_df.columns = ['Predicted Word', 'Confidence']
                            
                            st.dataframe(pred_df, use_container_width=True)
                            
                            # Show completed sentences
                            st.markdown("**Completed Sentences:**")
                            for j, pred in enumerate(mask_predictions):
                                completed = pred['sequence']
                                st.write(f"{j+1}. {completed}")
                            
                            st.markdown("---")
                    
                    else:
                        # Single mask
                        pred_df = pd.DataFrame(predictions)
                        pred_df['confidence'] = pred_df['score'].apply(lambda x: f"{x:.2%}")
                        pred_df = pred_df[['token_str', 'confidence']]
                        pred_df.columns = ['Predicted Word', 'Confidence']
                        
                        st.dataframe(pred_df, use_container_width=True)
                        
                        # Show completed sentences
                        st.markdown("**Completed Sentences:**")
                        for i, pred in enumerate(predictions):
                            completed = pred['sequence']
                            st.write(f"{i+1}. {completed}")
                        
                        # Visualization
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        words = [pred['token_str'] for pred in predictions]
                        scores = [pred['score'] for pred in predictions]
                        
                        bars = ax.bar(words, scores, color='lightblue')
                        ax.set_ylabel('Confidence Score')
                        ax.set_title('Prediction Confidence by Word')
                        ax.set_ylim(0, max(scores) * 1.1)
                        
                        # Add value labels on bars
                        for bar, score in zip(bars, scores):
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                                   f'{score:.2%}', ha='center', va='bottom')
                        
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    # Analysis
                    st.markdown("### Analysis:")
                    
                    if isinstance(predictions[0], list):
                        total_masks = len(predictions)
                        st.write(f"✅ Successfully filled {total_masks} mask(s)")
                        
                        all_predictions = [pred for mask_preds in predictions for pred in mask_preds]
                        avg_confidence = np.mean([pred['score'] for pred in all_predictions])
                        st.write(f"📊 Average confidence: {avg_confidence:.2%}")
                    else:
                        st.write(f"✅ Successfully filled 1 mask")
                        avg_confidence = np.mean([pred['score'] for pred in predictions])
                        st.write(f"📊 Average confidence: {avg_confidence:.2%}")
                        
                        top_prediction = predictions[0]
                        st.write(f"🎯 Top prediction: **{top_prediction['token_str']}** ({top_prediction['score']:.2%})")
                    
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
        else:
            if "[MASK]" not in masked_text:
                st.warning("Please include [MASK] tokens in your text.")
            else:
                st.warning("Please enter some text with [MASK] tokens.")

# Footer
def show_footer():
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; padding: 20px;">
        🌱 Environmental AI Hub - Powered by Transformers & Streamlit<br>
        Created for environmental awareness and education
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
    show_footer()
