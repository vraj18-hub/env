import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMaskedLM
import spacy
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
import torch

# Load models (replace with your preferred or custom fine-tuned models)
@st.cache_resource
def load_models():
    # Sentence classification (example: 5 environmental categories)
    classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")
    # NER
    nlp = spacy.load("en_core_web_sm")
    # Masked LM (fill the blank)
    mask_filler = pipeline("fill-mask", model="bert-base-uncased")
    return classifier, nlp, mask_filler

classifier, nlp, mask_filler = load_models()

st.title("Environmental NLP & Image Generation App")

# Sidebar for navigation
task = st.sidebar.selectbox(
    "Choose a task",
    [
        "Sentence Classification",
        "Image Generation",
        "NER Graph Map",
        "Fill the Blank (Masking)"
    ]
)

if task == "Sentence Classification":
    st.header("Sentence Classification (Environmental Categories)")
    text = st.text_area("Enter a sentence related to environment, energy, pollution, etc.")
    if st.button("Classify"):
        if text:
            # Example: using a generic classifier, replace with your fine-tuned environmental classifier
            preds = classifier(text, top_k=5)
            st.write("Predicted categories and scores:")
            for pred in preds:
                st.write(f"**{pred['label']}**: {pred['score']:.2f}")
        else:
            st.warning("Please enter a sentence.")

elif task == "Image Generation":
    st.header("Image Generation (Environmental Themes)")
    prompt = st.text_input("Describe the environmental image you want to generate (e.g., 'A forest with clean river')")
    if st.button("Generate Image"):
        if prompt:
            # Example: use Stable Diffusion or similar model if available
            from diffusers import StableDiffusionPipeline
            import torch
            sd_pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("cpu")
            image = sd_pipe(prompt).images[0]
            st.image(image, caption=prompt)
        else:
            st.warning("Please enter an image description.")

elif task == "NER Graph Map":
    st.header("Named Entity Recognition (NER) with Graph Map")
    ner_text = st.text_area("Enter text to extract entities and visualize relationships.")
    if st.button("Extract & Visualize NER"):
        if ner_text:
            doc = nlp(ner_text)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            st.write("Extracted Entities:")
            for ent, label in entities:
                st.write(f"**{ent}** ({label})")
            # Create a simple entity graph
            G = nx.Graph()
            for ent, label in entities:
                G.add_node(ent, label=label)
            # Example: connect entities that appear in the same sentence
            for sent in doc.sents:
                ents_in_sent = [ent.text for ent in sent.ents]
                for i in range(len(ents_in_sent)):
                    for j in range(i + 1, len(ents_in_sent)):
                        G.add_edge(ents_in_sent[i], ents_in_sent[j])
            plt.figure(figsize=(8, 6))
            nx.draw(G, with_labels=True, node_color='lightgreen', font_size=10)
            st.pyplot(plt)
        else:
            st.warning("Please enter text for NER.")

elif task == "Fill the Blank (Masking)":
    st.header("Fill the Blank (Environmental Masking)")
    mask_text = st.text_input("Enter a sentence with [MASK] for the blank (e.g., 'Solar energy is a [MASK] resource.')")
    if st.button("Predict Mask"):
        if mask_text and "[MASK]" in mask_text:
            results = mask_filler(mask_text)
            st.write("Predicted words for the blank:")
            for res in results:
                st.write(f"**{res['sequence']}** (score: {res['score']:.2f})")
        else:
            st.warning("Please enter a sentence with [MASK].")
