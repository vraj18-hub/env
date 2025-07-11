import streamlit as st
from transformers import pipeline
import networkx as nx
import matplotlib.pyplot as plt

# Load models (cached for performance)
@st.cache_resource
def load_pipelines():
    # Sentence classification: Replace with your environmental classifier or use zero-shot
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    # NER
    ner = pipeline("ner", grouped_entities=True)
    # Masked LM
    mask_filler = pipeline("fill-mask", model="bert-base-uncased")
    return classifier, ner, mask_filler

classifier, ner, mask_filler = load_pipelines()

st.title("Environmental NLP & Image Generation App (No spaCy)")

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
    labels = ["Climate Change", "Pollution", "Wildlife", "Renewable Energy", "Waste Management"]
    if st.button("Classify"):
        if text:
            result = classifier(text, labels)
            for label, score in zip(result['labels'], result['scores']):
                st.write(f"**{label}**: {score:.2f}")
        else:
            st.warning("Please enter a sentence.")

elif task == "Image Generation":
    st.header("Image Generation (Environmental Themes)")
    prompt = st.text_input("Describe the environmental image you want to generate (e.g., 'A forest with clean river')")
    if st.button("Generate Image"):
        if prompt:
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
            entities = ner(ner_text)
            st.write("Extracted Entities:")
            for ent in entities:
                st.write(f"**{ent['word']}** ({ent['entity_group']})")
            # Build entity graph
            G = nx.Graph()
            for ent in entities:
                G.add_node(ent['word'], label=ent['entity_group'])
            # Simple: connect all entities found in this text
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    G.add_edge(entities[i]['word'], entities[j]['word'])
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
