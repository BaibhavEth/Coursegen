import streamlit as st
import json
import os
from concept_map_extractor import (
    process_pdf,
    extract_higher_level_concepts,
    extract_dependencies,
    convert_dependencies_to_json,
    save_dependency_graph
)
from learning_strategy_content_generator import generate_content_for_concept, learning_styles, initialize_rag_components
from langchain.chat_models import ChatOpenAI
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import logging
from streamlit_agraph import agraph, Node, Edge, Config
import graphviz

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def send_email(recipient_email, attachment_path):
    # Email configuration
    sender_email = "metaaligator8@gmail.com"  # Replace with your Gmail address
    sender_password = "mvgf novu dkfz yjhx"  # Replace with your app password

    # Create the email message
    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = recipient_email
    message['Subject'] = "Your Generated Content is Ready"

    # Email body
    body = "Your requested content has been generated. Please find it attached."
    message.attach(MIMEText(body, 'plain'))

    # Attach the JSON file
    with open(attachment_path, "rb") as attachment:
        part = MIMEApplication(attachment.read(), Name=os.path.basename(attachment_path))
    part['Content-Disposition'] = f'attachment; filename="{os.path.basename(attachment_path)}"'
    message.attach(part)

    # Send the email
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, sender_password)
            server.send_message(message)
        print(f"Email sent successfully to {recipient_email}")
    except Exception as e:
        print(f"Error sending email: {e}")

def display_concept_map(dependency_dict):
    # Create a new directed graph
    dot = graphviz.Digraph()
    dot.attr(rankdir='TB')  # Top to Bottom direction
    
    # Set default node style
    dot.attr('node', 
            shape='rectangle',
            style='rounded,filled',
            fillcolor='lightblue',
            fontname='Arial',
            fontsize='12',
            margin='0.2')
    
    # Set default edge style
    dot.attr('edge', 
            color='gray50',
            arrowsize='0.8')

    # Find root concepts
    all_deps = set(dep for deps in dependency_dict.values() for dep in deps)
    root_concepts = set(dependency_dict.keys()) - all_deps

    # Add root nodes with different style
    for concept in root_concepts:
        dot.node(concept, concept, 
                shape='rectangle',
                style='rounded,filled',
                fillcolor='lightcoral')

    # Add other nodes and edges
    for concept, dependencies in dependency_dict.items():
        if concept not in root_concepts:
            dot.node(concept, concept)
        for dep in dependencies:
            dot.edge(dep, concept)

    # Render the graph in Streamlit
    st.graphviz_chart(dot)

st.title("Concept Map and Content Generator")

# Add learning strategy selection
learning_strategy = st.selectbox(
    "Choose a learning strategy",
    options=list(learning_styles.keys())
)

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    print("Uploaded file is not None")
    with open("uploaded_file.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.write("Loading PDF...")
    concepts = process_pdf("uploaded_file.pdf")
    
    st.write("Extracting higher-level concepts...")
    higher_level_concepts = extract_higher_level_concepts(concepts)
    
    st.write("Extracting concept dependencies...")
    dependencies = extract_dependencies(higher_level_concepts)
    
    st.write("Converting dependencies to JSON...")
    dependency_dict = convert_dependencies_to_json(dependencies)
    
    save_dependency_graph(dependency_dict, "dependency_graph.json")
    
    st.write("Dependency graph saved as dependency_graph.json")
    
    st.write("\nCore Concepts:")
    for concept in sorted(concepts):
        st.write(f"- {concept}")
    
    st.write("\nHigher-level Concepts:")
    for concept in higher_level_concepts:
        st.write(f"- {concept}")
    
    st.write("\nConcept Dependencies:")
    for dependency in dependencies:
        st.write(f"- {dependency}")
    
    # Add this after saving the dependency graph
    st.write("### Concept Dependency Map")
    display_concept_map(dependency_dict)
    
    # Initialize RAG components using the uploaded PDF
    retriever = initialize_rag_components("uploaded_file.pdf")
    llm = ChatOpenAI(model="gpt-4", temperature=0.1)
    
    # Generate content for all concepts using the selected learning strategy
    st.write("\nStarting content generation for all learning styles...")
    
    if st.button("Generate Content"):
        generated_content = {}
        progress_bar = st.progress(0)
        status_container = st.empty()
        current_status = st.empty()
        content_display = st.empty()
        
        total_tasks = len(dependency_dict.keys()) * len(learning_styles.keys())
        current_task = 0
        
        # Initialize the formatted content structure
        formatted_content = {concept: {} for concept in dependency_dict.keys()}
        
        # Create metrics containers
        col1, col2, col3 = st.columns(3)
        with col1:
            topic_metric = st.empty()
        with col2:
            strategy_metric = st.empty()
        with col3:
            progress_metric = st.empty()
            
        # Create tabs for each learning style
        tabs = st.tabs(list(learning_styles.keys()))
        
        for concept in dependency_dict.keys():
            for idx, (strategy, steps) in enumerate(learning_styles.items()):
                # Update metrics
                topic_metric.metric("Current Topic", concept)
                strategy_metric.metric("Current Strategy", strategy)
                progress_metric.metric("Progress", f"{current_task}/{total_tasks}")
                
                status_container.write(f"""
                ### Current Generation Status:
                - 📚 Topic: **{concept}**
                - 🎯 Learning Style: **{strategy}**
                - 📊 Progress: {current_task}/{total_tasks} tasks completed
                """)
                
                # Generate content
                content = generate_content_for_concept(concept, strategy, retriever, llm)
                
                # Store in formatted structure
                formatted_content[concept][strategy] = content
                
                # Update progress
                current_task += 1
                progress = current_task / total_tasks
                progress_bar.progress(progress)
                
                # Display in appropriate tab
                with tabs[idx]:
                    st.write(f"""
                    ### {concept} ({strategy})
                    {content}
                    """)
                
                # Save continuously
                with open('formatted_content.json', 'w') as f:
                    json.dump(formatted_content, f, indent=2)
        
        status_container.write("✅ Content generation complete!")
        topic_metric.empty()
        strategy_metric.empty()
        progress_metric.empty()
        
        # Show visualization
        st.write("### View All Generated Content")
        if st.button("Open Content Viewer"):
            import viz_content
            viz_content.show_content()

# Clear all temporary files
for file in os.listdir():
    if file.startswith("temp_"):
        os.remove(file)

# Add this at the end of your script to display logs in Streamlit
if st.checkbox("Show logs"):
    st.text(open("streamlit_app.log").read())
