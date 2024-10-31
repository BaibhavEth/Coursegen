import streamlit as st
import json

def show_content():
    try:
        # Load the JSON data
        with open('formatted_content.json', 'r') as f:
            data = json.load(f)

        # Extract all unique learning strategies and topics
        learning_strategies = set()
        topics = list(data.keys())
        for topic in data:
            for strategy in data[topic]:
                learning_strategies.add(strategy)

        # Convert set to list for Streamlit selectbox
        learning_strategies = list(learning_strategies)

        # Streamlit UI
        st.title("Learning Content Viewer")

        # Select a topic
        selected_topic = st.selectbox("Select a topic", topics)

        # Select a learning strategy
        selected_strategy = st.selectbox("Select a learning strategy", learning_strategies)

        # Display content for the selected strategy and topic
        if selected_topic in data and selected_strategy in data[selected_topic]:
            st.header(selected_topic)
            st.write(data[selected_topic][selected_strategy])
        else:
            st.write("No content available for the selected strategy and topic.")

    except FileNotFoundError:
        st.error("No content has been generated yet. Please generate content first.")

if __name__ == "__main__":
    show_content()
