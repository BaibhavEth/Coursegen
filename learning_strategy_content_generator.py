import json
import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch  # Replace FAISS import
mport streamlit as st


# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
# Load the dependency graph
with open('dependency_graph.json', 'r') as f:
    dependency_graph = json.load(f)

# After loading the dependency graph
print("Dependency graph loaded.")

# Set up RAG components
loader = PyPDFLoader("College_Physics_for_AP_Courses_2e-WEB_DkhNbxV.pdf")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
#db = Chroma.from_documents(texts, embeddings)
#retriever = db.as_retriever()

# After setting up RAG components
print("RAG components initialized.")

# Set up LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

# After setting up LLM
print("LLM initialized.")

# Learning styles and their steps
learning_styles = {
    # All of the steps were pretty repetitive, no context on previous steps.
    "Apprentice": [
        "steps to approach this concept (context on the procedure)",
        "description of the concept",
        "examples of the concept",
        "Optional exposition - further explanation in a different way",
        "procedure - specific instructions to solve problems related to this concept",
        "practice quiz"
    ],
    # Only take one big case study and explain all the concept in context of that case study.
    "Incidental": [
        "Case Study - description of the case study",
        "event - example of the application of the concepts in the case study",
        "additional example - a different example than the case study to illustrate the concept",
        "Optional exposition - rehashing of the content in a different way",
        "practice quiz"
    ],
    # keep a summary of the principle. Each step should add to a collpasible pull-down, add matplotlib visualization for data
    "Inductive": [
        "principle - explain the core principle behind the concept",
        "example - summarize the principle again, and give an example of where the principle applies",
        "analysis - show how to manipulate the principle to analyse its applications",
        "Data/trend showing an example of the applications of the principles",
        "Optional exposition - rehashing of the content in a different way",
        "practice quiz"
    ],
    #add matplotlib visualization for data
    "Deductive": [
        "Data - show data with a trend showing the concept",
        "analysis - prompt the user to analyse the data themselves, which should lead them to the concept",
        "example - add more context about the data",
        "principle - extract the principle from the data",
        "Optional exposition - rehashing of the content in a different way",
        "practice quiz"
    ],
    #add matplotlib visualization for data
    # use pereplexity to find a simulation of the concept
    "Discovery": [
        "experiment - prompt the user to conduct an experiment",
        "data - Collect the data from the experiment",
        "analysis - find trends in the data",
        "principle - extract the principle from these trends",
        "practice quiz"
    ]
}

def initialize_rag_components(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = DocArrayInMemorySearch.from_documents(texts, embeddings)
    retriever = db.as_retriever()
    
    return retriever

def generate_content_for_concept(concept, learning_style, retriever, llm):
    # Retrieve relevant content using RAG
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    context = qa.run(f"Provide information about {concept}")
    
    # Initialize the content with the context
    content = ""
    
    for step in learning_styles[learning_style]:
        prompt = f"""
        Given the following context about {concept}:
        {context}
        
        And considering the following content generated so far:
        {content}
        
        Generate unique content for the "{step}" step of the {learning_style} learning style. Make sure to specifically only cover the content for the "{step}" step.
        Ensure the new content transitions smoothly from the previous content, without repeating information already covered in previous steps.
        """
        print(prompt)
        step_content = llm.predict(prompt)
        content += f"\n{step}:\n{step_content}\n"
    
    print(f"Generating content for {concept} using {learning_style} style...")
    return content

# Remove all the code that directly generates content and saves it
# The main script (streamlit_test.py) will call these functions as needed
