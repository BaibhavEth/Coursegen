import os
import json
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
def process_pdf(file_path):
    print("Processing PDF...")
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = DocArrayInMemorySearch.from_documents(texts, embeddings)

    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    retriever = vectorstore.as_retriever()

    concept_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are an expert in the given topic. Your task is to extract the main concepts of the given text, in one phrase.
        
        Context: {context}
        
        Question: {question}
        
        Please respond with the main concepts in one phrase.
        """
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": concept_prompt}
    )

    concepts = set()
    total_chunks = len(texts)
    for i in range(0, total_chunks, 5):
        batch = texts[i:i+5]
        batch_text = " ".join([doc.page_content for doc in batch])
        result = qa_chain.run(f"Extract core concepts from this text: {batch_text}")
        batch_concepts = [concept.strip() for concept in result.split('\n') if concept.strip()]
        concepts.update(batch_concepts)

    return concepts

def extract_higher_level_concepts(concepts):
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    higher_level_prompt = PromptTemplate(
        input_variables=["concepts"],
        template="""
        Identify the most fundamental concepts that are repeatedly mentioned in the following text and also try to break it down into smaller concepts For example,dont do something like "work and energy", "work" and "energy" should be separate concepts. Focus on basic principles, not specific applications, analyses or examples:

        Text: {concepts}
        """
    )

    higher_level_chain = LLMChain(llm=llm, prompt=higher_level_prompt)

    concepts_text = "\n".join(sorted(concepts))
    higher_level_result = higher_level_chain.run(concepts=concepts_text)

    higher_level_concepts = [concept.strip() for concept in higher_level_result.split('\n') if concept.strip()]

    return higher_level_concepts

def extract_dependencies(higher_level_concepts):
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    dependency_prompt = PromptTemplate(
        input_variables=["concepts"],
        template="""
        Given the following list of fundamental concepts, identify the dependencies between them. For each concept, list the concepts it depends on. If a concept does not depend on any other concept, state "None".

        Concepts: {concepts}
        """
    )

    dependency_chain = LLMChain(llm=llm, prompt=dependency_prompt)

    concepts_text = "\n".join(higher_level_concepts)
    dependency_result = dependency_chain.run(concepts=concepts_text)

    dependencies = [line.strip() for line in dependency_result.split('\n') if line.strip()]

    return dependencies

def convert_dependencies_to_json(dependencies):
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    json_conversion_prompt = PromptTemplate(
        input_variables=["dependencies"],
        template="""
        Convert the following list of dependencies into a structured JSON format. Each dependency should be represented as a key-value pair where the key is the concept and the value is a list of concepts it depends on. Ensure the output is valid JSON with no extraneous characters or formatting such as quotations and the word json.

        Dependencies: {dependencies}
        """
    )

    json_conversion_chain = LLMChain(llm=llm, prompt=json_conversion_prompt)

    dependencies_text = "\n".join(dependencies)
    json_result = json_conversion_chain.run(dependencies=dependencies_text)

    json_result = json_result.lstrip('```json').rstrip('```').strip()

    try:
        dependency_dict = json.loads(json_result)
    except json.JSONDecodeError as e:
        dependency_dict = {}

    return dependency_dict

def save_dependency_graph(dependency_dict, file_path):
    with open(file_path, "w") as json_file:
        json.dump(dependency_dict, json_file, indent=4)
