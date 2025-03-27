import concurrent.futures
import json
import sys
import time
from functools import lru_cache
from typing import AsyncGenerator

import requests
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

load_dotenv()


def initialize_llm():
    """Initialize the model through Ollama."""
    print("🦙 Initializing model through Ollama...")
    try:
        llm = OllamaLLM(model="llama3.1", base_url="http://localhost:11434")
        return llm
    except Exception as e:
        print(f"❌ Error initializing LLM: {e}")
        print("Make sure Ollama is running and selected model is installed.")
        sys.exit(1)


def initialize_tavily():
    try:
        tavily_search = TavilySearchResults(max_results=3)
        return tavily_search
    except Exception as e:
        print(f"❌ Error initializing Tavily: {e}")
        sys.exit(1)


@lru_cache(maxsize=32)
def scrape_website(url: str):
    print(f"🕸️ Scraping {url}...")
    for _ in range(3):
        try:
            loader = WebBaseLoader(url)
            documents = loader.load()
            return documents
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Network error while scraping {url}: {e}")
            print("Retrying in 2 seconds...")
            time.sleep(2)
        except Exception as e:
            print(f"❌ Error scraping website {url}: {e}")
            return None
    print(f"❌ Failed to scrape {url} after multiple attempts")
    return None


def process_documents(documents):
    """Process and split the documents."""
    if not documents:
        return None

    print("✂️ Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


def create_vector_store(chunks):
    """Create vector store from document chunks."""
    if not chunks:
        return None

    print("🧠 Creating vector embeddings with Chroma...")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        persist_directory = "chroma_db"
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        return vector_store
    except Exception as e:
        print(f"❌ Error creating Chroma vector store: {e}")
        return None


def setup_rag_pipeline(llm, vector_store):
    """Set up RAG pipeline for phone specs."""
    if not vector_store:
        return None

    template = """
        You are an AI assistant specialized in extracting and summarizing phone specifications.
        Use the following information to provide a detailed summary of the phone specifications.

        Format your response following these rules:
        - Main headings should be between ** (e.g., **Display**)
        - Bullet points should use + instead of - (e.g., + 6.7-inch screen)

        Focus on key specifications like:
        + Display (size, resolution, technology)
        + Processor (chipset, CPU, GPU)
        + Camera (main camera, selfie camera, video capabilities)
        + Battery (capacity, charging speed)
        + Storage and RAM options
        + Connectivity (5G, WiFi, Bluetooth)
        + Operating System
        + Special features
        + Release date and price (if available)

        Present the information in a clear, organized manner with sections for each category.

        Context information: {context}

        Question: {question}

        Detailed phone specifications:
        """

    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={'k': 5}),
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
            return_source_documents=True
        )
        return qa_chain
    except Exception as e:
        print(f"❌ Error setting up RAG pipeline: {e}")
        return None


def search_phone_specs(query: str, tavily_search):
    """Search for phone specifications using Tavily."""
    print(f"🔍 Searching for '{query}'...")
    try:
        search_results = tavily_search.invoke(query + " phone specifications technical details")
        return search_results
    except Exception as e:
        print(f"❌ Error during search: {e}")
        return []

def get_phone_specs(query: str, llm, tavily_search, return_sources=False):
    """Main function to process user query."""
    print("🔍 Searching for phone specifications...")
    search_results = search_phone_specs(query, tavily_search)

    if not search_results:
        return (
            "No search results found for the query.",
            []) if return_sources else "No search results found for the query."

    all_documents = []
    total_results = len(search_results)
    print(f"📱 Found {total_results} relevant sources to analyze...")

    sources = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_url = {}
        for i, result in enumerate(search_results):
            if 'url' in result:
                print(f"🔄 Queuing source {i + 1}/{total_results}")
                sources.append(result['url'])
                future_to_url[executor.submit(scrape_website, result['url'])] = result['url']

        for future in concurrent.futures.as_completed(future_to_url):
            documents = future.result()
            if documents:
                all_documents.extend(documents)

    if not all_documents:
        return ("Could not retrieve any useful content from the search results.",
                sources) if return_sources else "Could not retrieve any useful content from the search results."

    print("✂️ Processing and analyzing content...")
    chunks = process_documents(all_documents)
    if not chunks:
        return ("Failed to process the documents.", sources) if return_sources else "Failed to process the documents."

    print("🧠 Creating knowledge base...")
    vector_store = create_vector_store(chunks)
    if not vector_store:
        return ("Failed to create vector store from the documents.",
                sources) if return_sources else "Failed to create vector store from the documents."

    print("🔗 Creating RAG pipeline...")
    qa_chain = setup_rag_pipeline(llm, vector_store)
    if not qa_chain:
        return ("Failed to set up the processing pipeline.",
                sources) if return_sources else "Failed to set up the processing pipeline."

    print("📱 Generating detailed specifications...")
    try:
        result = qa_chain.invoke({"query": f"What are the detailed specifications of {query}?"})
        if 'result' in result:
            return (result['result'], sources) if return_sources else result['result']
    except Exception as e:
        error_msg = f"Error generating summary: {e}"
        return (error_msg, sources) if return_sources else error_msg

    error_msg = "Could not generate a summary of the phone specifications."
    return (error_msg, sources) if return_sources else error_msg

async def stream_specs_generator(query: str, llm, tavily_search) -> AsyncGenerator[str, None]:
    """Generator function for streaming phone specifications with real-time LLM output."""
    try:
        search_results = search_phone_specs(query, tavily_search)

        if not search_results:
            yield json.dumps({"status": "error", "message": "No search results found for the query."}) + "\n"
            return

        all_documents = []
        sources = []
        # total_results = len(search_results)

        for i, result in enumerate(search_results):
            if 'url' in result:
                sources.append(result['url'])
                documents = scrape_website(result['url'])
                if documents:
                    all_documents.extend(documents)

        if not all_documents:
            yield json.dumps(
                {"status": "error", "message": "Could not retrieve any useful content from the search results."}) + "\n"
            return

        chunks = process_documents(all_documents)
        if not chunks:
            yield json.dumps({"status": "error", "message": "Failed to process the documents."}) + "\n"
            return

        vector_store = create_vector_store(chunks)
        if not vector_store:
            yield json.dumps({"status": "error", "message": "Failed to create vector store from the documents."}) + "\n"
            return

        qa_chain = setup_rag_pipeline(llm, vector_store)
        if not qa_chain:
            yield json.dumps({"status": "error", "message": "Failed to set up the processing pipeline."}) + "\n"
            return

        try:
            query_text = f"What are the detailed specifications of {query}?"

            partial_response = {
                "status": "generating",
                "query": query,
                "sources": sources,
                "specifications": ""
            }

            retriever = vector_store.as_retriever(search_kwargs={'k': 5})
            docs = retriever.get_relevant_documents(query_text)

            template = """
            You are an AI assistant specialized in extracting and summarizing phone specifications.
            Use the following information to provide a detailed summary of the phone specifications.

            Format your response following these rules:
            - Main headings should be between ** (e.g., **Display**)
            - Bullet points should use + instead of - (e.g., + 6.7-inch screen)

            Focus on key specifications like:
            + Display (size, resolution, technology)
            + Processor (chipset, CPU, GPU)
            + Camera (main camera, selfie camera, video capabilities)
            + Battery (capacity, charging speed)
            + Storage and RAM options
            + Connectivity (5G, WiFi, Bluetooth)
            + Operating System
            + Special features
            + Release date and price (if available)

            Present the information in a clear, organized manner with sections for each category.

            Context information: {context}

            Question: {question}

            Detailed phone specifications:
            """

            prompt = PromptTemplate(
                input_variables=["context", "question"],
                template=template,
            )

            context_text = "\n\n".join([doc.page_content for doc in docs])
            formatted_prompt = prompt.format(context=context_text, question=query_text)

            # Stream the response from the LLM
            for chunk in llm.stream(formatted_prompt):
                partial_response["specifications"] += chunk

                # Send the updated partial response
                yield json.dumps(partial_response) + "\n"

            # Send the final complete response
            final_response = {
                "status": "complete",
                "specifications": partial_response["specifications"],
                "query": query,
                "sources": sources
            }
            yield json.dumps(final_response) + "\n"

        except Exception as e:
            yield json.dumps({"status": "error", "message": f"Error generating specifications: {str(e)}"}) + "\n"

    except Exception as e:
        yield json.dumps({"status": "error", "message": str(e)}) + "\n"