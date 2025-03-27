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

COMPARE_TEMPLATE = """
You are an expert mobile device specialist with deep knowledge of smartphones, their specifications, market trends, and user experiences. Your expertise covers:

1. Detailed Technical Analysis:
- Display technology (OLED, LCD, refresh rates, brightness, resolution)
- Processing capabilities (chipset benchmarks, real-world performance)
- Camera systems (sensor specs, computational photography features)
- Battery technology (capacity, charging speeds, battery life)
- Build quality and materials
- Storage and memory configurations
- Connectivity features (5G bands, WiFi standards)
- Special features and unique selling points

2. Comparative Analysis:
- Price-to-performance ratio
- Feature comparison across different models
- Market positioning and target audience
- Historical context and generational improvements

3. User Experience Insights:
- Real-world performance analysis
- Long-term reliability factors
- Software update policies and support
- Common issues and solutions

4. Market Context:
- Release timing and availability
- Price trends and value retention
- Competition in the same segment
- Regional variations and availability

Use the provided context to give detailed, expert-level responses about mobile devices.
If comparing phones, highlight key differentiating factors and make clear recommendations based on use cases.
If analyzing a single phone, provide comprehensive insights about its position in the market and its strengths/weaknesses.

Context information: {context}

Question: {question}

Analysis:

[Provide your detailed analysis here]

When comparing multiple phones, ALWAYS end your response with a structured comparison table that summarizes key specifications and features. Format the table like this:

| Feature | Phone Model 1 | Phone Model 2 | ... |
|---------|--------------|--------------|-----|
| Display | Display specs | Display specs | ... |
| Processor | Processor specs | Processor specs | ... |
| Camera | Camera specs | Camera specs | ... |
| Battery | Battery specs | Battery specs | ... |
| Storage | Storage options | Storage options | ... |
| Price | Price range | Price range | ... |
| Verdict | Brief verdict | Brief verdict | ... |

Include at least 8-10 key comparison points in the table. For the verdict row, provide a one-sentence summary of each phone's strengths.
"""


def initialize_llm():
    """Initialize the model through Ollama."""
    print("🦙 Initializing model through Ollama...")
    try:
        llm = OllamaLLM(model="qwen2.5", base_url="http://localhost:11434")
        return llm
    except Exception as e:
        print(f"❌ Error initializing LLM: {e}")
        print("Make sure Ollama is running and selected model is installed.")
        sys.exit(1)


def initialize_tavily():
    """Initialize Tavily search API for web searches."""
    print("🔍 Initializing Tavily search...")
    try:
        tavily_search = TavilySearchResults()
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


@lru_cache(maxsize=1)
def get_embeddings():
    """Get cached embeddings model."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )


def create_vector_store(chunks):
    if not chunks:
        return None

    print("🧠 Creating vector embeddings with Chroma...")
    try:
        embeddings = get_embeddings()
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
    if not vector_store:
        return None

    template = COMPARE_TEMPLATE

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


def analyze_query_type(query: str, llm) -> dict:
    """Analyze the type of phone-related query and extract relevant information."""
    analysis_prompt = f"""
    Analyze this phone-related query: "{query}"
    Determine:
    1. Phone models mentioned (comma-separated)
    2. Specific aspects to focus on (if any)

    Return a JSON object with these keys:
    {{"type": "specs|compare|info", "models": ["model1", "model2"], "focus": "specific aspects"}}

    If multiple phone models are mentioned, always set type to "compare".
    Extract exact phone model names accurately.
    """

    try:
        response = llm.invoke(analysis_prompt)
        import json
        return json.loads(response)
    except Exception as e:
        print(f"⚠️ Error analyzing query type: {e}")
        return {"type": "specs", "models": [query], "focus": "general"}


def search_phone_specs(query: str, tavily_search, llm, max_sources=4):
    """Search for phone specifications using Tavily."""
    print(f"🔍 Analyzing query: '{query}'...")
    try:
        query_info = analyze_query_type(query, llm)

        tavily_search.max_results = max_sources

        all_results = []
        for model in query_info["models"]:
            search_terms = [
                model,
                "phone",
                query_info["type"] if query_info["type"] != "specs" else "specifications",
                query_info["focus"] if query_info["focus"] != "general" else "review"
            ]
            search_query = " ".join(search_terms)

            results = tavily_search.invoke(search_query)
            all_results.extend(results)

        return all_results, query_info
    except Exception as e:
        print(f"❌ Error during search: {e}")
        return [], {"type": "specs", "models": [query], "focus": "general"}


def analyze_phones(query: str, llm, tavily_search, max_sources=4, return_sources=False):
    """Analyze phones and provide detailed comparison."""
    search_results, query_info = search_phone_specs(query, tavily_search, llm, max_sources)

    if not search_results:
        return {"result": "No search results found for the query.", "sources": []}

    all_documents = []
    sources = []
    total_results = len(search_results)
    print(f"📱 Processing {total_results} search results...")

    # Replace sequential processing with parallel processing
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
        return {"result": "Could not retrieve any useful content from the search results.", "sources": sources}

    print("✂️ Processing and analyzing content...")
    chunks = process_documents(all_documents)
    if not chunks:
        return {"result": "Failed to process the documents.", "sources": sources}

    print("🧠 Creating knowledge base...")
    vector_store = create_vector_store(chunks)
    if not vector_store:
        return {"result": "Failed to create vector store from the documents.", "sources": sources}

    print("🔗 Creating RAG pipeline...")
    qa_chain = setup_rag_pipeline(llm, vector_store)
    if not qa_chain:
        return {"result": "Failed to set up the processing pipeline.", "sources": sources}

    print(f"📱 Generating {query_info['type']} analysis...")
    try:
        if query_info['type'] == 'compare':
            analysis_query = f"""Compare these phones in detail: {', '.join(query_info['models'])}.
            Focus on {query_info['focus'] if query_info['focus'] != 'general' else 'all major aspects'}.
            Include specific recommendations based on different use cases."""
        elif query_info['type'] == 'info':
            analysis_query = f"""Provide detailed information about {query_info['models'][0]},
            focusing on {query_info['focus'] if query_info['focus'] != 'general' else 'all aspects'}.
            Include market context and user experience insights."""
        else:  # specs
            analysis_query = f"""What are the detailed specifications and features of {query_info['models'][0]}?
            {f"Focus especially on {query_info['focus']}." if query_info['focus'] != 'general' else ''}"""

        result = qa_chain.invoke({"query": analysis_query})
        if 'result' in result:
            return {"result": result['result'], "sources": sources}
    except Exception as e:
        return {"result": f"Error generating analysis: {e}", "sources": sources}

    return {"result": "Could not generate the requested analysis.", "sources": sources}


async def stream_compare_generator(query: str, llm, tavily_search, max_sources=4) -> AsyncGenerator[str, None]:
    """Generator function for streaming phone comparison with real-time LLM output."""
    try:
        search_results, query_info = search_phone_specs(query, tavily_search, llm, max_sources)

        if not search_results:
            yield json.dumps({"status": "error", "message": "No search results found for the query."}) + "\n"
            return

        all_documents = []
        sources = []
        total_results = len(search_results)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_url = {}
            for i, result in enumerate(search_results):
                if 'url' in result:
                    sources.append(result['url'])
                    future_to_url[executor.submit(scrape_website, result['url'])] = result['url']

            for future in concurrent.futures.as_completed(future_to_url):
                documents = future.result()
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

        # Stream the LLM response
        try:
            # Determine the appropriate query based on query type
            if query_info['type'] == 'compare':
                analysis_query = f"""Compare these phones in detail: {', '.join(query_info['models'])}.
                Focus on {query_info['focus'] if query_info['focus'] != 'general' else 'all major aspects'}.
                Include specific recommendations based on different use cases."""
            elif query_info['type'] == 'info':
                analysis_query = f"""Provide detailed information about {query_info['models'][0]},
                focusing on {query_info['focus'] if query_info['focus'] != 'general' else 'all aspects'}.
                Include market context and user experience insights."""
            else:  # specs
                analysis_query = f"""What are the detailed specifications and features of {query_info['models'][0]}?
                {f"Focus especially on {query_info['focus']}." if query_info['focus'] != 'general' else ''}"""

            # Start a partial response object
            partial_response = {
                "status": "generating",
                "query": query,
                "sources": sources,
                "comparison": "",
                "phones": query_info['models'],
                "focus": query_info['focus'] if query_info['focus'] != 'general' else "all aspects"
            }

            retriever = vector_store.as_retriever(search_kwargs={'k': 5})
            docs = retriever.get_relevant_documents(analysis_query)

            context_text = "\n\n".join([doc.page_content for doc in docs])

            template = COMPARE_TEMPLATE

            prompt = PromptTemplate(
                input_variables=["context", "question"],
                template=template,
            )

            formatted_prompt = prompt.format(context=context_text, question=analysis_query)

            # Stream the response from the LLM
            for chunk in llm.stream(formatted_prompt):
                partial_response["comparison"] += chunk

                # Send the updated partial response
                yield json.dumps(partial_response) + "\n"

            # Send the final complete response
            final_response = {
                "status": "complete",
                "comparison": partial_response["comparison"],
                "query": query,
                "sources": sources,
                "phones": query_info['models'],
                "focus": query_info['focus'] if query_info['focus'] != 'general' else "all aspects"
            }
            yield json.dumps(final_response) + "\n"

        except Exception as e:
            yield json.dumps({"status": "error", "message": f"Error generating comparison: {str(e)}"}) + "\n"

    except Exception as e:
        yield json.dumps({"status": "error", "message": str(e)}) + "\n"
