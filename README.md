# PhoneSearchAI

A powerful API for retrieving detailed smartphone specifications and comparing different phone models using AI-powered search and analysis.

## Overview

PhoneSearchAI is a FastAPI-based application that leverages Large Language Models (LLMs) and vector search to provide detailed information about smartphones. The system uses Ollama for local LLM inference and Tavily Search API for retrieving up-to-date information from the web.

## Features

- **Detailed Phone Specifications**: Get comprehensive technical details about any smartphone model
- **Phone Comparison**: Compare multiple phones with detailed analysis of their differences
- **Focus-Based Analysis**: Request information about specific aspects (camera, display, performance, etc.)
- **Real-time Streaming**: Stream responses as they're being generated for a better user experience
- **Source Attribution**: All information comes with source URLs for verification

## Requirements

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running locally
- Tavily API key

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/phoneSearchAI.git
   cd phoneSearchAI
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Add your Tavily API key to the `.env` file

4. Make sure Ollama is running with the required models:
   ```bash
   # For specs search
   ollama pull llama3.1
   
   # For comparison search
   ollama pull qwen2.5
   ```

## Usage

Start the API server:

```bash
python api.py
```

The API will be available at `http://localhost:8080`.

## API Endpoints

### Get Phone Specifications

```http
POST /specs
```

Request body:
```json
{
  "query": "iPhone 15 Pro"
}
```

Response:
```json
{
  "specifications": "Detailed specifications in formatted text",
  "query": "iPhone 15 Pro",
  "sources": ["https://example.com/source1", "https://example.com/source2"]
}
```

### Stream Phone Specifications

```http
POST /specs/stream
```

Request body:
```json
{
  "query": "iPhone 15 Pro"
}
```

Returns a stream of JSON objects with real-time updates.

### Compare Phones

```http
POST /compare
```

Request body:
```json
{
  "phones": ["iPhone 15 Pro", "Samsung Galaxy S24"],
  "focus": "camera",
  "max_sources": 4
}
```

Response:
```json
{
  "comparison": "Detailed comparison with analysis and comparison table",
  "phones": ["iPhone 15 Pro", "Samsung Galaxy S24"],
  "focus": "camera",
  "sources": ["https://example.com/source1", "https://example.com/source2"]
}
```

### Stream Phone Comparison

```http
POST /compare/stream
```

Request body:
```json
{
  "phones": ["iPhone 15 Pro", "Samsung Galaxy S24"],
  "focus": "camera",
  "max_sources": 4
}
```

Returns a stream of JSON objects with real-time updates.

### Health Check

```http
GET /health
```

Returns the status of the API and its services.

## How It Works

1. **Web Search**: The system uses Tavily Search API to find relevant information about the requested phones
2. **Content Scraping**: It scrapes the content from the search results
3. **Vector Embedding**: The content is split into chunks and embedded into a vector store
4. **RAG Pipeline**: A Retrieval-Augmented Generation pipeline retrieves relevant information and passes it to the LLM
5. **LLM Processing**: The LLM generates a detailed response based on the retrieved information

## Architecture

- `api.py`: FastAPI application with endpoint definitions
- `specsSearch.py`: Logic for retrieving and processing phone specifications
- `compareSearch.py`: Logic for comparing multiple phone models

## License

This project is licensed under the terms of the license included in the repository.