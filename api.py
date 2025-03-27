from contextlib import asynccontextmanager
from typing import Optional, List

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from compareSearch import analyze_phones, stream_compare_generator
from specsSearch import (
    initialize_llm,
    initialize_tavily,
    get_phone_specs,
    stream_specs_generator
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm, tavily_search
    llm = initialize_llm()
    tavily_search = initialize_tavily()
    yield


app = FastAPI(
    title="Phone Information API",
    description="API for retrieving detailed phone specifications and comparing phones",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class SpecsRequest(BaseModel):
    query: str


class SpecsResponse(BaseModel):
    specifications: str
    query: str
    sources: List[str] = []


class CompareRequest(BaseModel):
    phones: List[str] = []
    focus: Optional[str] = "everything"
    max_sources: Optional[int] = 4

    class Config:
        json_schema_extra = {
            "example": {
                "phones": ["iPhone 15 Pro", "Samsung Galaxy S24"],
                "focus": "camera",
                "max_sources": 4
            }
        }


class CompareResponse(BaseModel):
    comparison: str
    phones: List[str]
    focus: str
    sources: List[str] = []


@app.post("/specs", response_model=SpecsResponse)
async def get_phone_specifications(request: SpecsRequest):
    """
    Get detailed specifications for a specific phone.

    Example request:
    {
        "query": "iPhone 15 Pro"
    }
    """
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        result, sources = get_phone_specs(request.query, llm, tavily_search, return_sources=True)
        if not result or result.startswith("Error"):
            raise HTTPException(status_code=500, detail=result)

        return SpecsResponse(
            specifications=result,
            query=request.query,
            sources=sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/specs/stream")
async def stream_phone_specifications(request: SpecsRequest):
    """
    Stream detailed specifications for a specific phone with real-time LLM output.
    
    This endpoint returns a stream of JSON objects, each representing a step in the process.
    During LLM generation, you'll receive partial responses with the "generating" status.
    The final object contains the complete specifications with "complete" status.
    
    Example request:
    {
        "query": "iPhone 15 Pro"
    }
    """
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    return StreamingResponse(
        stream_specs_generator(request.query, llm, tavily_search),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/compare", response_model=CompareResponse)
async def compare_phones(request: CompareRequest):
    """
    Compare multiple phones and get detailed analysis.
    
    Example request:
    {
        "phones": ["iPhone 15 Pro", "Samsung Galaxy S24"],
        "focus": "camera",
        "max_sources": 4
    }
    """
    if not request.phones or len(request.phones) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least two phones must be provided for comparison"
        )

    try:
        comparison_query = f"compare {' vs '.join(request.phones)}"
        if request.focus and request.focus != "general":
            comparison_query += f" focusing on {request.focus}"

        result = analyze_phones(comparison_query, llm, tavily_search, max_sources=request.max_sources)
        if not result or (isinstance(result, dict) and result.get("result", "").startswith("Error")):
            raise HTTPException(status_code=500, detail=str(result))

        return CompareResponse(
            comparison=result["result"] if isinstance(result, dict) else result,
            phones=request.phones,
            focus=request.focus,
            sources=result["sources"] if isinstance(result, dict) and "sources" in result else []
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare/stream")
async def stream_phone_comparison(request: CompareRequest):
    """
    Stream detailed comparison of multiple phones with real-time LLM output.
    
    This endpoint returns a stream of JSON objects, each representing a step in the process.
    During LLM generation, you'll receive partial responses with the "generating" status.
    The final object contains the complete comparison with "complete" status.
    
    Example request:
    {
        "phones": ["iPhone 15 Pro", "Samsung Galaxy S24"],
        "focus": "camera",
        "max_sources": 4
    }
    """
    if not request.phones or len(request.phones) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least two phones must be provided for comparison"
        )

    try:
        comparison_query = f"compare {' vs '.join(request.phones)}"
        if request.focus and request.focus != "general":
            comparison_query += f" focusing on {request.focus}"

        return StreamingResponse(
            stream_compare_generator(comparison_query, llm, tavily_search, max_sources=request.max_sources),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """
    Check if the API is running and services are initialized
    """
    return {
        "status": "healthy",
        "services": {
            "llm": llm is not None,
            "tavily_search": tavily_search is not None
        },
        "endpoints": {
            "specs": "/specs - Get detailed phone specifications",
            "specs/stream": "/specs/stream - Stream detailed phone specifications",
            "compare": "/compare - Compare multiple phones",
            "compare/stream": "/compare/stream - Stream detailed phone comparison"
        }
    }


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8080, reload=True)
