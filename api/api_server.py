"""FastAPI server for ARA Bot"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from pathlib import Path
from datetime import datetime
import json

from config import API_CONFIG, RESULTS_DIR, CHARTS_DIR
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Import main scanner (will be imported in main)
scanner_instance = None

app = FastAPI(title="ARA Bot API", version="2.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=API_CONFIG["CORS_ORIGINS"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ScanRequest(BaseModel):
    tickers: Optional[List[str]] = None
    use_cache: bool = True

class ScanResponse(BaseModel):
    success: bool
    message: str
    results_count: int = 0

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "ARA Bot API",
        "version": "2.0",
        "status": "running"
    }

@app.post("/scan", response_model=ScanResponse)
async def scan_tickers(request: ScanRequest, background_tasks: BackgroundTasks):
    """
    Scan tickers and return results
    
    Args:
        request: Scan request with optional ticker list
        background_tasks: Background tasks
    
    Returns:
        Scan response
    """
    try:
        # This would trigger the main scanner
        # For now, return placeholder
        return ScanResponse(
            success=True,
            message="Scan initiated",
            results_count=0
        )
    except Exception as e:
        logger.error(f"Error in scan endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/scan/ticker/{ticker}")
async def scan_single_ticker(ticker: str):
    """
    Scan single ticker
    
    Args:
        ticker: Ticker symbol
    
    Returns:
        Result for single ticker
    """
    try:
        # Placeholder - would call scanner for single ticker
        return {
            "ticker": ticker,
            "signal": "NONE",
            "score": 0.0
        }
    except Exception as e:
        logger.error(f"Error scanning {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/results/latest")
async def get_latest_results():
    """
    Get latest scan results
    
    Returns:
        Latest results
    """
    try:
        results_file = RESULTS_DIR / "ara_latest.json"
        if not results_file.exists():
            return {"results": [], "count": 0}
        
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        return data
    except Exception as e:
        logger.error(f"Error getting latest results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/charts/{ticker}")
async def get_chart(ticker: str):
    """
    Get chart for ticker
    
    Args:
        ticker: Ticker symbol
    
    Returns:
        Chart image
    """
    try:
        # Find chart file
        chart_files = list(CHARTS_DIR.glob(f"{ticker}_*.png"))
        if not chart_files:
            raise HTTPException(status_code=404, detail="Chart not found")
        
        # Return most recent
        chart_file = sorted(chart_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
        return FileResponse(chart_file)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chart for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """
    Simple HTML dashboard
    
    Returns:
        HTML dashboard
    """
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ARA Bot Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #4CAF50; color: white; }
            tr:nth-child(even) { background-color: #f2f2f2; }
        </style>
    </head>
    <body>
        <h1>ARA Bot Dashboard</h1>
        <p>Latest scan results will appear here</p>
        <div id="results"></div>
        <script>
            fetch('/results/latest')
                .then(r => r.json())
                .then(data => {
                    const div = document.getElementById('results');
                    if (data.results && data.results.length > 0) {
                        let html = '<table><tr><th>Ticker</th><th>Signal</th><th>Score</th><th>Price</th></tr>';
                        data.results.forEach(r => {
                            html += `<tr><td>${r.ticker}</td><td>${r.signal}</td><td>${r.score}</td><td>${r.latest_price}</td></tr>`;
                        });
                        html += '</table>';
                        div.innerHTML = html;
                    } else {
                        div.innerHTML = '<p>No results available</p>';
                    }
                });
        </script>
    </body>
    </html>
    """
    return html

def create_app():
    """Create and configure FastAPI app"""
    return app

