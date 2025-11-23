# ARA BOT V2 - Architecture Documentation

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        MAIN.PY                               │
│                    (Orchestrator)                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ DATA FETCH   │ │  INDICATORS  │ │   SCREENER   │
│              │ │              │ │              │
│ - Yahoo      │ │ - MA, RSI    │ │ - Scoring    │
│ - Cache      │ │ - BB, OBV    │ │ - Patterns   │
│ - Fallback   │ │ - ATR, VWAP  │ │ - Signals    │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       └────────────────┼────────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │    RANKING      │
              │                 │
              │ - ML Scoring    │
              │ - Sort & Filter │
              └────────┬────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   CHARTS     │ │  TELEGRAM    │ │    STATE     │
│              │ │              │ │              │
│ - Candles    │ │ - Notify     │ │ - Track      │
│ - Indicators │ │ - HTML       │ │ - Changes    │
│ - Entry/SL/TP│ │ - Summary    │ │ - History    │
└──────────────┘ └──────────────┘ └──────────────┘
```

## Data Flow

### 1. Data Fetching Pipeline

```
Ticker List
    │
    ▼
[DataFetcher]
    │
    ├─→ Check Cache (warm fetch)
    │   └─→ Return if valid
    │
    ├─→ Yahoo Finance API
    │   ├─→ Retry on failure
    │   └─→ Store in cache
    │
    └─→ Fallback APIs (if Yahoo fails)
        └─→ IDX API / EODHD / TradingView
```

### 2. Indicator Calculation

```
OHLCV DataFrame
    │
    ▼
[DataNormalizer]
    │
    ├─→ Fix multi-index
    ├─→ Remove outliers
    └─→ Validate data
    │
    ▼
[IndicatorEngine]
    │
    ├─→ Moving Averages (20, 50, 200)
    ├─→ RSI(14)
    ├─→ Bollinger Bands
    ├─→ OBV
    ├─→ VWAP
    ├─→ ATR(14)
    ├─→ RVOL
    └─→ MA20 Slope
```

### 3. Screening & Scoring

```
DataFrame with Indicators
    │
    ▼
[ScreenerEngine]
    │
    ├─→ Basic Filters
    │   ├─→ Price range
    │   └─→ Volume minimum
    │
    ├─→ Score Calculation
    │   ├─→ RVOL check (25%)
    │   ├─→ Bollinger Breakout (20%)
    │   ├─→ MA Structure (15%)
    │   ├─→ RSI Momentum (15%)
    │   ├─→ OBV Rising (10%)
    │   ├─→ VWAP Position (10%)
    │   └─→ MA20 Slope (5%)
    │
    ├─→ Pattern Detection
    │   ├─→ Parabolic
    │   ├─→ Volume Climax
    │   ├─→ VCP
    │   ├─→ Darvas Box
    │   ├─→ Pocket Pivot
    │   ├─→ Gap-Up
    │   └─→ Reaccumulation Base
    │
    └─→ Entry/Exit Calculation
        ├─→ Entry Zone
        ├─→ Stop Loss
        └─→ Take Profit (TP1, TP2)
```

### 4. Ranking & ML

```
Screener Results
    │
    ▼
[RankingEngine]
    │
    ├─→ ML Scoring (optional)
    │   └─→ LightGBM probability
    │
    ├─→ Combined Score
    │   └─→ Technical (70%) + ML (30%)
    │
    └─→ Sort & Filter
        └─→ Top N results
```

### 5. Output Generation

```
Ranked Results
    │
    ├─→ [ChartEngine]
    │   └─→ Generate PNG charts
    │
    ├─→ [StateManager]
    │   └─→ Track signal changes
    │
    ├─→ Save Results
    │   ├─→ JSON (latest)
    │   ├─→ CSV (latest)
    │   └─→ JSON (history)
    │
    └─→ [TelegramNotifier]
        ├─→ Send signal changes
        └─→ Send summary
```

## Module Responsibilities

### data_fetch/
- **YahooFetcher**: Fetch from Yahoo Finance with retry
- **FallbackFetcher**: Alternative API sources
- **DataFetcher**: Main fetcher with cache integration

### data_cleaner/
- **DataNormalizer**: Clean and normalize OHLCV data

### indicators/
- **IndicatorEngine**: Calculate all technical indicators

### screener/
- **ScreenerEngine**: Main screening logic and scoring
- **PatternDetector**: Advanced pattern detection

### ranking/
- **RankingEngine**: Sort and filter results
- **MLScorer**: ML-based probability prediction

### charts/
- **ChartEngine**: Generate professional candlestick charts

### notifier/
- **TelegramNotifier**: Send Telegram messages with HTML formatting

### state_manager/
- **StateManager**: Track signal changes to prevent spam

### cache_layer/
- **SQLiteCache**: SQLite-based data cache
- **CacheManager**: High-level cache management

### processing/
- **MultiprocessingEngine**: Parallel processing for multiple tickers

### api/
- **APIServer**: FastAPI REST endpoints

## Design Patterns

### 1. Strategy Pattern
- Different data sources (Yahoo, IDX, EODHD)
- Different pattern detectors
- Different ML models

### 2. Factory Pattern
- Indicator creation
- Chart style creation
- Notification formatter creation

### 3. Observer Pattern
- State changes trigger notifications
- Signal changes trigger Telegram messages

### 4. Singleton Pattern
- Logger instance
- Config instance
- Cache manager

## Performance Optimizations

### 1. Caching
- SQLite cache for OHLCV data
- TTL-based cache invalidation
- Warm fetch for faster scans

### 2. Parallel Processing
- ThreadPoolExecutor for I/O bound (data fetching)
- ProcessPoolExecutor for CPU bound (calculations)
- AsyncIO for batch operations

### 3. Memory Management
- Pandas slicing instead of copying
- Chunked processing for large ticker lists
- Garbage collection hints

### 4. Rate Limiting
- Delay between API calls
- Batch processing
- Cache-first strategy

## Error Handling

### 1. Data Fetch Errors
- Retry with exponential backoff
- Fallback to alternative APIs
- Use cached data if available

### 2. Calculation Errors
- Skip ticker on error
- Log error details
- Continue with next ticker

### 3. Notification Errors
- Log but don't fail scan
- Retry mechanism for Telegram

## Scalability

### Horizontal Scaling
- Multiple workers for parallel processing
- Distributed cache (Redis)
- Message queue for notifications

### Vertical Scaling
- Increase worker threads
- Larger cache size
- More memory for pandas operations

## Security Considerations

### 1. API Keys
- Environment variables
- Never commit to git
- Rotate regularly

### 2. Data Privacy
- Local storage by default
- Encrypt sensitive data
- Secure API endpoints

### 3. Rate Limiting
- Respect API limits
- Implement backoff
- Monitor usage

## Future Enhancements

### 1. Real-time Streaming
- WebSocket for live updates
- Real-time data feed
- Live signal alerts

### 2. Advanced ML
- Deep learning models
- Reinforcement learning
- Ensemble methods

### 3. Backtesting
- Historical strategy testing
- Parameter optimization
- Performance metrics

### 4. Database Integration
- PostgreSQL for production
- Time-series database
- Advanced queries

### 5. Web Dashboard
- Real-time charts
- Interactive filters
- Custom alerts

