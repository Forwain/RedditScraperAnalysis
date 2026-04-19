# Reddit Scraper & Analysis


## Requirements
- Python 3.8+
- ffmpeg (optional, for video with audio)

## Scraping
    ## Command-Line Arguments

    ### Scraping Options
    - `target` (positional): Subreddit or username to scrape
    - `--mode`: Choices: `monitor`, `history`, `full` (default: `full`)
    - `--user`: Target is a user
    - `--limit`: Max posts to scrape (default: 100)
    - `--no-media`: Skip media download
    - `--no-comments`: Skip comments

    ### Dashboard
    - `--dashboard`: Launch web dashboard

    ### Search
    - `--search`: Search scraped data
    - `--subreddit`: Filter by subreddit
    - `--min-score`: Filter by minimum score
    - `--author`: Filter by author

    ### Analytics
    - `--analyze`: Run analytics on subreddit
    - `--sentiment`: Run sentiment analysis
    - `--keywords`: Extract keywords

    ### Schedule
    - `--schedule`: Schedule scraping for target
    - `--every`: Interval in minutes

    ### Alerts
    - `--alert`: Set keyword alert
    - `--discord-webhook`: Discord webhook URL
    - `--telegram-token`: Telegram bot token
    - `--telegram-chat`: Telegram chat ID

    ### Observability & Maintenance
    - `--dry-run`: Simulate scrape without saving data
    - `--plugins`: Enable post-processing plugins
    - `--list-plugins`: List available plugins
    - `--job-history`: View job history
    - `--backup`: Backup SQLite database
    - `--vacuum`: Optimize SQLite database
    - `--export-parquet`: Export subreddit to Parquet format
    - `--api`: Start REST API server (port 8000)

## Analysis

### Sentiment Analysis
```
python main.py --analyze delhi --sentiment
```

This project implements a basic sentiment analysis system to classify text as positive, negative, or neutral.

The current implementation is lexicon-based:

A predefined sentiment dictionary (word list) is used.
The input text is tokenized and matched against the lexicon.
Each matched word contributes to a cumulative sentiment score.
The final score is used to determine the overall sentiment.
This approach is simple, interpretable, and does not require model training.

The system can be extended with more advanced techniques(RNNs, BERT, LLMs)

### Extract keywords
```
python main.py --analyze delhi --keywords
```

The current implementation uses a top-k based method:

The text is processed and candidate keywords are scored based on their importance. The top k highest-scoring terms are selected as keywords.

| Word | Count |
| --- | --- |
| python | 2610 |
| github | 1558 |
| project | 1159 |
| code | 744 |
| data | 600 |
| using | 515 |
| built | 496 |
| use | 443 |
| target | 415 |
| tool | 410 |
| audience | 387 |
| comparison | 379 |
| api | 375 |
| file | 337 |
| pypi | 324 |
| tools | 318 |
| based | 317 |
| source | 310 |
| type | 297 |
| thread | 293 |
