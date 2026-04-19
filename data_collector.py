import argparse
import json
import os
import time
from datetime import date, datetime
from pathlib import Path

from dotenv import load_dotenv
import requests

load_dotenv(dotenv_path=Path(__file__).parent / '.env')

API_URL = 'https://newsapi.org/v2/everything'

def get_api_key(provided_key: str = None) -> str:
    api_key = provided_key or os.getenv('NEWSAPI_KEY')
    if not api_key:
        raise ValueError('NEWSAPI_KEY is required. Set it in environment variables or pass --api-key.')
    return api_key


def build_query(topic: str, from_date: str, page: int, page_size: int, language: str, sort_by: str) -> dict:
    return {
        'q': topic,
        'from': from_date,
        'sortBy': sort_by,
        'language': language,
        'pageSize': page_size,
        'page': page,
    }


def fetch_page(api_key: str, topic: str, from_date: str, page: int, page_size: int, language: str, sort_by: str) -> dict:
    params = build_query(topic, from_date, page, page_size, language, sort_by)
    headers = {'Authorization': api_key}
    response = requests.get(API_URL, params=params, headers=headers, timeout=20)
    response.raise_for_status()
    data = response.json()
    if data.get('status') != 'ok':
        raise RuntimeError(f"NewsAPI error: {data.get('code', 'unknown')} {data.get('message')}")
    return data


def collect_news(topic: str, from_date: str, max_pages: int, page_size: int, language: str, sort_by: str, delay: float, api_key: str) -> dict:
    all_articles = []
    total_results = None

    for page in range(1, max_pages + 1):
        print(f'Fetching page {page} for topic "{topic}"...')
        data = fetch_page(api_key, topic, from_date, page, page_size, language, sort_by)
        if total_results is None:
            total_results = data.get('totalResults', 0)
        page_articles = data.get('articles', [])
        all_articles.extend(page_articles)
        if len(page_articles) < page_size:
            break
        if len(all_articles) >= total_results:
            break
        time.sleep(delay)

    return {
        'status': 'ok',
        'topic': topic,
        'from': from_date,
        'collected_at': datetime.utcnow().isoformat() + 'Z',
        'total_results': total_results,
        'collected_count': len(all_articles),
        'articles': all_articles,
    }


def save_raw_json(payload: dict, topic: str, output_dir: Path) -> Path:
    timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    safe_topic = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in topic)
    filename = f'news_{safe_topic}_{date.today().isoformat()}_{timestamp}.json'
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / filename
    with target.open('w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Collect raw news data from NewsAPI.')
    parser.add_argument('--topic', required=True, help='Topic to search for.')
    parser.add_argument('--from-date', default=date.today().isoformat(), help='Start date for news search (YYYY-MM-DD).')
    parser.add_argument('--max-pages', type=int, default=5, help='Maximum number of pages to fetch.')
    parser.add_argument('--page-size', type=int, default=100, help='Number of articles per page (max 100).')
    parser.add_argument('--language', default='en', help='Language filter for the news articles.')
    parser.add_argument('--sort-by', default='publishedAt', choices=['relevancy', 'popularity', 'publishedAt'], help='Sort order for the results.')
    parser.add_argument('--delay', type=float, default=1.0, help='Delay in seconds between API requests.')
    parser.add_argument('--output-dir', default='raw_data', help='Directory to save raw JSON output.')
    parser.add_argument('--api-key', default=None, help='NewsAPI API key. If omitted, reads from NEWSAPI_KEY environment variable.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = get_api_key(args.api_key)
    payload = collect_news(
        topic=args.topic,
        from_date=args.from_date,
        max_pages=args.max_pages,
        page_size=args.page_size,
        language=args.language,
        sort_by=args.sort_by,
        delay=args.delay,
        api_key=api_key,
    )
    output_path = save_raw_json(payload, args.topic, Path(args.output_dir))
    print(f'Saved raw data to: {output_path}')


if __name__ == '__main__':
    main()
