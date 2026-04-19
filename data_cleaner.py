import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


def load_raw_json(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def flatten_articles(raw_data: dict) -> pd.DataFrame:
    records = []
    for article in raw_data.get('articles', []):
        records.append(
            {
                'source_id': article.get('source', {}).get('id'),
                'source_name': article.get('source', {}).get('name'),
                'author': article.get('author'),
                'title': article.get('title'),
                'description': article.get('description'),
                'url': article.get('url'),
                'url_to_image': article.get('urlToImage'),
                'published_at': article.get('publishedAt'),
                'content': article.get('content'),
            }
        )
    return pd.DataFrame(records)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.dropna(subset=['title', 'url'], inplace=True)
    df['source_name'] = df['source_name'].fillna('unknown')
    df['author'] = df['author'].fillna('unknown')
    df['description'] = df['description'].fillna('')
    df['content'] = df['content'].fillna('')

    df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
    df = df[df['published_at'].notna()]
    df['published_date'] = df['published_at'].dt.date
    df['published_hour'] = df['published_at'].dt.hour

    df['url_hash'] = df['url'].astype(str).apply(hash)
    df.drop_duplicates(subset=['url_hash'], inplace=True)
    df.drop(columns=['url_hash'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def compute_metrics(raw_count: int, df: pd.DataFrame, duplicate_count: int) -> dict:
    return {
        'raw_article_count': raw_count,
        'clean_article_count': int(len(df)),
        'duplicate_count': int(duplicate_count),
        'drop_missing_title_or_url': int(raw_count - len(df) - duplicate_count),
        'source_count': int(df['source_name'].nunique()),
        'top_sources': df['source_name'].value_counts().head(5).to_dict(),
    }


def save_cleaned_data(df: pd.DataFrame, raw_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    target_csv = output_dir / f'cleaned_{raw_path.stem}.csv'
    target_parquet = output_dir / f'cleaned_{raw_path.stem}.parquet'
    df.to_csv(target_csv, index=False, encoding='utf-8')
    df.to_parquet(target_parquet, index=False)
    return target_csv


def save_metrics(metrics: dict, output_dir: Path, raw_path: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f'metrics_{raw_path.stem}.json'
    with target.open('w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Clean raw news JSON and compute quality metrics.')
    parser.add_argument('--input-file', required=True, help='Path to raw JSON file produced by data_collector.py.')
    parser.add_argument('--output-dir', default='cleaned_data', help='Directory to save cleaned data and metrics.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_file)
    raw_data = load_raw_json(input_path)
    raw_df = flatten_articles(raw_data)
    raw_count = len(raw_df)

    before = len(raw_df)
    cleaned_df = clean_dataframe(raw_df)
    after = len(cleaned_df)
    duplicates_removed = before - after

    metrics = compute_metrics(raw_count, cleaned_df, duplicates_removed)
    print('=== Cleaning Metrics ===')
    for key, value in metrics.items():
        print(f'{key}: {value}')

    cleaned_path = save_cleaned_data(cleaned_df, input_path, Path(args.output_dir))
    metrics_path = save_metrics(metrics, Path(args.output_dir), input_path)
    print(f'Cleaned CSV saved to: {cleaned_path}')
    print(f'Metrics saved to: {metrics_path}')


if __name__ == '__main__':
    main()
