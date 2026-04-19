import argparse
import json
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer, util


def load_cleaned_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding='utf-8', parse_dates=['published_at'])


def get_top_sources(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    return df['source_name'].value_counts().head(top_n).reset_index(names=['source_name', 'count'])


def get_temporal_distribution(df: pd.DataFrame) -> pd.DataFrame:
    df['published_date'] = pd.to_datetime(df['published_at']).dt.date
    return df.groupby('published_date').size().reset_index(name='count')


def get_top_keywords(df: pd.DataFrame, field: str = 'title', top_n: int = 20) -> list:
    texts = df[field].fillna('').astype(str).tolist()
    vectorizer = CountVectorizer(stop_words='english', max_features=2000)
    matrix = vectorizer.fit_transform(texts)
    sums = np.asarray(matrix.sum(axis=0)).flatten()
    terms = vectorizer.get_feature_names_out()
    top_indices = np.argsort(sums)[::-1][:top_n]
    return [(terms[idx], int(sums[idx])) for idx in top_indices]


def build_sentence_embeddings(texts: list[str], model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
    model = SentenceTransformer(model_name)
    return model.encode(texts, convert_to_tensor=True, show_progress_bar=False)


def semantic_similarity_report(df: pd.DataFrame, model_name: str = 'all-MiniLM-L6-v2', top_n: int = 10) -> dict:
    titles = df['title'].fillna('').astype(str).tolist()
    embeddings = build_sentence_embeddings(titles, model_name)
    similarities = util.pytorch_cos_sim(embeddings, embeddings)
    pairs = []
    num_texts = min(len(titles), 200)
    for i in range(num_texts):
        for j in range(i + 1, num_texts):
            pairs.append((float(similarities[i, j]), i, j))
    pairs.sort(reverse=True, key=lambda item: item[0])
    top_pairs = []
    for score, i, j in pairs[:top_n]:
        top_pairs.append({
            'score': round(score, 3),
            'title_1': titles[i],
            'title_2': titles[j],
        })
    return {'model': model_name, 'top_similar_pairs': top_pairs}


def sentiment_analysis(df: pd.DataFrame, max_examples: int = 50) -> dict:
    from transformers import pipeline

    texts = df['title'].fillna('').astype(str).tolist()[:max_examples]
    sentiment_pipe = pipeline('sentiment-analysis')
    results = sentiment_pipe(texts)
    counts = Counter([item['label'] for item in results])
    return {'sentiment_counts': dict(counts), 'sample_results': results[:10]}


def emotion_detection(df: pd.DataFrame, max_examples: int = 50) -> dict:
    from transformers import pipeline

    texts = df['title'].fillna('').astype(str).tolist()[:max_examples]
    emotion_pipe = pipeline('text-classification', model='nateraw/bert-base-uncased-emotion', return_all_scores=False)
    results = emotion_pipe(texts)
    counts = Counter([item['label'] for item in results])
    return {'emotion_counts': dict(counts), 'sample_results': results[:10]}


def gpt_sentiment_analysis(sample_texts: list[str], max_examples: int = 5) -> dict:
    try:
        import openai
    except ImportError:
        return {'error': 'openai library not installed'}

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return {'error': 'OPENAI_API_KEY not set'}

    openai.api_key = api_key
    examples = sample_texts[:max_examples]
    prompt = (
        'Please analyze the sentiment and emotional tone of the following news headlines. '
        'For each headline, return a JSON array with fields: headline, sentiment(positive/neutral/negative), emotion.'
    )
    for idx, headline in enumerate(examples, start=1):
        prompt += f"\n{idx}. {headline}"

    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0.0,
        max_tokens=800,
    )
    content = response['choices'][0]['message']['content']
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {'raw': content}
    return {'gpt_response': parsed}


def save_report(report: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run news analysis on cleaned data.')
    parser.add_argument('--input-file', required=True, help='Path to cleaned CSV file from data_cleaner.py.')
    parser.add_argument('--output-dir', default='analysis_results', help='Directory to save analysis results.')
    parser.add_argument('--topic', default='topic', help='Topic label used in report names.')
    parser.add_argument('--top-n-sources', type=int, default=10, help='Number of top sources to include.')
    parser.add_argument('--top-n-keywords', type=int, default=20, help='Number of top keywords to include.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_file)
    df = load_cleaned_data(input_path)
    report = {
        'generated_at': datetime.utcnow().isoformat() + 'Z',
        'topic': args.topic,
        'row_count': int(len(df)),
        'top_sources': get_top_sources(df, args.top_n_sources).to_dict(orient='records'),
        'temporal_distribution': get_temporal_distribution(df).to_dict(orient='records'),
        'top_keywords': get_top_keywords(df, 'title', args.top_n_keywords),
    }

    report['semantic_similarity'] = semantic_similarity_report(df)
    report['sentiment_analysis'] = sentiment_analysis(df)
    report['emotion_detection'] = emotion_detection(df)
    report['gpt_sentiment_analysis'] = gpt_sentiment_analysis(df['title'].fillna('').astype(str).tolist())

    output_path = Path(args.output_dir) / f'analysis_{args.topic}_{datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")}.json'
    save_report(report, output_path)
    print(f'Analysis report saved to {output_path}')


if __name__ == '__main__':
    main()
