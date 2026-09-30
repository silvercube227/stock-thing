"""Archive reviewed deal sources by content hash, without applying any labels."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path

import httpx

from backend.config import get_settings
from backend.ml.research import write_json_new


async def run(reviews, output):
    raw = Path(reviews).read_bytes()
    review = json.loads(raw)
    archived = []
    async with httpx.AsyncClient(timeout=60, follow_redirects=True,
                                headers={'User-Agent': get_settings().sec_edgar_user_agent}) as client:
        for event in review['events']:
            prior = event.get('archived_source')
            if prior:
                content = Path(prior).read_bytes()
                if hashlib.sha256(content).hexdigest() != event['source_sha256']:
                    raise ValueError(f"Archived source changed: {event['symbol']}")
            else:
                response = await client.get(event['source'])
                response.raise_for_status()
                content = response.content
            digest = hashlib.sha256(content).hexdigest()
            suffix = '.pdf' if content.startswith(b'%PDF') else '.html'
            path = Path('.research/terminal_sources') / f'{digest}{suffix}'
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                if path.read_bytes() != content:
                    raise ValueError('Hash-addressed archive content mismatch')
            else:
                with path.open('xb') as stream:
                    stream.write(content)
            archived.append({**event, 'source_sha256': digest, 'archived_source': str(path)})
    write_json_new(output, {**review, 'events': archived,
                            'review_sha256': hashlib.sha256(raw).hexdigest()})
    print({'archived_events': len(archived), 'output': output, 'labels_applied': 0})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reviews', default='docs/terminal_event_research.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    asyncio.run(run(args.reviews, args.output))
