#!/usr/bin/env python3
"""Extract quote data from JSON-LD embedded in a web page."""

import json
from bs4 import BeautifulSoup
fetch_html = __import__('0-fetch_html').fetch_html


def extract_jsonld(url):
    """Extract quotes from JSON-LD blocks in a web page."""
    html = fetch_html(url)
    soup = BeautifulSoup(html, 'html.parser')
    quotes = []

    scripts = soup.find_all(
        'script',
        type='application/ld+json'
    )

    for script in scripts:
        data = json.loads(script.get_text())

        if data.get("@type") == "Quote":
            keywords = data.get("keywords", [])

            if isinstance(keywords, str):
                keywords = [
                    tag.strip()
                    for tag in keywords.split(",")
                ]

            quotes.append({
                "text": data.get("text"),
                "author": data.get("author", {}).get("name"),
                "tags": keywords
            })

    return quotes
