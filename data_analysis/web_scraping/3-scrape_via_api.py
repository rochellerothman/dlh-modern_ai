#!/usr/bin/env python3
"""Scrape quote data from an API."""

import json
fetch_html = __import__('0-fetch_html').fetch_html


def scrape_via_api(base_url):
    """Fetch and return quotes from all pages of the quotes API."""
    quotes = []
    page = 1
    has_next = True

    while has_next:
        url = f"{base_url}/api/quotes?page={page}"
        response = fetch_html(url)
        data = json.loads(response)

        for quote in data["quotes"]:
            quotes.append({
                "text": quote["text"],
                "author": quote["author"]["name"],
                "tags": quote["tags"]
            })

        has_next = data["has_next"]
        page += 1

    return quotes
