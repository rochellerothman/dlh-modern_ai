#!/usr/bin/env python3
"""Scrape quotes from a static web page."""

from bs4 import BeautifulSoup
fetch_html = __import__('0-fetch_html').fetch_html


def scrape_basic(url):
    """Scrape quotes, authors, and tags from a Quotes to Scrape page."""
    html = fetch_html(url)
    soup = BeautifulSoup(html, 'html.parser')
    quotes = []

    for quote in soup.find_all('div', class_='quote'):
        text = quote.find('span', class_='text').get_text()
        author = quote.find('small', class_='author').get_text()
        tags = [
            tag.get_text()
            for tag in quote.find_all('a', class_='tag')
        ]

        quotes.append({
            "text": text,
            "author": author,
            "tags": tags
        })

    return quotes
