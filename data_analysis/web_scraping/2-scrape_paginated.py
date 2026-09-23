#!/usr/bin/env python3
"""Scrape quotes from multiple paginated web pages."""

from bs4 import BeautifulSoup
import time
from urllib import parse
fetch_html = __import__('0-fetch_html').fetch_html
scrape_basic = __import__('1-scrape_basic').scrape_basic


def scrape_paginated(base_url):
    """Scrape all quote pages by following each Next link."""
    quotes = []
    current_url = base_url

    while current_url:
        quotes.extend(scrape_basic(current_url))

        html = fetch_html(current_url)
        soup = BeautifulSoup(html, 'html.parser')
        next_link = soup.select_one('li.next a')

        if next_link:
            current_url = parse.urljoin(
                current_url,
                next_link.get('href')
            )
            time.sleep(1)
        else:
            current_url = None

    return quotes
