#!/usr/bin/env python3
"""Log in to a website and scrape quote data."""

import requests
from bs4 import BeautifulSoup


def login_and_scrape(login_url, user, pwd):
    """Log in with a session and scrape quotes from the protected page."""
    session = requests.Session()

    response = session.get(login_url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')
    csrf_input = soup.find('input', {'name': 'csrf_token'})
    csrf_token = csrf_input.get('value')

    login_data = {
        'username': user,
        'password': pwd,
        'csrf_token': csrf_token
    }

    response = session.post(
        login_url,
        data=login_data
    )
    response.raise_for_status()

    quotes_url = "https://quotes.toscrape.com/"
    response = session.get(quotes_url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')
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
