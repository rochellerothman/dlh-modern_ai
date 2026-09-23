#!/usr/bin/env python3
"""Scroll through a page and scrape unique products using Selenium."""

import time
from selenium import webdriver


def scroll_and_scrape(url, scroll_pause=2.0):
    """Scroll to the end of a page and return all unique products."""
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1920,1080")

    driver = webdriver.Chrome(options=options)

    try:
        driver.get(url)

        last_height = driver.execute_script(
            "return document.body.scrollHeight"
        )

        while True:
            driver.execute_script(
                "window.scrollTo(0, document.body.scrollHeight);"
            )

            time.sleep(scroll_pause)

            new_height = driver.execute_script(
                "return document.body.scrollHeight"
            )

            if new_height == last_height:
                break

            last_height = new_height

        cards = driver.find_elements(
            "css selector",
            "div.thumbnail"
        )

        products = []
        seen = set()

        for card in cards:
            title_link = card.find_element(
                "css selector",
                "a.title"
            )
            title = title_link.get_attribute("title")

            price = card.find_element(
                "css selector",
                "h4.price"
            ).text

            key = (title, price)

            if key in seen:
                continue

            seen.add(key)

            description = card.find_element(
                "css selector",
                "p.description"
            ).text

            stars = card.find_elements(
                "css selector",
                ".ratings .ws-icon-star"
            )

            products.append({
                "title": title,
                "price": price,
                "description": description,
                "rating": len(stars)
            })

        return products
    finally:
        driver.quit()
