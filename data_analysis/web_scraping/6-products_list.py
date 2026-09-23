#!/usr/bin/env python3
"""Scrape product information from a static page using Selenium."""

import time
from selenium import webdriver


def scrape_products(url):
    """Scrape product details from a static e-commerce page."""
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1920,1080")

    driver = webdriver.Chrome(options=options)

    try:
        driver.get(url)
        time.sleep(2)

        products = []
        cards = driver.find_elements(
            "css selector",
            ".product-wrapper"
        )

        for card in cards:
            title_link = card.find_element(
                "css selector",
                "a.title"
            )
            price = card.find_element(
                "css selector",
                "h4.price"
            )
            description = card.find_element(
                "css selector",
                "p.description"
            )
            rating = card.find_element(
                "css selector",
                ".ratings p[data-rating]"
            )

            products.append({
                "title": title_link.get_attribute("title"),
                "price": price.text,
                "description": description.text,
                "rating": int(rating.get_attribute("data-rating"))
            })

        return products
    finally:
        driver.quit()
