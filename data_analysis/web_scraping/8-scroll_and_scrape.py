#!/usr/bin/env python3
"""Scroll through a page and scrape unique products using Selenium."""

import time
from selenium import webdriver


def scroll_and_scrape(url, scroll_pause=2.0):
    """Scroll to the end of a page and return all unique products."""
    options = webdriver.ChromeOptions()
    options.page_load_strategy = "none"
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1920,1080")
    options.add_experimental_option(
        "prefs",
        {"profile.managed_default_content_settings.images": 2}
    )

    driver = webdriver.Chrome(options=options)

    try:
        driver.get(url)

        start = time.time()

        while True:
            count = driver.execute_script(
                "return document.querySelectorAll("
                "'div.thumbnail').length;"
            )

            if count > 0:
                break

            if time.time() - start >= scroll_pause:
                break

            time.sleep(0.05)

        driver.set_script_timeout(25)

        driver.execute_async_script(
            """
            const done = arguments[arguments.length - 1];
            let lastHeight = document.body.scrollHeight;
            let stableSince = Date.now();
            const started = Date.now();

            function scrollPage() {
                window.scrollTo(
                    0,
                    document.body.scrollHeight
                );

                const newHeight = document.body.scrollHeight;

                if (newHeight > lastHeight) {
                    lastHeight = newHeight;
                    stableSince = Date.now();
                }

                if (Date.now() - stableSince >= 1000) {
                    done();
                    return;
                }

                if (Date.now() - started >= 20000) {
                    done();
                    return;
                }

                setTimeout(scrollPage, 50);
            }

            scrollPage();
            """
        )

        data = driver.execute_script(
            """
            return Array.from(
                document.querySelectorAll('div.thumbnail')
            ).map(card => {
                const title = card.querySelector('a.title');
                const price = card.querySelector('h4.price');
                const description = card.querySelector(
                    'p.description'
                );
                const stars = card.querySelectorAll(
                    '.ratings .ws-icon-star'
                );

                return {
                    title: title ? title.getAttribute('title') : '',
                    price: price ? price.textContent.trim() : '',
                    description: description
                        ? description.textContent.trim() : '',
                    rating: stars.length
                };
            });
            """
        )

        products = []
        seen = set()

        for product in data:
            key = (
                product["title"],
                product["price"]
            )

            if key not in seen:
                seen.add(key)
                products.append(product)

        return products
    finally:
        driver.quit()
