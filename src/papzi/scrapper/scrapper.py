# from selenium import webdriver
# from webdriver_manager.chrome import ChromeDriverManager

import io
from PIL import Image

# from selenium.webdriver.chrome.service import Service as ChromeService

import time


from pathlib import Path
from papzi.constants import BASE_DIR
import asyncio
import aiohttp
from urllib.parse import urlencode, quote, urlunparse

import undetected_chromedriver as uc


class PhotoScrapper:
    def __init__(self) -> None:
        self.driver = uc.Chrome(headless=True, use_subprocess=False)

        # webdriver.Chrome(
        #     service=ChromeService(ChromeDriverManager().install())
        # )
        self.sem = asyncio.Semaphore(50)
        self.suffixes = [
            "single photo",
            "red carpet",
            "headshot",
            "gettyimages",
            "face jpg",
        ]

    def _url(self, search_term: str, suffix: str):

        query_params = {
            "q": f"{search_term} {suffix}",
            "t": "h_",
            "iar": "images",
            "iax": "images",
            "ia": "images",
            "iaf": "layout:Square",
        }

        # Encode the query parameters
        encoded_query = urlencode(query_params, quote_via=quote)

        # Build the full URL
        url = urlunparse(
            ("https", "duckduckgo.com", "/", "", encoded_query, "")
        )

        return url

    async def _img_urls(self, search_term: str, suffix: str) -> list[str]:
        self.driver.get(self._url(search_term, suffix))
        thumb_count = 0

        for _ in range(6):
            self.driver.execute_script(
                "window.scrollTo(0, document.body.scrollHeight);"
            )
            for i in range(5):
                thumbnails = self.driver.find_elements(
                    "xpath", "//img[contains(@class, 'tile--img__img')]"
                )
                if thumb_count != len(thumbnails):
                    break

                await asyncio.sleep(1)

                thumb_count = len(thumbnails)
        _urls = []
        for thumb in thumbnails:
            if thumb.get_attribute("src") is not None:
                _urls.append(thumb.get_attribute("src"))

        return _urls

    async def _save(
        self,
        sess: aiohttp.ClientSession,
        url: str,
        path: Path,
    ):
        async with self.sem:
            try:
                resp = await sess.get(url)
                Image.open(io.BytesIO(await resp.content.read())).save(path)
            except Exception:
                print("problem with", path)

    async def _save_imgs(self, img_urls: list[str], store_base_path: Path):
        tasks = []
        async with aiohttp.ClientSession() as sess:
            for i, img_url in enumerate(img_urls):
                task = asyncio.create_task(
                    self._save(
                        sess,
                        img_url,
                        store_base_path / f"{i}.jpg",
                    ),
                )

                tasks.append(task)
            print(len(tasks))

            print("all tasks started")
            await asyncio.gather(*tasks)

    async def scrap(self, search_term: str):
        img_urls = []

        tasks = []
        for suffix in self.suffixes:
            task = asyncio.create_task(self._img_urls(search_term, suffix))
            tasks.append(task)
        results = await asyncio.gather(*tasks)
        img_urls = [img_url for result in results for img_url in result]
        img_urls = list(set(img_urls))
        print(search_term, len(img_urls))

        store_path = BASE_DIR / "scraped" / search_term
        if not store_path.exists():
            store_path.mkdir()

        await self._save_imgs(img_urls, store_path)

    async def scrap_all(self, search_terms: list[str]):

        print("worker started")
        self.driver.set_window_size(1400, 1050)
        for search_term in search_terms:
            start = time.perf_counter()
            print("downloading", search_term)
            await self.scrap(search_term)
            print(search_term, "took", time.perf_counter() - start)

    def run(self, search_terms: list[str]):
        asyncio.run(self.scrap_all(search_terms))


def run_scrapper(search_terms: list[str]):
    scrapper = PhotoScrapper()
    scrapper.run(search_terms)
