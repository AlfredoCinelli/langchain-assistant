"""
Module aimed at scraping the LangChain docs about tutorials, how to and concepts.

The module has been optimised to run faster using async operations.
"""

import aiohttp
import asyncio
from bs4 import BeautifulSoup
import os
from urllib.parse import urlparse, urljoin
import warnings
import logging
from typing import Set, List
import aiofiles

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AsyncLangchainDocsScraper:
    def __init__(self, max_concurrent: int = 50, max_retries: int = 3):
        self.visited_urls: Set[str] = set()
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.session = None

    def is_valid_doc_url(self, url: str) -> bool:
        parsed = urlparse(url)
        path = parsed.path.lower()
        return any(
            path.startswith(prefix)
            for prefix in ["/docs/how_to", "/docs/concepts", "/docs/tutorials"]
        )

    def clean_filename(self, url: str) -> str:
        parsed = urlparse(url)
        path = parsed.path.strip("/")

        if not path:
            return "index.html"

        if not path.endswith(".html"):
            path = os.path.join(path, "index.html")

        try:
            docs_index = [
                i for i, part in enumerate(path.split("/")) if part == "docs"
            ][0]
            path = "/".join(path.split("/")[docs_index:])
        except IndexError:
            pass

        return path

    async def save_html(self, file_path: str, content: str):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
            await f.write(content)

    def extract_links(self, html: str, base_url: str) -> List[str]:
        soup = BeautifulSoup(html, "html.parser")
        base_domain = urlparse(base_url).netloc
        new_urls = []

        for link in soup.find_all("a", href=True):
            href = link["href"]
            if href.startswith("#"):
                continue

            absolute_url = urljoin(base_url, href)
            parsed_href = urlparse(absolute_url)

            if (
                parsed_href.netloc == base_domain
                and absolute_url not in self.visited_urls
                and not any(
                    absolute_url.lower().endswith(ext)
                    for ext in [
                        ".png",
                        ".jpg",
                        ".jpeg",
                        ".gif",
                        ".pdf",
                        ".zip",
                        ".css",
                        ".js",
                    ]
                )
            ):
                new_urls.append(absolute_url)

        return new_urls

    async def fetch_url(self, url: str, output_dir: str) -> List[str]:
        if url in self.visited_urls:
            return []

        self.visited_urls.add(url)
        retry_count = 0

        while retry_count < self.max_retries:
            try:
                async with self.semaphore:
                    async with self.session.get(
                        url, timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        if response.status == 200:
                            if "text/html" not in response.headers.get(
                                "content-type", ""
                            ):
                                return []

                            html_content = await response.text()

                            if self.is_valid_doc_url(url):
                                file_path = os.path.join(
                                    output_dir, self.clean_filename(url)
                                )
                                await self.save_html(file_path, html_content)
                                logger.info(f"Saved: {file_path}")

                            return self.extract_links(html_content, url)

            except Exception as e:
                retry_count += 1
                if retry_count < self.max_retries:
                    await asyncio.sleep(1)
                    continue
                logger.error(f"Error processing {url}: {str(e)}")

            return []

    async def process_queue(self, queue: asyncio.Queue, output_dir: str):
        while True:
            url = await queue.get()
            new_urls = await self.fetch_url(url, output_dir)

            for new_url in new_urls:
                await queue.put(new_url)

            queue.task_done()

    async def scrape(self, base_urls: List[str], output_dir: str):
        connector = aiohttp.TCPConnector(
            limit=self.max_concurrent,
            force_close=False,
            enable_cleanup_closed=True,
            ssl=False,
        )

        async with aiohttp.ClientSession(
            connector=connector,
            headers={"User-Agent": "Mozilla/5.0"},
            raise_for_status=True,
        ) as session:
            self.session = session
            queue = asyncio.Queue()

            for url in base_urls:
                await queue.put(url)

            workers = [
                asyncio.create_task(self.process_queue(queue, output_dir))
                for _ in range(self.max_concurrent)
            ]

            await queue.join()

            for worker in workers:
                worker.cancel()

            await asyncio.gather(*workers, return_exceptions=True)


async def main():
    base_urls = [
        "https://python.langchain.com/docs/how_to/",
        "https://python.langchain.com/docs/concepts/",
        "https://python.langchain.com/docs/tutorials/",
        "https://python.langchain.com/docs/integrations/",
    ]

    output_dir = "./langchain-docs/"

    if os.path.exists(output_dir):
        user_input = input(
            f"Directory {output_dir} already exists. Download again? (y/n): "
        )
        if user_input.lower() == "y":
            os.system(f"rm -rf {output_dir}")
        else:
            logger.info("Exiting...")
            return

    os.makedirs(output_dir, exist_ok=True)

    scraper = AsyncLangchainDocsScraper()
    await scraper.scrape(base_urls, output_dir)
    logger.info(f"\nScraping completed! Total pages: {len(scraper.visited_urls)}")


if __name__ == "__main__":
    asyncio.run(main())
