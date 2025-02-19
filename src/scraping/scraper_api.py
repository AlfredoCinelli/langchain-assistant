"""
Module aimed at scraping the LangChain API docs.

The module has been optimised to run faster using async operations.
"""

import aiohttp
import asyncio
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin, urlparse
import warnings
from typing import Set, List
import aiofiles
import logging
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FastLangchainAPIScraper:
    def __init__(
        self,
        base_url: str = "https://python.langchain.com/api_reference/",
        max_concurrent_requests: int = 50,
        max_retries: int = 3,
    ):
        self.base_url = base_url
        self.visited_urls: Set[str] = set()
        self.max_concurrent_requests = max_concurrent_requests
        self.max_retries = max_retries
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.session = None
        self.thread_pool = ThreadPoolExecutor(max_workers=10)  # For file I/O operations

    def is_valid_url(self, url: str) -> bool:
        """Check if URL belongs to API reference documentation."""
        return url.startswith(self.base_url)

    def clean_filename(self, url: str) -> str:
        """Convert URL to a valid filename."""
        parsed = urlparse(url)
        path = parsed.path.split("api_reference/")[-1]

        if not path:
            return "index.html"

        if not path.endswith(".html"):
            if path.endswith("/"):
                path = path + "index.html"
            else:
                path = path + "/index.html"

        return path

    async def save_html(self, file_path: str, content: str):
        """Save HTML content to file asynchronously."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
            await f.write(content)

    def extract_links(self, html: str, base_url: str) -> List[str]:
        """Extract valid links from HTML content."""
        soup = BeautifulSoup(html, "html.parser")
        new_urls = []

        for link in soup.find_all("a", href=True):
            href = link["href"]
            absolute_url = urljoin(base_url, href)

            if (
                self.is_valid_url(absolute_url)
                and absolute_url not in self.visited_urls
                and not any(
                    absolute_url.endswith(ext)
                    for ext in [".png", ".jpg", ".jpeg", ".gif", ".pdf"]
                )
            ):
                new_urls.append(absolute_url)

        return new_urls

    async def fetch_url(self, url: str, output_dir: str) -> List[str]:
        """Fetch and process a single URL with retries."""
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
                            html_content = await response.text()

                            # Save HTML content
                            rel_path = self.clean_filename(url)
                            file_path = os.path.join(output_dir, rel_path)
                            await self.save_html(file_path, html_content)

                            logger.info(f"Successfully scraped: {url}")
                            return self.extract_links(html_content, url)
                        else:
                            logger.warning(
                                f"Failed to fetch {url}: Status {response.status}"
                            )

            except Exception as e:
                retry_count += 1
                if retry_count < self.max_retries:
                    await asyncio.sleep(1)
                    continue
                logger.error(f"Error processing {url}: {str(e)}")

            return []

    async def process_queue(self, queue: asyncio.Queue, output_dir: str):
        """Process URLs from the queue concurrently."""
        while True:
            url = await queue.get()
            new_urls = await self.fetch_url(url, output_dir)

            for new_url in new_urls:
                await queue.put(new_url)

            queue.task_done()

    async def scrape(self, output_dir: str):
        """Scrape the API reference concurrently."""
        os.makedirs(output_dir, exist_ok=True)

        # Create persistent session with connection pooling
        connector = aiohttp.TCPConnector(
            limit=self.max_concurrent_requests,
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

            # Initialize URL queue
            queue = asyncio.Queue()
            await queue.put(self.base_url)

            # Create worker tasks
            workers = [
                asyncio.create_task(self.process_queue(queue, output_dir))
                for _ in range(self.max_concurrent_requests)
            ]

            # Wait for all URLs to be processed
            await queue.join()

            # Cancel worker tasks
            for worker in workers:
                worker.cancel()

            await asyncio.gather(*workers, return_exceptions=True)


async def main():
    # Initialize scraper
    scraper = FastLangchainAPIScraper()
    output_dir = "./langchain-api"

    logger.info("Starting API reference scrape...")
    logger.info(f"Saving HTML files to: {output_dir}")

    if os.path.exists(output_dir):
        user_input = input(
            f"Directory {output_dir} already exists. Do you want to continue? (y/n): "
        )
        if user_input.lower() != "y":
            logger.info("Exiting...")
            return

    # Start scraping
    await scraper.scrape(output_dir)

    logger.info("\nScraping completed!")
    logger.info(f"Total pages scraped: {len(scraper.visited_urls)}")


if __name__ == "__main__":
    asyncio.run(main())
