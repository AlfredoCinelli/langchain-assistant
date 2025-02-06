import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin, urlparse
import time
import warnings
warnings.filterwarnings("ignore")

class LangchainAPIScraper:
    def __init__(self, base_url: str = "https://python.langchain.com/api_reference/"):
        self.base_url = base_url
        self.visited_urls = set()

    def is_valid_url(self, url: str) -> bool:
        """Check if URL belongs to API reference documentation."""
        return url.startswith(self.base_url)

    def clean_filename(self, url: str) -> str:
        """Convert URL to a valid filename."""
        parsed = urlparse(url)
        # Get the path after 'api_reference'
        path = parsed.path.split('api_reference/')[-1]
        
        # If path is empty, use index.html
        if not path:
            return 'index.html'
        
        # If path doesn't end in .html, append index.html
        if not path.endswith('.html'):
            if path.endswith('/'):
                path = path + 'index.html'
            else:
                path = path + '/index.html'
        
        return path

    def scrape_page(self, url: str, output_dir: str) -> list:
        """Scrape a single page and save its HTML content."""
        if url in self.visited_urls:
            return []
        
        print(f"Scraping: {url}")
        self.visited_urls.add(url)
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Save the HTML content
            rel_path = self.clean_filename(url)
            file_path = os.path.join(output_dir, rel_path)
            
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save the HTML file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            print(f"Saved: {file_path}")
            
            # Parse HTML for links
            soup = BeautifulSoup(response.text, 'html.parser')
            new_urls = []
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                absolute_url = urljoin(url, href)
                
                # Only follow API reference links
                if (self.is_valid_url(absolute_url) and 
                    absolute_url not in self.visited_urls and
                    not any(absolute_url.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.pdf'])):
                    new_urls.append(absolute_url)
            
            return new_urls
            
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return []

    def scrape(self, output_dir: str, delay: float = 1.0):
        """
        Scrape the API reference and save HTML files.
        
        Args:
            output_dir: Directory to save HTML files
            delay: Time to wait between requests in seconds
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        urls_to_visit = [self.base_url]
        
        while urls_to_visit:
            url = urls_to_visit.pop(0)
            new_urls = self.scrape_page(url, output_dir)
            urls_to_visit.extend(new_urls)
            
            # Rate limiting
            time.sleep(delay)

if __name__ == "__main__":
    # Initialize scraper
    scraper = LangchainAPIScraper()
    
    # Define output directory
    output_dir = "./langchain_api_docs"
    
    # Start scraping
    print("Starting API reference scrape...")
    print(f"Saving HTML files to: {output_dir}")
    
    # Check if directory exists
    if os.path.exists(output_dir):
        user_input = input(f"Directory {output_dir} already exists. Do you want to continue? (y/n): ")
        if user_input.lower() != 'y':
            print("Exiting...")
            exit()
    
    # Scrape and save HTML files
    scraper.scrape(output_dir)
    
    # Print completion message
    print("\nScraping completed!")
    print(f"Total pages scraped: {len(scraper.visited_urls)}")