import requests
from bs4 import BeautifulSoup
import os
import urllib.parse
import warnings
from typing import Set
from urllib.parse import urlparse, urljoin
from queue import Queue

warnings.filterwarnings("ignore")

def is_valid_doc_url(url: str) -> bool:
    """Check if the URL is part of the Langchain documentation paths."""
    parsed = urlparse(url)
    path = parsed.path.lower()
    return any(path.startswith(prefix) for prefix in [
        '/docs/how_to', 
        '/docs/concepts', 
        '/docs/tutorials'
    ])

def clean_filename(url: str) -> str:
    """Convert URL path to a clean filename/directory structure."""
    parsed = urlparse(url)
    path = parsed.path.strip('/')
    
    # If path is empty, use 'index.html'
    if not path:
        return 'index.html'
    
    # If path doesn't end in .html, append index.html
    if not path.endswith('.html'):
        path = os.path.join(path, 'index.html')
    
    return path

def download_docs(base_url: str, output_dir: str) -> str:
    """
    Iteratively downloads Langchain documentation HTML pages from a given URL.
    
    Args:
        base_url: The base URL to start scraping from
        output_dir: Directory to save the HTML files
    
    Returns:
        str: Status message indicating success or failure
    """
    visited_urls = set()
    url_queue = Queue()
    url_queue.put(base_url)
    
    while not url_queue.empty():
        try:
            url = url_queue.get()
            
            if url in visited_urls:
                continue
            
            visited_urls.add(url)
            
            print(f"Fetching: {url}")
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Save only if it's an HTML page and in valid documentation paths
            if 'text/html' in response.headers.get('content-type', '') and is_valid_doc_url(url):
                # Create clean file path 
                rel_path = clean_filename(url)
                path_parts = rel_path.split('/')
                
                # Preserve docs structure
                try:
                    docs_index = [i for i, part in enumerate(path_parts) if part == 'docs'][0]
                    rel_path = '/'.join(path_parts[docs_index:])
                except IndexError:
                    rel_path = rel_path  # Keep full path if docs not found
                
                file_path = os.path.join(output_dir, rel_path)
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                # Save the HTML file
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(response.text)
                print(f"Saved: {file_path}")
            
            # Find all links
            soup = BeautifulSoup(response.text, 'html.parser')
            links = soup.find_all('a', href=True)
            base_domain = urlparse(url).netloc
            
            for link in links:
                href = link['href']
                
                # Skip fragment identifiers
                if href.startswith('#'):
                    continue
                    
                # Make absolute URL if necessary
                if not href.startswith('http'):
                    href = urljoin(url, href)
                
                # Only follow links from the same domain
                parsed_href = urlparse(href)
                if (parsed_href.netloc == base_domain and 
                    href not in visited_urls and
                    not any(href.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.pdf', '.zip', '.css', '.js'])):
                    
                    url_queue.put(href)
            
        except Exception as e:
            print(f"Error downloading {url}: {e}")
    
    return f"Successfully downloaded Langchain docs to {output_dir}"

if __name__ == "__main__":
    # List of base URLs to scrape
    base_urls = [
        "https://python.langchain.com/docs/how_to/",
        "https://python.langchain.com/docs/concepts/",
        "https://python.langchain.com/docs/tutorials/",
        "https://python.langchain.com/docs/integrations/",
    ]
    
    output_dir = "./langchain-docs/"
    
    docs_exists = os.path.exists(output_dir)
    if docs_exists:
        print(f"Directory {output_dir} already exists.")
        download = input("Do you want to download the docs again? (y/n): ")
        if download == "y":
            print("Downloading docs...")
            os.system(f"rm -rf {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
            
            # Download from multiple base URLs
            for url in base_urls:
                output = download_docs(url, output_dir)
                print(output)
        else:
            output = "Exiting..."
    else:
        print(f"Directory {output_dir} does not exist. Creating it now.")
        os.makedirs(output_dir, exist_ok=True)
        
        # Download from multiple base URLs
        for url in base_urls:
            output = download_docs(url, output_dir)
            print(output)