app:
	uv run streamlit run app.py

scraper_api:
	uv run src/scraping/scraper_api.py

scraper_docs:
	uv run src/scraping/scraper_docs.py

linters:
	@sh bash/execute_linters.sh $(path)

ingestion:
	uv run src/ingestion/ingestion.py