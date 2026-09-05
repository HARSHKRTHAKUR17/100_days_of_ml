# Web Scraping

This folder contains practical Python web-scraping work using `Requests`,
`BeautifulSoup`, NumPy, and Pandas.

## What I Learned

- Sending HTTP requests with `requests`
- Understanding HTML responses
- Parsing HTML using BeautifulSoup
- Navigating HTML elements
- `find()` and `find_all()`
- Extracting text from HTML
- Identifying elements using tags and CSS classes
- Converting scraped data into Pandas DataFrames
- Handling missing fields during scraping
- Scraping multiple pages
- Pagination
- Combining data from multiple pages
- Basic handling of HTTP 403 responses

## Main Scraping Pipeline

Website
→ HTTP Request
→ HTML Response
→ BeautifulSoup
→ Locate Elements
→ Extract Data
→ Pandas DataFrame

## Project

The main scraping example collects company information from AmbitionBox,
including:

- Company name
- Rating
- Number of reviews
- Company type
- Headquarters
- Company age
- Number of employees

The multi-page scraper extends this to hundreds of pages and combines the
results into one DataFrame.

## Libraries

- `requests`
- `beautifulsoup4`
- `lxml`
- `numpy`
- `pandas`

## Key Takeaway

The goal was not to become a specialized web-scraping engineer. The important
skill is being able to collect structured data from the web and turn it into
a dataset that can subsequently be analyzed or used for ML.