# Web Scraping Basics
# Learn the core flow: request -> parse HTML -> find elements.

import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np


# Request the webpage and get its HTML.
webpage = requests.get(
    "https://www.ambitionbox.com/list-of-companies?page=1"
).text

# Parse the HTML using BeautifulSoup.
soup = BeautifulSoup(webpage, "lxml")


# Print the page heading.
print(soup.find_all("h1")[0].text.strip())


# Print all company names from h2 tags.
for item in soup.find_all("h2"):
    print(item.text.strip())


# Print all paragraph text on the page.
for item in soup.find_all("p"):
    print(item.text.strip())


# Find all company containers.
company = soup.find_all("div", class_="company-content-wrapper")

# Check how many company cards were found.
print("Number of companies:", len(company))


# Store scraped fields in separate lists.
name = []
rating = []
reviews = []
ctype = []
hq = []
how_old = []
no_of_employee = []


# Extract information from every company card.
for item in company:
    name.append(item.find("h2").text.strip())
    rating.append(item.find("p", class_="rating").text.strip())
    reviews.append(item.find("a", class_="review-count").text.strip())

    info = item.find_all("p", class_="infoEntity")
    ctype.append(info[0].text.strip())
    hq.append(info[1].text.strip())
    how_old.append(info[2].text.strip())
    no_of_employee.append(info[3].text.strip())


# Convert the lists into a DataFrame.
df = pd.DataFrame({
    "name": name,
    "rating": rating,
    "reviews": reviews,
    "company_type": ctype,
    "Head_Quarters": hq,
    "Company_Age": how_old,
    "No_of_Employee": no_of_employee,
})


# Inspect the scraped data.
print(df.sample(5))
print("Shape:", df.shape)
