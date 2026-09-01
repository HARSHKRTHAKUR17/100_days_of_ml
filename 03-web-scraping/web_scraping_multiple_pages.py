# Web Scraping Multiple Pages
# Learn how to loop through many pages and handle missing fields safely.

import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np


# Store results from every page here.
final = pd.DataFrame()


# Scrape pages 1 through 1000.
for page in range(1, 1001):

    # Download the current page.
    webpage = requests.get(
        "https://www.ambitionbox.com/list-of-companies?page={}".format(page)
    ).text

    # Parse the HTML.
    soup = BeautifulSoup(webpage, "lxml")

    # Find all company cards.
    company = soup.find_all("div", class_="company-content-wrapper")

    # Create empty lists for this page.
    name = []
    rating = []
    reviews = []
    ctype = []
    hq = []
    how_old = []
    no_of_employee = []

    # Extract information from each company card.
    for item in company:

        # Missing fields are stored as NaN instead of crashing the loop.
        try:
            name.append(item.find("h2").text.strip())
        except:
            name.append(np.nan)

        try:
            rating.append(item.find("p", class_="rating").text.strip())
        except:
            rating.append(np.nan)

        try:
            reviews.append(
                item.find("a", class_="review-count").text.strip()
            )
        except:
            reviews.append(np.nan)

        try:
            ctype.append(
                item.find_all("p", class_="infoEntity")[0].text.strip()
            )
        except:
            ctype.append(np.nan)

        try:
            hq.append(
                item.find_all("p", class_="infoEntity")[1].text.strip()
            )
        except:
            hq.append(np.nan)

        try:
            how_old.append(
                item.find_all("p", class_="infoEntity")[2].text.strip()
            )
        except:
            how_old.append(np.nan)

        try:
            no_of_employee.append(
                item.find_all("p", class_="infoEntity")[3].text.strip()
            )
        except:
            no_of_employee.append(np.nan)

    # Build a DataFrame for the current page.
    df = pd.DataFrame({
        "name": name,
        "rating": rating,
        "reviews": reviews,
        "company_type": ctype,
        "Head_Quarters": hq,
        "Company_Age": how_old,
        "No_of_Employee": no_of_employee,
    })

    # Add the current page's data to the complete DataFrame.
    final = pd.concat([final, df], ignore_index=True)


# Inspect the complete scraped dataset.
print(final.sample(5))
print("Shape:", final.shape)