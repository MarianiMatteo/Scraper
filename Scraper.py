from apify_client import ApifyClient
from Utility import *
import time


client = ApifyClient("apify_api_CV1KyT7LXglO43g6YtQHvl9t2cdhmD3kRm9h")

#temp_url = "https://www.instagram.com/p/CeRZ7dUtvnb/"

#prepare_for_scraping()
#links = scrappy_automator("post_urls.txt", client, 200)
#print(links)

detect_fake_accounts()
