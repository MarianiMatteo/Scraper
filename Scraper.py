from apify_client import ApifyClient
from Utility import *
import time

api_key = "INSERT HERE YOUR ZEMBRA KEY"
client = ApifyClient("INSERT HERE YOUR APIFY KEY")
#temp_url = "https://www.instagram.com/p/CeRZ7dUtvnb/"

prepare_for_scraping()
links = scrappy_automator("post_urls.txt", api_key, client)

detect_fake_accounts()
