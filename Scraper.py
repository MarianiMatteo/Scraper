from apify_client import ApifyClient
from Utility import *
client = ApifyClient("apify_api_sK0VFtrbVbixtUIQasaJj46H6MbWFL2jitER")

temp_url = "https://www.instagram.com/p/CeRZ7dUtvnb/"

links = scrappy_automator("post_urls.txt", client, 200)
print(links)