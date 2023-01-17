from apify_client import ApifyClient
from Utility import *
#client = ApifyClient("apify_api_sK0VFtrbVbixtUIQasaJj46H6MbWFL2jitER")

temp_url = "https://www.instagram.com/p/CeRZ7dUtvnb/"

# users = scrappy(client, temp_url, 200)
# data_2_csv(client, users)
links = scrappy_automator("post_urls.txt")
print(links)