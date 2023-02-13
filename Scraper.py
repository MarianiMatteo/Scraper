from apify_client import ApifyClient
from Utility import *
import time


# client = ApifyClient("apify_api_CV1KyT7LXglO43g6YtQHvl9t2cdhmD3kRm9h")
# Fgl8ujy0qgiL7aiGsHGPcbmZi1fk2Tu8T9b8K4OFXmPdvbKlp599jzTt73RbqnDX1GL2cYfCLXEXSSzTvGt2yKfWXhRolB6oDbva0qwLXg7jNK6XNPiosVIA5f4eHaaF
api_key = "uEq8jDPC08AzZF17URS6cRN3hMGsxHmLgIgP5NlFZJMg4WGrl14dwGVAxvahIFay5nxEQyVaLbVmZL1zmP7DJZpdZmjyYJJNWBQGfdQxSJPWA5Gk7R3A8cgs70O3Ikn0"
client = ApifyClient("apify_api_CV1KyT7LXglO43g6YtQHvl9t2cdhmD3kRm9h")
#temp_url = "https://www.instagram.com/p/CeRZ7dUtvnb/"

#prepare_for_scraping()
links = scrappy_automator("post_urls.txt", api_key, client)
#print(links)

detect_fake_accounts()
