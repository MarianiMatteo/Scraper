import csv


# -------------------------------------------- Scrappy --------------------------------------------
# Function that execute the scraping
# INPUT:
#   - post_url: url of the choosen post (String)
#   - result_limit: limit on the number of comments that need to be scraped (int)
# OUTPUT:
#   - run_users: raw data of user's profile 
def scrappy(client, post_url, result_limit):
    # Initialize the ApifyClient with your API token
    # TI PREGO METTI LA TUA API KEY SENNO MI SI BEVONO TUTTI I CREDITI GRATIS
    # api token fra: apify_api_CV1KyT7LXglO43g6YtQHvl9t2cdhmD3kRm9h
    # api token m: apify_api_sK0VFtrbVbixtUIQasaJj46H6MbWFL2jitER
    
    # Prepare the actor input
    run_input_commenti = {
        "directUrls": [post_url],
        "resultsType": "comments",
        "resultsLimit": result_limit,
        "searchType": "hashtag",
        "searchLimit": 1,
        "proxy": {
            "useApifyProxy": True,
            "apifyProxyGroups": ["RESIDENTIAL"],
        },
        "extendOutputFunction": """async ({ data, item, helpers, page, customData, label }) => {
    return item;
    }""",
        "extendScraperFunction": """async ({ page, request, label, response, helpers, requestQueue, logins, addProfile, addPost, addLocation, addHashtag, doRequest, customData, Apify }) => {
    
    }""",
        "customData": {},
    }

    # Run the actor and wait for it to finish
    run_commenti = client.actor("jaroslavhejlek/instagram-scraper").call(run_input=run_input_commenti)

    commenters = []
    # Fetch and print actor results from the run's dataset (if there are any)
    for item in client.dataset(run_commenti["defaultDatasetId"]).iterate_items():
        commenters.append(item['ownerUsername']) if item['ownerUsername'] not in commenters else commenters

    # Create the urls needed to fetch data of the users
    directUrls = []
    for commenter in commenters:
        directUrls.append('https://www.instagram.com/' + commenter + '/') 

    # Prepare the actor's input
    run_input_users = {
        "directUrls": directUrls,
        "resultsType": "details",
        "resultsLimit": 200,
        "searchType": "hashtag",
        "searchLimit": 1,
        "proxy": {
            "useApifyProxy": True,
            "apifyProxyGroups": ["RESIDENTIAL"],
        },
        "extendOutputFunction": """async ({ data, item, helpers, page, customData, label }) => {
    return item;
    }""",
        "extendScraperFunction": """async ({ page, request, label, response, helpers, requestQueue, logins, addProfile, addPost, addLocation, addHashtag, doRequest, customData, Apify }) => {
    
    }""",
        "customData": {},
    }

    # Run the actor
    run_users = client.actor("jaroslavhejlek/instagram-scraper").call(run_input=run_input_users)

    return run_users


# -------------------------------------------- data2csv --------------------------------------------

# Function to convert raw data to csv
# INPUT:
#   - post_url: url of the choosen post (String)
#   - result_limit: limit on the number of comments that need to be scraped (int)
# OUTPUT:
#   - run_users: raw data of user's profile 
def data_2_csv(client, users):
    # Create the data that will be written in the CSV
    headers = ['username', 'follower_num', 'following_num', 'number_videos', 'edge_owner_to_timeline_media', 
    'username_len', 'fullname_len', 'Digits_in_username', 'bio_len',
    'Number_of_nonalphabetic_in_fullname', 'is_private', 'is_verified', 'is_business_account', 'Has_external_url']
    data = []
    for item in client.dataset(users["defaultDatasetId"]).iterate_items():
        row = [None] * 13
        row[0] = item['username']
        row[1] = item['followersCount']
        row[2] = item['followsCount']
        row[3] = item['igtvVideoCount']
        row[4] = item['postsCount']
        row[5] = len(item['username'])
        row[6] = len(item['fullName'])
        row[7] = sum(c.isdigit() for c in item['username'])
        row[8] = sum(c.isdigit() for c in item['bio'])
        row[8] = str(item['fullName']).count(r'[^a-zA-Z0-9 ]')
        row[9] = item['private']
        row[10] = item['verified']
        row[11] = item['isBusinessAccount']
        row[12] = True if item['externalUrl'] else False

        data.append(row)

    # Write the CSV
    with open('dataset_users.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(data)

# -------------------------------------------- Scrappy automator --------------------------------------------

# Function to automate scraping on multiple posts
# INPUT:
#   - file: text file with urls divided by comma or newline
# OUTPUT:
#   - links: python list of post urls
def scrappy_automator(file, client, result_limit):
    #list of post_urls
    post_urls = []
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            # if any "\n" remove it
            if line[len(line)-1:] == "\n":
                line = line[:len(line)-1]
            post_urls.append(line)
    f.close
    
    # scrape all
    users_info = []
    for url in post_urls:
        users_info.append(scrappy(client, url, result_limit))
    
    #convert to csv
    for user_info in users_info:
        data_2_csv(client, user_info)

    return post_urls



