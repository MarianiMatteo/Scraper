import csv
import json
import emoji
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statistics import mean
from googletrans import Translator
from emosent import get_emoji_sentiment_rank
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn import datasets, metrics, model_selection, svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from statsmodels.graphics.mosaicplot import mosaic
from sklearn.metrics import classification_report, confusion_matrix, plot_roc_curve,roc_auc_score, roc_curve, accuracy_score


def print_dictionary(dictionary):
    for key, value in dictionary.items():
        print(key, ' : ', value)


def print_dictionary_last_n(dictionary, n):
    for x in list(reversed(list(dictionary)))[0:n]:
        print(x, ' : ', dictionary[x])


def extract_emojis(s):
    return ''.join(c for c in s if c in emoji.EMOJI_DATA)


def get_sentiment(comments):
    scores = []
    sia = SentimentIntensityAnalyzer()
    for commenter, comments_user in comments.items():
        score_user = []
        for comment in comments_user:
            score_user.append(sia.polarity_scores(comment)['compound'])

        scores.append(mean(score_user))

    
    return mean(scores)


def get_emoji_sentiment(dict_emoji):
    scores = []
    n = 5
    for x in list(reversed(list(dict_emoji)))[0:n]:
        try:
            scores.append(get_emoji_sentiment_rank(x)['sentiment_score'])
        except:
            # errore emoji non catalogata
            n += 1
            continue

    return mean(scores)

# -------------------------------------------- Scrappy --------------------------------------------
def prepare_for_scraping():
    open('comments.txt', 'w').close()

# Function that execute the scraping
# INPUT:
#   - post_url: url of the choosen post (String)
#   - result_limit: limit on the number of comments that need to be scraped (int)
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
    with open('comments.txt','a') as f:
        for item in client.dataset(run_commenti["defaultDatasetId"]).iterate_items():
            f.write(json.dumps(item))
            f.write('\n')
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
    headers = ['username','follower_num', 'following_num', 'is_private', 'is_verified', 'has_clips','highlight_reel_count', 'is_business_account', 'edge_felix_video_timeline', 'edge_owner_to_timeline_media', 'username_len', 'fullname_len', 'bio_len', 'Digits_in_username',  'Number_of_nonalphabetic_in_fullname', 'Number_of_HashtagsMentions','Has_external_url', 'Has_business_category_name', 'Has_category_enum', 'Has_category_name']
    data = []
    for user in users:
        for item in client.dataset(user["defaultDatasetId"]).iterate_items():
            row = [None] * 20
            row[0] = item['username']
            row[1] = item['followersCount']
            row[2] = item['followsCount']
            row[3] = item['private']
            row[4] = item['verified']
            row[5] = False if item['igtvVideoCount'] == 0 and item['highlightReelCount'] == 0 else True
            row[6] = item['highlightReelCount']
            row[7] = item['isBusinessAccount']
            row[8] = item['igtvVideoCount']
            row[9] = item['postsCount']
            row[10] = len(item['username'])
            row[11] = len(item['fullName'])
            row[12] = len(item['biography'])
            row[13] = sum(c.isdigit() for c in item['username'])
            row[14] = str(item['fullName']).count(r'[^a-zA-Z0-9 ]')
            hashtag_mentions = item['biography'].count('#')
            hashtag_mentions += item['biography'].count('@')
            row[15] = hashtag_mentions
            row[16] = False if item['externalUrl'] == None else True
            row[17] = True if item['businessCategoryName'] else False
            row[18] = True if item['businessCategoryName'] else False
            row[19] = True if item['businessCategoryName'] else False

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
    data_2_csv(client, users_info)

    return post_urls


def detect_fake_accounts():
    random_forest = joblib.load('random_forest_model.joblib')

    dataset = pd.read_csv('dataset_users.csv')
    column_usernames = ['username']
    usernames = dataset.loc[:,column_usernames]
    dataset = dataset.drop(columns=['username'])

    predictions = random_forest.predict(dataset)

    fake_comments = {}
    fake_comments_emojis = {}
    true_comments = {}
    true_comments_emojis = {}
    translator = Translator()
    number_fake_comments = 0
    with open('comments.txt', 'r') as f:
        for comment in f:
            comment = json.loads(comment)
            for i in range(len(predictions)):
                # se il profilo è segnamato come falso
                if predictions[i] == 0:
                    # se il commento che sto guardando corrisponde al nome presente nelle predizioni
                    if comment['ownerUsername'] == usernames.loc[i].at['username']:
                        # se il commentatore non è già stato inserito nel dizionario
                        if comment['ownerUsername'] not in fake_comments:
                            # translate the comment in english
                            translated_comment = translator.translate(comment['text'],dest='en').text
                            # extract emojis from the comment
                            emojis = extract_emojis(translated_comment)
                            for emoji in emojis:
                                if emoji not in fake_comments_emojis:
                                    fake_comments_emojis[emoji] = 1
                                else:
                                    fake_comments_emojis[emoji] += 1
                            # insert the comment in the final dictionary
                            fake_comments[comment['ownerUsername']] = [translated_comment]
                            # increment the number of comments found
                            number_fake_comments += 1

                        # se il commentatore era presente nel dizionario
                        else:
                            translated_comment = translator.translate(comment['text'],dest='en').text
                            emojis = extract_emojis(translated_comment)
                            for emoji in emojis:
                                if emoji not in fake_comments_emojis:
                                    fake_comments_emojis[emoji] = 1
                                else:
                                    fake_comments_emojis[emoji] += 1

                            fake_comments[comment['ownerUsername']].append(translated_comment)
                            number_fake_comments += 1
                # se il profilo è segnalato come vero
                else:
                    if comment['ownerUsername'] == usernames.loc[i].at['username']:
                        if comment['ownerUsername'] not in true_comments:
                            translated_comment = translator.translate(comment['text'],dest='en').text
                            emojis = extract_emojis(translated_comment)
                            for emoji in emojis:
                                if emoji not in true_comments_emojis:
                                    true_comments_emojis[emoji] = 1
                                else:
                                    true_comments_emojis[emoji] += 1

                            true_comments[comment['ownerUsername']] = [translated_comment]
                        elif comment['ownerUsername'] in true_comments and translator.translate(comment['text'],dest='en').text not in true_comments[comment['ownerUsername']]:
                            translated_comment = translator.translate(comment['text'],dest='en').text
                            emojis = extract_emojis(translated_comment)
                            for emoji in emojis:
                                if emoji not in true_comments_emojis:
                                    true_comments_emojis[emoji] = 1
                                else:
                                    true_comments_emojis[emoji] += 1

                            true_comments[comment['ownerUsername']].append(translated_comment)


    #print('Commenti falsi:')
    #print_dictionary(fake_comments)
    #print()
    #print('Commenti veri:')
    #print_dictionary(true_comments)
    #print()
    total_comments = len(dataset.index)
    print('-------------------------------------------------------------------')
    print('LEGENDA SENTIMENT:')
    print('In un intorno di 0 -> generalmente neutrale \nMaggiore di zero -> generalmente positivo \nTendente ad uno -> molto positivo \nMinore di zero -> generalmente negativo \nTendente a meno uno -> molto negativo \n')
    print('Sentiment utenti crowdturfing: ', get_sentiment(fake_comments))
    print()
    print('Sentiment utenti reali: ', get_sentiment(true_comments))
    print()
    print('Sentiment emoji utenti crowdturfing: ', round(get_emoji_sentiment(fake_comments_emojis),2))
    print()
    print('Sentiment emoji utenti reali: ', round(get_emoji_sentiment(true_comments_emojis),2))
    print()
    print('-------------------------------------------------------------------')
    print('Percentuale di fake engagement: ', (number_fake_comments / total_comments) * 100, '%')
    print()
    print('Percentuale di engagement reale: ', (1 - (number_fake_comments / total_comments)) * 100, '%')
    print()
    print('-------------------------------------------------------------------')
    print('Most used emojis by crowdturfing accounts')
    fake_comments_emojis = {k: v for k, v in sorted(fake_comments_emojis.items(), key=lambda item: item[1])}
    print_dictionary_last_n(fake_comments_emojis, 5)
    print()
    print('Most used emojis by real accounts')
    true_comments_emojis = {k: v for k, v in sorted(true_comments_emojis.items(), key=lambda item: item[1])}
    print_dictionary_last_n({k: v for k, v in sorted(true_comments_emojis.items(), key=lambda item: item[1])}, 5)
    print('-------------------------------------------------------------------')
