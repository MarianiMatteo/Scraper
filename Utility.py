import os
import re
import csv
import json
import emoji
import shutil
import joblib
import requests
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statistics import mean
from Comment import Comment
from datetime import datetime
from googletrans import Translator
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from emosent import get_emoji_sentiment_rank
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn import datasets, metrics, model_selection, svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from statsmodels.graphics.mosaicplot import mosaic
#from sklearn.metrics import classification_report, confusion_matrix, plot_roc_curve,roc_auc_score, roc_curve, accuracy_score
from requests.structures import CaseInsensitiveDict
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd


def print_dictionary(dictionary):
    for key, value in dictionary.items():
        print(key, ' : ', value)


def print_dictionary_last_n(dictionary, n):
    for x in list(reversed(list(dictionary)))[0:n]:
        print(x, ' : ', dictionary[x])


def extract_emojis(s):
    return ''.join(c for c in s if c in emoji.EMOJI_DATA)

def give_emoji_free_text(text):
    allchars = [str for str in text]
    emoji_list = [c for c in allchars if c in emoji.EMOJI_DATA]
    clean_text = ' '.join([str for str in text.split() if not any(i in str for i in emoji_list)])
        
    return clean_text

def give_emoji_free_text(text):
    allchars = [str for str in text]
    emoji_list = [c for c in allchars if c in emoji.EMOJI_DATA]
    clean_text = ' '.join([str for str in text.split() if not any(i in str for i in emoji_list)])
        
    return clean_text


def orderTime(oggetto):
    return oggetto.timestamp


def get_sentiment(comments):
    scores = []
    sia = SentimentIntensityAnalyzer()
    for commenter, comments_user in comments.items():
        score_user = []
        for comment in comments_user:
            score_user.append(sia.polarity_scores(comment)['compound'])

        scores.append(mean(score_user))

    return mean(scores)


def get_sentiment_graph(comments_list):
    true_comments_list = []
    false_comments_list = []
    y_scores_true = []
    y_scores_false = []
    y_scores = []
    comments_so_far = []
    true_comments_so_far = []
    false_comments_so_far = []
    sia = SentimentIntensityAnalyzer()

    for comment in comments_list:
        comment.timestamp = pd.to_datetime(comment.timestamp, infer_datetime_format=True)
        comment.text = give_emoji_free_text(comment.text)
        if len(comment.text) == 0:
            comment.to_remove = True

    i = 0
    while i < len(comments_list):
        if comments_list[i].to_remove == True:
            comments_list.pop(i)
        i+=1

    x = range(0,len(comments_list))
    comments_list = list(filter(lambda item: item is not None, comments_list))

    comments_list.sort(key=orderTime)
    for comment_list in comments_list:
        comments_so_far.append(comment_list.text)

        if comment_list.category == 'crowdturfing':
            false_comments_list.append(comment_list)
        else:
            true_comments_list.append(comment_list)

        scores_so_far = []
        for comment in comments_so_far:
            scores_so_far.append(sia.polarity_scores(comment)['compound'])

        y_scores.append(sum(scores_so_far)/len(scores_so_far))

    for comment_list in false_comments_list:
        false_comments_so_far.append(comment_list.text)
        scores_so_far = []
        for comment in false_comments_so_far:
            scores_so_far.append(sia.polarity_scores(comment)['compound'])

        y_scores_false.append(sum(scores_so_far)/len(scores_so_far))

    x_scores_false = range(0, len(false_comments_list))

    for comment_list in true_comments_list:
        true_comments_so_far.append(comment_list.text)
        scores_so_far = []
        for comment in true_comments_so_far:
            scores_so_far.append(sia.polarity_scores(comment)['compound'])

        y_scores_true.append(sum(scores_so_far)/len(scores_so_far))

    x_scores_true = range(0, len(true_comments_list))

    fig, axs = plt.subplots(ncols=3)
    
    fig.suptitle('Sentiment over time')
    axs[0].plot(x, y_scores)
    axs[0].set_title('Overall sentiment')
    plt.sca(axs[0])
    plt.yticks(np.arange(-1.0,1.1, 0.4))
    axs[1].plot(x_scores_true, y_scores_true)
    axs[1].set_title('True sentiment')
    plt.sca(axs[1])
    plt.yticks(np.arange(-1.0,1.1, 0.4))
    axs[2].plot(x_scores_false, y_scores_false)
    axs[2].set_title('Crowdturfing Sentiment')
    plt.sca(axs[2])
    plt.yticks(np.arange(-1.0,1.1, 0.4))

    for ax in axs.flat:
        ax.set(xlabel='Number of comments',ylabel='Sentiment')

    plt.show()

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


def get_topic_from_comments(dict_comments):
    stop_words = set(stopwords.words('english'))
    with open('STTM/corpus.txt', 'a') as outfile:
        for commenter, comments in dict_comments.items():
            for comment in comments:
                word_tokens = word_tokenize(comment)
                filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words and w.isalnum()]
                filtered_sentence = TreebankWordDetokenizer().detokenize(filtered_sentence)
                if len(filtered_sentence) > 0:
                    outfile.write(filtered_sentence)
                    outfile.write('\n')

def create_wordcloud(path):
        
    stopwords = set(STOPWORDS)
    
    lines = []
    text = ""

    with open(path) as f:
        lines = f.readlines()
    
    for line in lines:
        text_list = line.split(" ")
        for wrd in text_list:
            if wrd != "\n":
                text = text + " " + wrd.lower()
    

    wordcloud = WordCloud(width = 800, height = 800,
                    background_color ='white',
                    stopwords = stopwords,
                    min_font_size = 10).generate(text)
    
    # plot the WordCloud image                      
    plt.figure(figsize = (6, 6), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 2)
    
    #save as png
    plt.savefig("word_cloud.png") 
    plt.show()

# -------------------------------------------- Scrappy --------------------------------------------
def prepare_for_scraping():
    open('comments.txt', 'w').close()
    open('likers.json', 'w').close()
    open('commenters.json', 'w').close()

# Function that execute the scraping
# INPUT:
#   - post_url: url of the choosen post (String)
#   - result_limit: limit on the number of comments that need to be scraped (int)
#   - likers_or_commenters: 0 if likers 1 if commenters
# OUTPUT:
#   - run_users: raw data of user's profile 
def scrappy(api_key, client, post_url, likers_or_commenters, debug=False):

    content = ""
    # likers case
    if likers_or_commenters == 0:
        content = "reactions"
    #commenters case
    else:
        content = "comments"

    slug = post_url.split('/')[4]
    url = "https://api.zembra.io/social/instagram/post/"+slug+"/"+content

    headers = CaseInsensitiveDict()
    headers["Accept"] = "application/json"
    headers["Authorization"] = "Bearer " + api_key

    if debug == False:
        resp = requests.get(url, headers=headers)

    if likers_or_commenters == 0:
        file_name = "likers"
    else:
        file_name = "commenters"
   
    if debug == False:
        with open(file_name + ".json", "a") as outfile:
            json.dump(resp.json(), outfile)
    
    if debug:
        profile_usernames = []
        if likers_or_commenters == 0:
            with open('likers.json', 'r') as file_json:
                data = json.load(file_json)
                if "reactions" in data["data"].keys():
                    for commenter in data['data']['reactions']:
                        profile_usernames.append(commenter["username"]) if commenter["username"] not in profile_usernames else profile_usernames
        else:
            with open('commenters.json', 'r') as file_json:
                data = json.load(file_json)
                if "comments" in data["data"].keys():
                    for commenter in data['data']['comments']:
                        profile_usernames.append(commenter["user"]) if commenter["user"] not in profile_usernames else profile_usernames
    else:
        profile_usernames = []
        if likers_or_commenters == 0:
            for commenter in resp.json()['data']['reactions']:
                profile_usernames.append(commenter["username"]) if commenter["username"] not in profile_usernames else profile_usernames
        else:
            if "comments" in resp.json()["data"].keys():
                for commenter in resp.json()['data']['comments']:
                    profile_usernames.append(commenter["user"]) if commenter["user"] not in profile_usernames else profile_usernames
    
    directUrls = []
    for username in profile_usernames:
        directUrls.append('https://www.instagram.com/' + username + '/') 

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
def data_2_csv(client, users, likers):
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

    if likers:
        # Write the CSV for likers
        with open('dataset_likers.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(data)
    else:
        # Write the CSV for commenters
        with open('dataset_commmenters.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(data)


# -------------------------------------------- Likerz --------------------------------------------

# Function that read the file containing the likers
# INPUT:
#   - file: csv file containing the likers
# OUTPUT:
#   - likers_urls: list of likers' urls  
def read_likers_csv(file):
    likers_urls = []
    with open(file, 'r') as likers_info:
        csv_reader = csv.reader(likers_info, delimiter=',')
        # lo so, non è bello da vedere, ma csv_reader è un oggetto csv.reader
        # e non so come altro saltare il primo elemento
        i = 0
        for row in csv_reader:
            if i == 0:
                i+=1
            else:
                likers_urls.append(row[0]+'/')

    return likers_urls

# -------------------------------------------- Scrappy automator --------------------------------------------

# Function to automate scraping on multiple posts
# INPUT:
#   - file: text file with urls divided by comma or newline
# OUTPUT:
#   - links: python list of post urls
def scrappy_automator(urls_post_file, api_key, client):
    #list of post_urls
    post_urls = []
    with open(urls_post_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            # if any "\n" remove it
            if line[len(line)-1:] == "\n":
                line = line[:len(line)-1]
            post_urls.append(line)
    f.close
    
    # scraping
    users_comments_info = []
    users_likes_info = []
    for url in post_urls:

        users_comments_info.append(scrappy(api_key, client, url, 1, True))
        users_likes_info.append(scrappy(api_key, client, url, 0, True))

    #convert to csv
    data_2_csv(client, users_comments_info, False)
    data_2_csv(client, users_likes_info, True)

    return post_urls


def detect_fake_accounts():
    random_forest = joblib.load('random_forest_model.joblib')

    for file_input in ['dataset_commmenters.csv', 'dataset_likers.csv']: 
        dataset = pd.read_csv(file_input)
        if file_input == 'dataset_commmenters.csv' and len(dataset) == 0:
            print("--------------------------------- NO COMMENTS UNDER THIS POST ---------------------------------")
            continue
        elif file_input == 'dataset_likers.csv' and len(dataset) == 0:
            print("--------------------------------- NO LIKES UNDER THIS POST ---------------------------------")
            break
        column_usernames = ['username']
        usernames = dataset.loc[:,column_usernames]
        dataset = dataset.drop(columns=['username'])
        print(len(dataset))

        predictions = random_forest.predict(dataset)

        # ------------ Commenters section ------------
        if file_input == 'dataset_commmenters.csv':

            fake_comments = {}
            fake_comments_emojis = {}
            true_comments = {}
            true_comments_emojis = {}
            translator = Translator()
            comments_list = []
            number_fake_comments = 0
            with open('commenters.json', 'r') as file_json:
                comments = json.load(file_json)
                for comment in comments['data']['comments']:
                    for i in range(len(predictions)):
                        # se il profilo è segnamato come falso
                        if predictions[i] == 0:
                            # se il commento che sto guardando corrisponde al nome presente nelle predizioni
                            if comment['rawData']['owner']['username'] == usernames.loc[i].at['username']:
                                # se il commentatore non è già stato inserito nel dizionario
                                if comment['rawData']['owner']['username'] not in fake_comments:
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
                                    fake_comments[comment['rawData']['owner']['username']] = [translated_comment]
                                    # increment the number of comments found
                                    number_fake_comments += 1
                                    comments_list.append(Comment('crowdturfing', translated_comment, comment['timestamp']))

                                # se il commentatore era presente nel dizionario
                                else:
                                    translated_comment = translator.translate(comment['text'],dest='en').text
                                    emojis = extract_emojis(translated_comment)
                                    for emoji in emojis:
                                        if emoji not in fake_comments_emojis:
                                            fake_comments_emojis[emoji] = 1
                                        else:
                                            fake_comments_emojis[emoji] += 1

                                    fake_comments[comment['rawData']['owner']['username']].append(translated_comment)
                                    number_fake_comments += 1
                                    comments_list.append(Comment('crowdturfing', translated_comment, comment['timestamp']))
                        # se il profilo è segnalato come vero
                        else:
                            if comment['rawData']['owner']['username'] == usernames.loc[i].at['username']:
                                if comment['rawData']['owner']['username'] not in true_comments:
                                    translated_comment = translator.translate(comment['text'],dest='en').text
                                    emojis = extract_emojis(translated_comment)
                                    for emoji in emojis:
                                        if emoji not in true_comments_emojis:
                                            true_comments_emojis[emoji] = 1
                                        else:
                                            true_comments_emojis[emoji] += 1

                                    true_comments[comment['rawData']['owner']['username']] = [translated_comment]
                                    comments_list.append(Comment('true', translated_comment, comment['timestamp']))

                                elif comment['rawData']['owner']['username'] in true_comments and translator.translate(comment['text'],dest='en').text not in true_comments[comment['rawData']['owner']['username']]:
                                    translated_comment = translator.translate(comment['text'],dest='en').text
                                    emojis = extract_emojis(translated_comment)
                                    for emoji in emojis:
                                        if emoji not in true_comments_emojis:
                                            true_comments_emojis[emoji] = 1
                                        else:
                                            true_comments_emojis[emoji] += 1

                                    true_comments[comment['rawData']['owner']['username']].append(translated_comment)
                                    comments_list.append(Comment('true', translated_comment, comment['timestamp']))


            open('STTM/corpus.txt', 'w').close()
            folder = os.getcwd()+'/STTM/results'
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))

            get_topic_from_comments(fake_comments)
            get_topic_from_comments(true_comments)
            os.chdir(os.getcwd()+'/STTM')
            os.system('./run.sh')
            os.chdir('../')

            create_wordcloud('STTM/results/model.topWords')
        
            os.system('clear')
            total_comments = len(dataset.index)
            print('-------------------------------- COMMENTERS --------------------------------')
            print('Numero totale commenti: ', total_comments )
            print()
            print('Numero di commenti veri: ', total_comments-number_fake_comments)
            print()
            print('Numero di commenti falsi: ', number_fake_comments)
            print()
            print('Numero totale di commenter veri: ', len(true_comments))
            print()
            print('Numero totale di commenter falsi: ', len(fake_comments))
            print()
            print('----------------------------------------------------------------------------')
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
            get_sentiment_graph(comments_list)
            print('----------------------------------------------------------------------------')
            print('Percentuale di fake engagement (basata sui commenti): ', round(((number_fake_comments / total_comments) * 100),2), '%')
            print()
            print('Percentuale di engagement reale (basata sui commenti): ', round(((1 - (number_fake_comments / total_comments)) * 100),2), '%')
            print()
            print('Percentuale di fake engagement (basata sui commenter): ', round(((len(fake_comments) / (len(fake_comments) + len(true_comments))) *100),2),'%')
            print()
            print('Percentuale di engagement reale (basata sui commenter): ', round(((len(true_comments) / (len(fake_comments) + len(true_comments))) *100),2),'%')
            print()
            print('----------------------------------------------------------------------------')
            print('Most used emojis by crowdturfing accounts')
            fake_comments_emojis = {k: v for k, v in sorted(fake_comments_emojis.items(), key=lambda item: item[1])}
            print_dictionary_last_n(fake_comments_emojis, 5)
            print()
            print('Most used emojis by real accounts')
            true_comments_emojis = {k: v for k, v in sorted(true_comments_emojis.items(), key=lambda item: item[1])}
            print_dictionary_last_n({k: v for k, v in sorted(true_comments_emojis.items(), key=lambda item: item[1])}, 5)
            print()
        # ------------ Likers section ------------
        else:
            number_true_likers = 0
            number_fake_likers = 0
            for i in range(len(predictions)):
                # se il profilo è segnamato come falso
                if predictions[i] == 0:
                    number_fake_likers += 1
                else:
                    number_true_likers += 1

            total_likers = number_true_likers + number_fake_likers

            print('--------------------------------- LIKERS -----------------------------------')
            print('Numero totale likers: ', total_likers)
            print()
            print('Numero di fake likers: ', number_fake_likers)
            print()
            print('Numero di true likers: ', number_true_likers)
            print()
            print('----------------------------------------------------------------------------')
            print('Percentuale di fake engagement (basata sui like): ', round(((number_fake_likers / total_likers) * 100),2), '%')
            print()
            print('Percentuale di engagement reale (basata sui like): ', round(((1 - (number_fake_likers / total_likers)) * 100),2), '%')
            print()
            print('----------------------------------------------------------------------------')
