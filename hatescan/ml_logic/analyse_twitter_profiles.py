import requests
from hatescan.ml_logic.model import initialize_model,train_model,evaluate_model
from hatescan.ml_logic.preprocessor import preprocessing, tokenizer
input = "twitter_username"
n_tweets_retrieved = 10
def analyse_twitter_profile(input:str,n_tweets_retrieved:int):
    #with name_of_twitter_account input, get profile card
    url = "https://twitter135.p.rapidapi.com/v2/UserByScreenName/"
    querystring = {"username":input}
    headers = {
	"X-RapidAPI-Key": "5b75ec0af7msh1bf64a6cc433349p125d64jsn5a59c3ca5bda",
	"X-RapidAPI-Host": "twitter135.p.rapidapi.com"
}
    response = requests.get(url, headers=headers, params=querystring).json()
    user_name = response['data']['user']['result']['legacy']['screen_name']
    nr_followers = response['data']['user']['result']['legacy']['followers_count']
    is_verified = response['data']['user']['result']['legacy']['verified']
    media_count = response['data']['user']['result']['legacy']['media_count']

    #get user's tweets
    rest_id = response['data']['user']['result']['rest_id']
    url = "https://twitter135.p.rapidapi.com/v2/UserTweets/"
    querystring = {"id":rest_id,
                "count":"40"}
    headers = {
        "X-RapidAPI-Key": "5b75ec0af7msh1bf64a6cc433349p125d64jsn5a59c3ca5bda",
        "X-RapidAPI-Host": "twitter135.p.rapidapi.com"
    }
    response_tweets = requests.get(url, headers=headers, params=querystring).json()

    #get list of 10 first tweets
    tweet_list=[]
    try:
        for i in range(0,n_tweets_retrieved):
            tweet_list.append(response_tweets['data']['user']['result']['timeline_v2']['timeline']['instructions'][1]['entries'][i]['content']['itemContent']['tweet_results']['result']['legacy']['full_text'])
    except:
        print("Twitter account does not have enough publications.")
    #clean list
    clean_list = []
    for ele in tweet_list:
        clean_list.append(preprocessing(ele))

###will depend on model and how to call API
    '''X_new = pd.DataFrame(clean_list, colums="clean_tweets") #turn clean tweets into df
    X_new_token, vocab_size = tokenizer(X_new) #tokenize X_new ! token according to model
    X_new_pad = pad_sequences(X_new_token, dtype='float32', padding='post') #Pad X_new

    #initiate model
    model = ? #call model API
'''
    #predict y
    result_list = []
    for ele in X_new_pad:
        result_list.append(model.predict(ele))

    return result_list
