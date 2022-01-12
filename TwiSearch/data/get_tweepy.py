# coding:utf-8

from os import link
import tweepy
import datetime
import pandas as pd
from data import check_API_limit as lim

# 認証キーの設定
Consumer_key = 'D9LR8DyxdTZ7em22ExiJwoa12'
Consumer_secret = 'MgpbmynhbwE97AZVojQ6lALqOAJt3CaFi5tzITLfjfJBOfVEj2'
Access_token = '1261226774224007168-DSQssFaChShIiT7BokDkFTi3soL5Vf'
Access_secret = "9xvphrh4711EFEWIRVV2fPHpci9knnx2xwfg4XPQvy81P"

### TwitterAPI認証用関数
def authTwitter():
  auth = tweepy.OAuthHandler(Consumer_key, Consumer_secret)
  auth.set_access_token(Access_token, Access_secret)
  api = tweepy.API(auth, wait_on_rate_limit = False) # API利用制限にかかった場合、解除まで待機する
  return(api)

### Tweetの検索結果を標準出力
def get_tweets(word, start_date, start_time, end_date, end_time, total):
  api = authTwitter() # 認証
  query = word + ' -filter:retweets'
  
  tweets = tweepy.Cursor(api.search, q = query,         # APIの種類と検索文字列
                        include_entities = True,   # 省略されたリンクを全て取得
                        tweet_mode = 'extended',   # 省略されたツイートを全て取得
                        result_type='recent',
                        lang = 'ja',               # 日本のツイートのみ取得
                        since = start_date+'_'+start_time+'_JST',
                        until = end_date+'_'+end_time+'_JST',
                        ).items()                  


  cnt = 0
  df = pd.DataFrame()
  text_list = []
  date_list = []
  id_list = []
  favo_list = []
  retw_list = []
  link_list = []
  
  for tweet in tweets:
      # print('+＝＝＝＝＝＝＝＝＝＝+')
      # print('twid : ',tweet.id)               # tweetのIDを出力。ユニークなもの
      # print('user : ',tweet.user.screen_name) # ユーザー名
      # print('date : ', tweet.created_at)      # 呟いた日時
      # print(tweet.full_text)                  # ツイート内容
      # print('favo : ', tweet.favorite_count)  # ツイートのいいね数
      # print('retw : ', tweet.retweet_count)   # ツイートのリツイート数

      _, remaining, _ = lim.get_rate_limit_status()
      
      if remaining != 0:

        #if not "RT @" in tweet.full_text[0:4]:
        text_list.append(str(tweet.full_text))
        id_list.append(str(tweet.id))
        date_list.append(str(tweet.created_at+datetime.timedelta(hours=9)))
        favo_list.append(str(tweet.favorite_count))
        retw_list.append(str(tweet.retweet_count))
        link_list.append('https://twitter.com/'+str(tweet.user.screen_name)+'/status/'+str(tweet.id))

        cnt += 1
        if cnt == total:
          print(cnt)
          break

      else: break
      
  df['id'] = id_list
  df['date'] = date_list
  df['link'] = link_list
  df['favo'] = favo_list
  df['retweet'] = retw_list
  df['text'] = text_list

  
  df.to_csv(f'data/{word}.csv',index=False)



