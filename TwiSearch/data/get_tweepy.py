# coding:utf-8

import tweepy
import datetime
from data import check_API_limit as lim

# 認証キーの設定
Consumer_key = ''
Consumer_secret = ''
Access_token = ''
Access_secret = ""

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

  with open('data/'+word+'.txt', 'w', encoding='utf-8')as f1, open('data/'+word+'_detail.txt', 'w', encoding='utf-8') as f2, open('data/'+word+'_リンク.txt', 'w', encoding='utf-8') as f3: 

    cnt = 0
    for tweet in tweets:
      #if tweet.favorite_count + tweet.retweet_count >= 100:
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
          f1.write(str(tweet.full_text))
          f1.write('+＝＝＝＝＝＝＝＝＝＝+')

          f2.write(str(tweet.id))
          f2.write(' ')
          f2.write(str(tweet.created_at+datetime.timedelta(hours=9)))
          f2.write('\n')
          f2.write(str(tweet.full_text))
          f2.write('\n')
          f2.write('+＝＝＝＝＝＝＝＝＝＝+')
          f2.write('\n')
          #f2.write(str(tweet.favorite_count))
          #f2.write(str(tweet.retweet_count))

          f3.write('https://twitter.com/'+str(tweet.user.screen_name)+'/status/'+str(tweet.id))
          f3.write('+＝＝＝＝＝＝＝＝＝＝+')
          f3.write('\n')

          cnt += 1
          if cnt == total:
            print(cnt)
            break

        else: break



