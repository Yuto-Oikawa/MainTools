from django.db import models

class Tweet(models.Model):
    text = models.CharField('テキスト', max_length=140)
    label = models.IntegerField('ラベル')
    link = models.URLField('リンク', default='https://twitter.com/')

    def __str__(self):
        return self.text