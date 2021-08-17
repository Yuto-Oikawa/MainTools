from django.test import TestCase
from .models import Tweet
from django.urls import reverse


def create_tweet(text, label):
    return Tweet.objects.create(text=text, label=label, link='https://twitter.com/')

class IndexViewTest(TestCase):
    def test_display(self):
        response = self.client.get(reverse('index'))
        self.assertEqual(response.status_code, 200)

class RedrawViewTest(TestCase):
    def test_display(self):
        response = self.client.get(reverse('redraw'))
        self.assertEqual(response.status_code, 200)
    
    def test_context(self):
        create_tweet('test', 0)
        response = self.client.get(reverse('redraw'))
        self.assertQuerysetEqual(
            response.context['tweet_list'],['<Tweet: test>']
        )
        self.assertContains(response, "1件を取得しました。")