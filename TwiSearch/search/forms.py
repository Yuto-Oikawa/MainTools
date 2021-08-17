from django import forms

class ChkForm(forms.Form):
     CHOICE = [
          ('1','1次情報'),
          ('2','2次情報'),
          ('3','1.5次情報')]
     
     dim = forms.MultipleChoiceField(
          label='結果を表示するカテゴリ',
          required=True,
          disabled=False,
          initial=[],
          choices=CHOICE,
          widget=forms.CheckboxSelectMultiple(attrs={
              'id': 'dim','class': 'form-check-input'}))

class get_numForm(forms.Form):
     get_num = forms.IntegerField(
          label='取得ツイート数',
          required=True,
          min_value=1,
          initial=2000,
          max_value=10000)