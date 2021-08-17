#coding:utf-8
 
import argparse
import MeCab
 
# 実行引数の受け取り
parser = argparse.ArgumentParser(description="MeCabで分かち書きを行うためのスクリプトです")
parser.add_argument('input', type=str, help="入力ファイルへのPath(必須)")
parser.add_argument('-o', '--output', type=str, default="data/lemma.txt", help="出力ファイルへのPath(Default:out.txt)")
parser.add_argument('-t', '--target', type=str, nargs='+', default=[], help="出力する品詞(Default:ALL)")
parser.add_argument('-e', '--exclusion', type=str, nargs='+', default=["記号","BOS/EOS"], help="解析時に除外する品詞(Default:記号,BOS/EOS)")
parser.add_argument('-e1','--exclusion1', type=str, nargs='+', default=[], help="解析時に除外する品詞細分類1(Default:None)")
parser.add_argument('-e2','--exclusion2', type=str, nargs='+', default=[], help="解析時に除外する品詞細分類2(Default:None)")
parser.add_argument('-e3','--exclusion3', type=str, nargs='+', default=[], help="解析時に除外する品詞細分類3(Default:None)")
parser.add_argument('-s', '--splitchar', type=str, default=' ', help="分かち書きを行う際に各形態素の間に挿入される文字です(Default:Space)")
parser.add_argument('-m', '--outputmode', type=str, default='genkei', choices=['hyousou', 'genkei', 'yomi'], help="出力する形態素の形式を設定します 表層形:hyousou,原形:genkei,読み:yomi(Default:genkei)")
args = parser.parse_args()
 
# 書き換え規則の定義
# 基本的には単純なディクショナリ型(キー:書き換える品詞, 値:書き換え後の単語)
# 書き換える品詞は入れ子状に記述可能(品詞→品詞細分類1→品詞細分類2→品詞細分類3の順で入れ子にしていく)
# Default 	1) "名詞"でかつ品詞細分類1が"数"ならば"[数値]"に単語を置き換えて出力
#			2) "名詞"でかつ品詞細分類1が"固有名詞"でかつ品詞細分類2が"組織"ならば"[固有名詞_組織]"に単語を置き換えて出力
 
# 入力ファイルと出力ファイルの準備
in_file = open(args.input)
out_file = open(args.output, 'w')
 
# MeCabのインスタンス作成
mecab = MeCab.Tagger('-Ochasen')
mecab.parseToNode('')				# MeCab上の不具合で一度空で解析すると解消される
 
# 一行毎に解析開始
for line in in_file:
	text = ''
	node = mecab.parseToNode(line)
 
	# 一単語毎に解析
	while node:
 
		# 形態素のノードから形態素情報を取得
		# 得られる形態素情報は基本的に以下の順番のリストになっている
		# [品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用形,活用型,原形,読み,発音]
		word_surface = node.surface
		word_features = node.feature.split(',')
		features = node.feature.split(',')

		# 単語の置き換え規則の確認及び実行
		local_replace_rule = {
            '名詞': word_features[6]+'|名詞',
            '動詞': word_features[6]+'|動詞',
            '形容詞': word_features[6]+'|形容詞',
            '形容動詞': word_features[6]+'|形容動詞',
            '副詞': word_features[6]+'|副詞',
            '助詞': word_features[6]+'|助詞',
            '助動詞': word_features[6]+'|助動詞',
            '連体詞': word_features[6]+'|連体詞',
            '接続詞': word_features[6]+'|接続詞',
            '接頭詞': word_features[6]+'|接頭詞',
            '感動詞': word_features[6]+'|感動詞'
        }

		for x in range(3):
			if word_features[x] in local_replace_rule:
				if isinstance(local_replace_rule[word_features[x]], dict):
					local_replace_rule = local_replace_rule[word_features[x]]
				elif isinstance(local_replace_rule[word_features[x]], str):
					word_surface = local_replace_rule[word_features[x]]
					word_features = word_features[:5] + [word_surface, word_surface, word_surface]
					break
 
		# 解析対象の品詞か確認
		if (not len(args.target) or word_features[0] in args.target) 	and \
				word_features[0] not in args.exclusion 					and	\
				word_features[1] not in args.exclusion1 				and \
				word_features[2] not in args.exclusion2					and \
				word_features[3] not in args.exclusion3:
 
			# 指定された形式で出力に追加
			if args.outputmode == 'hyousou':
				word = word_surface
			elif args.outputmode == 'genkei':
				word = word_features[6]
			elif args.outputmode == 'yomi':
				word = word_features[7]
 
			# 未定義(辞書内で"*"と表記される)の場合は出力しない
			if features[6] is '*':
				text += node.surface
			else :
				text += word + args.splitchar
 
		node = node.next
 
	# 解析した行に形態素があれば出力ファイルに記述
	if text is not '':
		out_file.write(text + "\n")
 
in_file.close()
out_file.close()
