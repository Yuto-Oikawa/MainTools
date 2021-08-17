import os
import sys
import re
import time
import unicodedata
import concurrent.futures
from math import ceil
from threading import Thread, Lock

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
import pyperclip as ppc


def html_gen(title, filename):
    global tr
    source = ""
    result = ""
    tr = sorted(tr, key=lambda x: x["order"])
    
    for item in tr:
        i, s, r = item.values()
        source += f'<br><a id="s{i}" onmouseover="over(' + f"'s{i}', 'r{i}'" + ')" onmouseout="out(' + f"'r{i}'" + f')">{s}</a><br>'
        result += f'<br><a id="r{i}" onmouseover="over(' + f"'r{i}', 's{i}'" + ')" onmouseout="out(' + f"'s{i}'" + f')">{r}</a><br>'
        
    with open(filename + ".html", "w", encoding='utf-8') as f:
        f.write(
            f'<meta charset="utf-8"/> <h1align="center">{title}</h1>\n<input id="btn-mode" type="checkbox">\n<hr>\n<body>\n<div class="parent">\n<div id="source">\n{source}\n</div>\n<div id="result">\n{result}\n</div>\n</div>'
            +
            '<style>\n:root {\n--main-text: #452b15;\n--main-bg: #f8f1e2;\n--highlight-text: #db8e3c;\n}\n:root[theme="dark"] {\n--main-text: #b0b0b0;\n--main-bg: #121212;\n--highlight-text: #fd8787;\n}\nh1 {\ncolor: var(--main-text);\n}\ninput {\nposition: absolute;\ntop: 1%;\nright: 1%;\n}\n#source {\nwidth: 43%;\nheight: 90%;\npadding: 0 2%;\nfloat: left;\nborder-right:1px solid #ccc;\nmargin: 1%;\noverflow: auto;\n}\n#result {\nwidth: 43%;\nheight: 90%;\npadding: 0 2%;\nfloat: right;\nmargin: 1%;\noverflow: auto;\n}\na,\na:hover,\na:visited,\na:link,\na:active {\ncolor: var(--main-text);\ntext-decoration: none;\n}\nbody {\nbackground-color: var(--main-bg);\n}\n</style>\n<script>\nvar a = document.getElementsByTagName("a");\nfunction over(s,o) {\nvar elements = document.getElementById(s);\nvar elemento = document.getElementById(o);\nvar rects = elements.getBoundingClientRect();\nvar recto = elemento.getBoundingClientRect();\nvar x = recto.left;\nvar y = recto.top - rects.top;\nelemento.parentNode.scrollBy(x, y);\nelemento.style.color = getComputedStyle(elemento).getPropertyValue("--highlight-text");\n}\nfunction out(e) {\ndocument.getElementById(e).style.color = getComputedStyle(document.getElementById(e)).getPropertyValue("--main-text");\n}\nconst btn = document.querySelector("#btn-mode");\nbtn.addEventListener("change", () => {\nif (btn.checked == true) {\ndocument.documentElement.setAttribute("theme", "dark");\n} else {\ndocument.documentElement.setAttribute("theme", "light");\n}\nfor (var i = 0; i < a.length; i++) {\na[i].style.color = getComputedStyle(a[i]).getPropertyValue("--main-text");\n}\n});\n</script>\n</body>'
        )


def len_(text):
    cnt = 0
    for t in text:
        if unicodedata.east_asian_width(t) in "FWA":
            cnt += 2
        else:
            cnt += 1
    return cnt


def textParser(text, n=30, bracketDetect=True):
    text = text.splitlines()
    sentences = []
    t = ""
    bra_cnt = ket_cnt = bra_cnt_jp = ket_cnt_jp = 0
    
    for i in range(len(text)):
        
        if not bool(re.search("\S", text[i])): continue
        if bracketDetect:
            bra_cnt += len(re.findall("[\(（]", text[i]))
            ket_cnt += len(re.findall("[\)）]", text[i]))
            bra_cnt_jp += len(re.findall("[｢「『]", text[i]))
            ket_cnt_jp += len(re.findall("[｣」』]", text[i]))
            
        if i != len(text) - 1:
            
            if bool(re.fullmatch(r"[A-Z\s]+", text[i])):
                if t != "": sentences.append(t)
                t = ""
                sentences.append(text[i])
            elif bool(
                    re.match(
                        "(\d{1,2}[\.,、．]\s?(\d{1,2}[\.,、．]*)*\s?|I{1,3}V{0,1}X{0,1}[\.,、．]|V{0,1}X{0,1}I{1,3}[\.,、．]|[・•●])+\s",
                        text[i])) or re.match("\d{1,2}．\w", text[i]) or (
                            bool(re.match("[A-Z]", text[i][0]))
                            and abs(len_(text[i]) - len_(text[i + 1])) > n
                            and len_(text[i]) < n):
                if t != "": sentences.append(t)
                t = ""
                sentences.append(text[i])
            elif (
                    text[i][-1] not in ("。", ".", "．") and
                (abs(len_(text[i]) - len_(text[i + 1])) < n or
                 (len_(t + text[i]) > len_(text[i + 1]) and bool(
                     re.search("[。\.．]\s\d|..[。\.．]|.[。\.．]", text[i + 1][-3:])
                     or bool(re.match("[A-Z]", text[i + 1][:1]))))
                 or bool(re.match("\s?[a-z,\)]", text[i + 1]))
                 or bra_cnt > ket_cnt or bra_cnt_jp > ket_cnt_jp)):
                t += text[i]
            else:
                sentences.append(t + text[i])
                t = ""
                
        else:
            sentences.append(t + text[i])
            
            
    return len(sentences), sentences


def translate(driver, i, texts):
    global tr
    preText = ""
    
    if tool == "DeepL":
        textarea = driver.find_element_by_css_selector(
            '.lmt__textarea.lmt__source_textarea.lmt__textarea_base_style')
    elif tool == "GT":
        textarea = driver.find_element_by_id('source')
        
    for j, t in enumerate(texts):
        etr = {"order": i * unit + j, "source": t}
        lock.acquire()
        ppc.copy(t)
        textarea.send_keys(Keys.COMMAND, "v")
        lock.release()
        transtext = ""
        cnt = 0
        
        while transtext in ("", preText):
            time.sleep(1)
            
            if tool == "DeepL":
                transtext = driver.find_element_by_css_selector(
                    '.lmt__textarea.lmt__target_textarea.lmt__textarea_base_style'
                ).get_property("value")
            elif tool == "GT":
                try:
                    transtext = driver.find_element_by_css_selector(
                        '.tlid-translation.translation').text
                except:
                    pass
                
            cnt += 1
            if cnt % 10 == 0: textarea.send_keys(".")
            
        etr["result"] = transtext
        tr.append(etr)
        textarea.send_keys(Keys.COMMAND, "a")
        textarea.send_keys(Keys.BACKSPACE)
        preText = transtext
        
    if len(tr) == length: html_gen(title, filename)


def runDriver(order):
    driver = webdriver.Chrome(ChromeDriverManager().install())
    url = 'https://www.deepl.com/ja/translator' if tool == "DeepL" else f'https://translate.google.co.jp/?hl=ja&tab=TT&authuser=0#view=home&op=translate&sl=auto&tl={"en" if inv else "ja"}'
    driver.get(url)
    translate(driver, order,
              source[order * unit:min(len(source), (order + 1) * unit)])
    driver.close()


def multiThreadTranslate(n):
    global tr, threads, source, length, unit, lock
    length, source = textParser(ppc.paste())
    unit = ceil(length / n)
    lock = Lock()
    clipboard = ppc.paste()
    
    for t in range(n):
        thread = Thread(target=runDriver, args=(t, ))
        thread.setDaemon(True)
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    ppc.copy(clipboard)


if __name__ == "__main__":
    tr = []
    threads = []
    n, inv, tool, filename, title = 10, False, "DeepL", "translated_text", "ORIGINAL　↔　TRANSLATED"
    
    n_ = input("いくつのChromeで並行翻訳しますか？    ")
    if n_.isdigit(): n = int(n_)
    
    if input("1. 英語 → 日本語    2. 日本語 → 英語   ") == "2": inv = True
    if input("1. DeepL 2.GoogleTranslate　　") == "2": tool = "GT"
    
    filename_ = input("出力ファイルにつける名前を入力してください（デフォルトは'translated_text.html'）　　")
    if filename_:
        filename = filename_.replace(" ", "_")
        
    title_ = input("（論文の）タイトルを入力してください   ")
    if title_:
        title = title_
        
    input("準備ができたらEnterを押してください")
    multiThreadTranslate(n)