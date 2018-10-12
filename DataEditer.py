import re
import string

import inflect
from gensim import corpora
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words



class DataEdit():
    def stringRemoverAndStopword(l):
        r = []

        filter_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
                       't',
                       'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2'
            , '3', '4', '5', '6', '7', '8', '9']
        table = str.maketrans({key: None for key in string.punctuation})
        q = ""
        for s in l:
            s = re.sub(r'^https?:\/\/.*[\r\n]*', '', s, flags=re.MULTILINE)
            s = re.sub(r'^http?:\/\/.*[\r\n]*', '', s, flags=re.MULTILINE)
            s = s.lower()
            s = s.replace("\\n", " ")
            s = s.replace("b'", " ")
            s = s.replace("'", " ")
            s = s.replace('b"', ' ')
            s = s.replace('"', ' ')
            s = s.replace("page", " ")
            s = s.replace("resume", " ")
            for x in filter_list:
                for y in filter_list:
                    s = s.replace("\\x" + x + y, "")
            s = s.translate(table)
            r.append(s.lower())
        return r

    def stemAndStopText(r):
        stemmer = SnowballStemmer('english')
        words = stopwords.words('english')
        l = []
        for s in r:
            q = ""

            for w in s.split(" "):
                if w not in words:
                    q += w + " "
            l.append(q)
            for s in l:
                s = s.replace("  ", " ")
        print(l)
        return l


