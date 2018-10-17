import pandas as pd
import re
import string

import inflect
from gensim import corpora
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import digits

global inputFile
def getInputFile():
    return inputFile

def setInputFile(inputFilee):
    global inputFile
    inputFile = inputFilee

def readData():
    rd = pd.read_csv(inputFile)
    return rd

class DataEdit():
    def stringRemoverAndStopword(l):
        r = []

        filter_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
                       't',
                       'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2'
            , '3', '4', '5', '6', '7', '8', '9']
        table = str.maketrans({key: None for key in string.punctuation})

        remove_digits = str.maketrans('', '', digits)
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
            s = s.replace("  ", " ")
            s = s.translate(remove_digits)
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
            q = q.replace("  ", " ")
            l.append(q)
        return l


