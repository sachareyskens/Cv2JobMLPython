import re
import string
from string import digits

import pandas as pd
from nltk.corpus import stopwords

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
    def stringRemoverAndStopword(string_list):
        cleaned_string_list = []

        filter_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
                       't',
                       'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2'
            , '3', '4', '5', '6', '7', '8', '9']
        table = str.maketrans({key: None for key in string.punctuation})

        remove_digits = str.maketrans('', '', digits)
        for line in string_list:
            line = re.sub(r'^https?:\/\/.*[\r\n]*', '', line, flags=re.MULTILINE)
            line = re.sub(r'^http?:\/\/.*[\r\n]*', '', line, flags=re.MULTILINE)
            line = line.lower()
            line = line.replace("\\n", " ")
            line = line.replace("b'", " ")
            line = line.replace("'", " ")
            line = line.replace('b"', ' ')
            line = line.replace('"', ' ')
            line = line.replace('\\uf0b7', '')
            line = line.replace("page", " ")
            line = line.replace("resume", " ")
            line = line.replace("  ", " ")
            line = line.translate(remove_digits)
            for x in filter_list:
                for y in filter_list:
                    line = line.replace("\\x" + x + y, "")
            line = line.translate(table)
            cleaned_string_list.append(line.lower())
        return cleaned_string_list

    def stemAndStopRemover(string_list):
        words = stopwords.words('english')
        l = []
        for line in string_list:
            cleaned = ""
            for word in line.split(" "):
                if word not in words:
                    cleaned += word + " "
            cleaned = cleaned.replace("  ", " ")
            l.append(cleaned)
        return l


