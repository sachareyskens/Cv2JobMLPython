import sys, getopt

import Classifier_sklearn


def main(argv):


    try:
        opts, args = getopt.getopt(argv,"ht:p",["train=", "predict="])
    except getopt.GetoptError:
        print('main.py -t <trainDataLocation> -p <predictionString>')
        sys.exit(2)

    for opt, args in opts:
        if opt=='-h':
            print('main.py -i <trainDataLocation> -p <PredictionString>')
            sys.exit(0)
        elif opt=='-t':
            Classifier_sklearn.train()
            sys.exit()

