import sys, getopt

def main(argv):
    trainData = ''
    trainMethod = ''

    try:
        opts, args = getopt.getopt(argv,"hi:o",["idata=", "odata="])
    except getopt.GetoptError:
        print('main.py -i <trainDataLocation> -o <trainMethod>')
        sys.exit(2)

    for opt, args in opts:
        if opt=='-h':
            print('main.py -i <trainDataLocation> -o <trainMethod>')
            sys.exit(0)


