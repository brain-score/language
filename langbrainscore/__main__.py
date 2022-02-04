
import argparse

def main(args):
    print(args)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('lang-brainscore-fuzzy-potato')
    args = parser.parse_args()
    
    main(args)