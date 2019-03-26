import numpy as np
import math, re, sys

def build_model():
    tokens = list()
    i = 1
    while True:
        print("/train/train-ham-{:05d}.txt".format(i))
        try:
            with open(".\\train\\train-ham-{:05d}.txt".format(i), encoding='utf-8') as test_file:
                txt = test_file.read()
                tokenized = re.split('\[\^a-zA-Z\]',txt)
                print(tokenized)
        except FileNotFoundError:
            print('Build model for ham is done')
            break
        except UnicodeDecodeError:
            # utf-8 code can't decode, then try with byte code
            with open(".\\train\\train-ham-{:05d}.txt".format(i), 'rb') as test_file:
                txt = test_file.read()
                print(txt)
        i += 1

def __main__():    
    is_build = input('Do you want to build a model? (Y/N)')
    if is_build.lower() == 'y':
        build_model()
    
if __name__ == "__main__":
    __main__()
