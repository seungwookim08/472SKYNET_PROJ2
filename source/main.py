import numpy as np
import math, re, sys, matplotlib

def build_model():
    tokens = list()
    i = 1
    while True:
        print("/train/train-ham-{:05d}.txt".format(i))
        try:
            with open("/train/train-ham-{:05d}.txt".format(i),'r') as test_file:
                txt = test_file.read()
                tokenized = re.split('\[\^a-zA-Z\]',txt)
                print(tokenized)
        except FileNotFoundError:
            print('Build model for ham is done')
            break
    i += 1

def __main__():
    is_build = input('Do you want to build a model? (Y/N)')
    if is_build.lower() == 'y':
        build_model()
    
if __name__ == "__main__":
    __main__()
