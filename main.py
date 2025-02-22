from dataset import TreeCrownGraphDataset
import argparse

def main(args):
    print('0')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a ContourMerge model')
    parser.add_argument('--dataPath', type=str, help='Directory containing the adjacent matrix files',default = './data/')
    '''
    Path to constuct the dataset
    '''