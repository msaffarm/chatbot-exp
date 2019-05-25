import sys
import os
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
UTILS_DIR = os.path.join(CURRENT_DIR,"../../utils")
sys.path.append(UTILS_DIR)
from evaluator import Evaluator
import argparse


def parse_arguments():

    parser = argparse.ArgumentParser()

    # File arguments
    parser.add_argument('--target_path', dest='target_file', help='Path to target file')
    parser.add_argument('--response_path', dest='response_file', help='Path to response file')
    parser.add_argument('--diagid_path', dest='diagid_file', help='Path to diagid file')
    parser.add_argument('--entity_path', dest='entity_file', help='Path to diag entities')

    return parser.parse_args()

def main():
    args = parse_arguments()
    target_file = args.target_file
    response_file = args.response_file
    diagid_file = args.diagid_file
    entity_file = args.entity_file

    eval = Evaluator()
    eval.read_data_from_files(target_file,response_file,diagid_file,entity_file)
    
    report = eval.get_eval_stats(mode='hard')
    print(report)


if __name__ == '__main__':
    main()
