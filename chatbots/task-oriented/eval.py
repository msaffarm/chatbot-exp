import sys
import os
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
UTILS_DIR = os.path.join(CURRENT_DIR,"../../utils")
sys.path.append(UTILS_DIR)
from evaluator import Evaluator


def main():
    response_file = "Movie-response.txt"
    target_file = "Movie-target-test.txt"

    eval = Evaluator()
    eval.read_data_from_files(os.path.join(CURRENT_DIR,target_file),os.path.join(CURRENT_DIR,response_file))
    
    report = eval.get_eval_stats()
    print(report)


if __name__ == '__main__':
    main()
