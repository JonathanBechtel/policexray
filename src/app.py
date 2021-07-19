# -*- coding: utf-8 -*-
"""
Allows you to run analyzer.py from the command line
"""
import argparse
from analyzer import PolicingAnalyzer
import warnings
warnings.filterwarnings("ignore")



parser = argparse.ArgumentParser(description="Arguments to execute policing analysis")
parser.add_argument("-p", "--path",
                    type     = str,
                    action   = 'store',
                    help     = "the filepath of the data for analyzing",
                    required = True)

parser.add_argument("-d", "--dir", 
                    type     = str,
                    action   = 'store',
                    help     = "the directory you want to output your results to",
                    required = True)

parser.add_argument("-o", "--outcome",
                    type     = str,
                    action   = 'store',
                    help     = 'outcome you are studying.  one of "arrest" or "search"',
                    required = True)

parser.add_argument("-s", "--scenario",
                    type     = str,
                    action   = 'store',
                    help     = 'scenario for study.  one of "pulled-over", "being-search", "search-completed"')



args = parser.parse_args()



if __name__ == '__main__':
    analyzer = PolicingAnalyzer(filepath   = args.path,
                                output_dir = args.dir,
                                outcome    = args.outcome,
                                scenario   = args.scenario)
    analyzer.evaluate_data()