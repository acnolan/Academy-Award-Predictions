import argparse
import pandas as pd

from BuildData import rebuildTable
from MachineLearning import executeMachineLearning
from VisualizeData import setVisualizeFlag

# Set up command line argument parsing and -h/--help flags
parser = argparse.ArgumentParser(
    prog = 'AcademyAwardPrediction',
    description = 'Runs some machine learning on historical Academy Award data as well as Letterboxd and Twitter info on the films.')

# Set up flags for rebuilding the csv with letterboxd and twitter data
parser.add_argument('-b', '--build', action='store_true', help='Runs the web scraping code to build the data tables and writes to csv. Note: This can take several hours...')
parser.add_argument('-v', '--visualize', action='store_true', help='Runs code to visualize the data and displays plots.')

# Set Pandas options
pd.set_option('display.max_columns', None)


# Optional -b flag for building
if __name__ == "__main__":
    # If the build flag is present, rebuild the data
    # Building is slow so this is optional
    args = parser.parse_args()
    if args.build:
        rebuildTable()
    if args.visualize:
        setVisualizeFlag()

    # Run the fancy machine learning?
    executeMachineLearning()