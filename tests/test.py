from project2 import main
import argparse
def test_fetchtextdata():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, action="append", help="Input all text files")
    args = parser.parse_args("--input ../cs5293sp22-project1/docs/*.txt".split())
    inputfiles = redactormain.fetchtextdata(args)
    assert isinstance(inputfiles,list)