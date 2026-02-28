import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("excel_path")
    args = parser.parse_args()
    df = pd.read_excel(args.excel_path, engine="openpyxl")
    print([str(c) for c in df.columns])
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
