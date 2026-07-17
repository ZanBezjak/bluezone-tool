import argparse
from dotenv import load_dotenv
load_dotenv()

from src.data.eurostat import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(force=args.force)
