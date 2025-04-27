import argparse
from utils.logger import setup_logger
from dataset.loader import DatasetLoader

def main():
    parser = argparse.ArgumentParser(description="Baxus Whisky Goggles")
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()

    logger = setup_logger(args.verbose)
    logger.info("Starting application")

    loader = DatasetLoader()
    loader.process_ocr_data()

    test_string = "h—_~\n\nJEFFERSONS\n\nKENTUCKY STRAIGHT\nBOBRBON WHISKEY\nVERYSSMALL BATCH\n\n41:15% ALGI/VOL (823 PROOF)"
    matches = loader.get_best_matches(test_string)
    print(matches)


if __name__ == "__main__":
    main()