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
    print(loader.OCR_data.head())
    loader.process_ocr_data()
    print(loader.OCR_data.head())

if __name__ == "__main__":
    main()