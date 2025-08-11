"""Command-line interface for the network anomaly detection project."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Network Anomaly Detection CLI")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train the autoencoder model")
    train_parser.add_argument("--data", default="data/processed/cicids2017.csv")

    detect_parser = subparsers.add_parser("detect", help="Run anomaly detection")
    detect_parser.add_argument("--data", default="data/processed/cicids2017_test.csv")

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate detection results")
    eval_parser.add_argument("--results", default="evaluation/detections.csv")

    args = parser.parse_args()

    if args.command == "train":
        import train_model

        train_model.train(args.data)
    elif args.command == "detect":
        import detect

        detect.detect(args.data)
    elif args.command == "evaluate":
        import evaluate

        evaluate.evaluate(args.results)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
