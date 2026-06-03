"""CLI for training from-scratch models.

Usage:
    python -m industry_ml_lab.training.from_scratch.cli --help
"""

from __future__ import annotations

import argparse
import sys

from .examples import train_tiny, train_small, train_medium, train_marcus


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Train from-scratch GPT-2 style models"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Tiny model
    parser_tiny = subparsers.add_parser("tiny", help="Train tiny model (~50K params)")
    parser_tiny.add_argument(
        "--dataset", "-d",
        type=str,
        default="shakespeare",
        choices=["shakespeare", "marcus"],
        help="Dataset to train on"
    )
    
    # Small model
    parser_small = subparsers.add_parser("small", help="Train small model (~200K params)")
    parser_small.add_argument(
        "--dataset", "-d",
        type=str,
        default="shakespeare",
        choices=["shakespeare", "marcus"],
        help="Dataset to train on"
    )
    
    # Medium model
    parser_medium = subparsers.add_parser("medium", help="Train medium model (~4M params)")
    parser_medium.add_argument(
        "--dataset", "-d",
        type=str,
        default="shakespeare",
        choices=["shakespeare", "marcus"],
        help="Dataset to train on"
    )
    
    # Marcus-specific
    parser_marcus = subparsers.add_parser("marcus", help="Train on Marcus Aurelius")
    parser_marcus.add_argument(
        "--size", "-s",
        type=str,
        default="tiny",
        choices=["tiny", "small", "medium"],
        help="Model size"
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    if args.command == "tiny":
        if args.dataset == "shakespeare":
            train_tiny.main()
        elif args.dataset == "marcus":
            train_marcus.main()
    
    elif args.command == "small":
        if args.dataset == "shakespeare":
            train_small.main()
        elif args.dataset == "marcus":
            # Use tiny for marcus since dataset is small
            train_tiny.main()
    
    elif args.command == "medium":
        if args.dataset == "shakespeare":
            train_medium.main()
        elif args.dataset == "marcus":
            # Use small for marcus since dataset is small
            train_small.main()
    
    elif args.command == "marcus":
        if args.size == "tiny":
            train_marcus.main()
        elif args.size == "small":
            train_small.main()
        elif args.size == "medium":
            train_medium.main()


if __name__ == "__main__":
    main()
