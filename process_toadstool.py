import argparse
from pathlib import Path

from mario.supervised.expert import (
    process_single_session,
    process_multiple_sessions,
)

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(
        description="A script used to replay the game session and store the frames."
    )

    argument_parser.add_argument("-i", "--input-path", type=str)
    argument_parser.add_argument("-o", "--output-path", type=str, default=None)
    argument_parser.add_argument("-l", "--length", type=int, default=None)
    argument_parser.add_argument("-r", "--render", action="store_true")

    args = argument_parser.parse_args()

    data_path = args.input_path
    output_path = args.output_path

    path = Path(data_path)
    if path.suffix == ".json":
        process_single_session(
            data_path, output_path, args.render, args.length
        )
    elif path.is_dir():
        process_multiple_sessions(
            data_path, output_path, args.render, args.length
        )
    else:
        raise ValueError("Invalid data path specified")
