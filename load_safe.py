import torch, argparse
import torch.serialization
from classification.main import main, get_args_parser

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    torch.serialization.add_safe_globals([argparse.Namespace])

    main(args)
