# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Print the reductions, channels and model YAML rows for one timm backbone.

`TimmBackbone` keeps the timm stages whose strides are requested (default P3/8, P4/16 and P5/32) and projects each one
to the channel count declared in the YAML. This tool reports what the model provides and prints the `TimmBackbone` +
`Index` rows that can be pasted straight into a `backbone:` section.

Usage:
    python tools/timm_check.py resnet18
    python tools/timm_check.py resnet18 --strides 4 8 16 32   # P2/4 model
    python tools/timm_check.py resnet50 --strides 8 16 32 64  # P6/64 model
"""

from __future__ import annotations

import argparse
import sys

from ultralytics.utils.checks import check_requirements

LEVELS = {4: "P2/4", 8: "P3/8", 16: "P4/16", 32: "P5/32", 64: "P6/64"}


def parse_args(args):
    parser = argparse.ArgumentParser(description="Print timm reductions/channels and the model YAML rows.")
    parser.add_argument("model", help="timm model name, e.g. resnet18")
    parser.add_argument(
        "--strides", nargs="+", type=int, default=[8, 16, 32], help="strides to keep, e.g. 4 8 16 32 for P2"
    )
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)

    check_requirements("timm")
    import timm

    try:
        trunk = timm.create_model(args.model, features_only=True)
    except Exception as error:
        sys.exit(f"{args.model}: {error}")

    reductions, channels = trunk.feature_info.reduction(), trunk.feature_info.channels()
    print(f"{args.model}\n  reductions: {reductions}\n  channels  : {channels}")

    missing = [stride for stride in args.strides if stride not in reductions]
    if missing:
        sys.exit(f"missing strides {missing}, available reductions {reductions}")

    out_channels = [channels[reductions.index(stride)] for stride in args.strides]
    width = ", ".join(str(channel) for channel in out_channels)

    print("\nbackbone:")
    print(f'  - [-1, 1, TimmBackbone, [{out_channels[0]}, "{args.model}", [{width}], True]]')
    for index, stride in enumerate(args.strides):
        level = LEVELS.get(stride, stride)
        print(f"  - [{-1 - index}, 1, Index, [{out_channels[index]}, {index}]] # {index + 1}-{level}")


if __name__ == "__main__":
    main(sys.argv[1:])
