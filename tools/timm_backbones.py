# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Search the installed timm registry for candidate backbones.

Prints plain timm model names, which are exactly what `TimmBackbone` accepts in a model YAML. Patterns are fuzzy by
default (`resnet` becomes `*resnet*`) and keep their fnmatch wildcards when written explicitly (`resnet*`, `*net?8*`).
Several patterns are unioned, so `-k conv -k swin` lists both families.

Usage:
    python tools/timm_backbones.py                              # full registry
    python tools/timm_backbones.py resnet                       # fuzzy: names containing "resnet"
    python tools/timm_backbones.py -k conv -k swin --limit 30   # several patterns (union), first 30
    python tools/timm_backbones.py -k resnet --pretrained       # only names that ship pretrained weights
"""

from __future__ import annotations

import argparse
import sys

from ultralytics.utils.checks import check_requirements


def parse_args(args):
    """Build the command line parser."""
    parser = argparse.ArgumentParser(description="Search the timm model registry for candidate backbones.")
    parser.add_argument("pattern", nargs="*", help="fuzzy name patterns, fnmatch wildcards are kept as-is")
    parser.add_argument("-k", "--keyword", action="append", default=[], help="extra fuzzy keyword, repeatable")
    parser.add_argument("-x", "--exclude", action="append", default=[], help="name pattern to drop, repeatable")
    parser.add_argument("--pretrained", action="store_true", help="only models that ship pretrained weights")
    parser.add_argument("--tags", action="store_true", help="include pretrained tags, e.g. resnet18.a1_in1k")
    parser.add_argument("--limit", type=int, default=0, help="print at most N names, 0 means all")
    return parser.parse_args(args)


def get_pattern(text: str) -> str:
    """Wrap a bare keyword in wildcards, leaving explicit fnmatch patterns untouched."""
    return text if any(char in text for char in "*?[") else f"*{text}*"


def search(args: argparse.Namespace) -> list[str]:
    """Return the sorted timm model names matching the requested filters."""
    check_requirements("timm")
    import timm

    return timm.list_models(
        filter=[get_pattern(pattern) for pattern in [*args.pattern, *args.keyword]],
        exclude_filters=[get_pattern(pattern) for pattern in args.exclude],
        pretrained=args.pretrained,
        include_tags=args.tags,
    )


def main(args):
    args = parse_args(args)
    models = search(args)

    if not models:
        print("no model matched", file=sys.stderr)
        return

    print("\n".join(models[: args.limit] if args.limit > 0 else models))


if __name__ == "__main__":
    main(sys.argv[1:])
