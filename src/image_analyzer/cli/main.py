from __future__ import annotations

import argparse
from pathlib import Path

from image_analyzer.cli.batch import resolve_batch_inputs
from image_analyzer.config.settings import load_settings
from image_analyzer.detailed_pipeline import run_detailed_pipeline
from image_analyzer.pipeline import analyze_batch, analyze_image


def main() -> None:
    parser = argparse.ArgumentParser(prog="image-analyzer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    image_parser = subparsers.add_parser("analyze-image")
    image_parser.add_argument("image_path")
    image_parser.add_argument("--output-dir", default=None)
    image_parser.add_argument("--no-debug", action="store_true")

    detailed_parser = subparsers.add_parser("analyze-detailed")
    detailed_parser.add_argument("image_path")
    detailed_parser.add_argument("--output-dir", default=None)
    detailed_parser.add_argument("--project-name", default=None)
    detailed_parser.add_argument("--iterations", type=int, default=None)
    detailed_parser.add_argument("--aspect-ratio", default=None)
    detailed_parser.add_argument("--enable-generation", action="store_true")
    detailed_parser.add_argument("--enable-comparison", action="store_true")
    detailed_parser.add_argument("--target-score", type=float, default=None)
    detailed_parser.add_argument("--max-iterations", type=int, default=None)

    batch_parser = subparsers.add_parser("analyze-batch")
    batch_parser.add_argument("input_dir")
    batch_parser.add_argument("--output-dir", default=None)
    batch_parser.add_argument("--no-debug", action="store_true")

    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[3]
    config = load_settings(project_root)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    save_debug = not getattr(args, "no_debug", False)

    if args.command == "analyze-image":
        result = analyze_image(Path(args.image_path), config, output_root=output_dir, save_debug=save_debug)
        print(result.summary.long_description)
        return

    if args.command == "analyze-detailed":
        report = run_detailed_pipeline(
            Path(args.image_path),
            config,
            output_root=output_dir,
            project_name=args.project_name,
            iterations=args.iterations,
            aspect_ratio=args.aspect_ratio,
            enable_generation=args.enable_generation,
            enable_comparison=args.enable_comparison,
            target_score=args.target_score,
            max_iterations=args.max_iterations,
        )
        print(f"Detailed run directory: {report.run_dir}")
        print(report.prompt_package.final_prompt)
        return

    input_dir = Path(args.input_dir)
    image_paths = resolve_batch_inputs(input_dir)
    results = analyze_batch(image_paths, config, output_root=output_dir, save_debug=save_debug)
    print(f"Analyzed {len(results)} images.")


if __name__ == "__main__":
    main()
