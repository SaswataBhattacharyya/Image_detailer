import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image

from image_analyzer.cli.main import main as cli_main
from image_analyzer.config.settings import load_settings
from image_analyzer.detailed_pipeline import run_image_flow
from image_analyzer.models.schemas import GenerationResult
from image_analyzer.providers.base import ProviderArtifact


def _visual_artifact_for_prompt(prompt: str) -> ProviderArtifact:
    lowered = prompt.lower()
    if "choose the single most important unresolved visual detail" in lowered:
        return ProviderArtifact(provider="ollama", data={"content": "Focus on the horizon placement and the foreground object size."})
    if "focus only on" in lowered:
        return ProviderArtifact(
            provider="ollama",
            data={
                "content": (
                    "The horizon sits slightly below center at roughly 48 percent from the top. "
                    "A small dark rounded rock occupies the lower-right foreground at about x 78 percent and y 80 percent."
                )
            },
        )
    return ProviderArtifact(
        provider="ollama",
        data={
            "content": (
                "Low-angle beach landscape with a reflective wet-sand foreground, a small dark rounded rock in the lower-right area, "
                "a calm ocean band, and a colorful sky with the brightest glow near the horizon. "
                "Follow-up is needed for exact horizon placement and the foreground object size."
            )
        },
    )


class DetailedPipelineTests(unittest.TestCase):
    def test_run_image_flow_writes_unified_run_artifacts(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        config = load_settings(project_root)
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            image_path = tmp_path / "beach.png"
            Image.new("RGB", (1536, 1024), (120, 90, 160)).save(image_path)

            with patch(
                "image_analyzer.providers.ollama.OllamaSynthesisProvider.analyze_visual_pass",
                side_effect=lambda image_path, prompt, num_predict=900: _visual_artifact_for_prompt(prompt),
            ), patch(
                "image_analyzer.providers.ollama.OllamaSynthesisProvider.generate_json",
                return_value=ProviderArtifact(provider="ollama", data={"json": {}}),
            ), patch(
                "image_analyzer.providers.ollama.OllamaSynthesisProvider.compare_images",
                return_value=ProviderArtifact(
                    provider="ollama",
                    data={"json": {"overall_similarity_score": 0.92, "semantic_similarity_score": 0.92, "issues": [], "negative_prompt_additions": []}},
                ),
            ), patch(
                "image_analyzer.detailed_pipeline._execute_generation_command",
                side_effect=_fake_generation_command,
            ):
                report = run_image_flow(image_path, config, output_root=tmp_path / "runs")

            run_dir = Path(report.run_dir)
            self.assertTrue((run_dir / "memory" / "final_scene_memory.json").exists())
            self.assertTrue((run_dir / "outputs" / "structured_scene_map.json").exists())
            self.assertTrue((run_dir / "outputs" / "concise_generation_prompt.txt").exists())
            self.assertTrue((run_dir / "comparisons" / "similarity_v1.json").exists())
            self.assertEqual(report.generation.status, "completed")
            self.assertGreaterEqual(report.termination.best_score, 0.0)
            self.assertTrue(report.question_history)
            self.assertIn("small dark rounded rock", report.prompt_package.final_prompt.lower())

    def test_cli_analyze_image_prints_run_directory_and_similarity(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            image_path = tmp_path / "beach.png"
            Image.new("RGB", (800, 600), (100, 110, 180)).save(image_path)

            stdout = io.StringIO()
            with patch(
                "image_analyzer.providers.ollama.OllamaSynthesisProvider.analyze_visual_pass",
                side_effect=lambda image_path, prompt, num_predict=900: _visual_artifact_for_prompt(prompt),
            ), patch(
                "image_analyzer.providers.ollama.OllamaSynthesisProvider.generate_json",
                return_value=ProviderArtifact(provider="ollama", data={"json": {}}),
            ), patch(
                "image_analyzer.providers.ollama.OllamaSynthesisProvider.compare_images",
                return_value=ProviderArtifact(
                    provider="ollama",
                    data={"json": {"overall_similarity_score": 0.92, "semantic_similarity_score": 0.92, "issues": [], "negative_prompt_additions": []}},
                ),
            ), patch(
                "image_analyzer.detailed_pipeline._execute_generation_command",
                side_effect=_fake_generation_command,
            ), patch(
                "sys.argv",
                ["image-analyzer", "analyze-image", str(image_path), "--output-dir", str(tmp_path / "runs")],
            ), patch("sys.stdout", stdout):
                cli_main()

            output = stdout.getvalue()
            self.assertIn("Run directory:", output)
            self.assertIn("Similarity:", output)

    def test_generation_without_backend_reports_runtime_failure(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        config = load_settings(project_root)
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            image_path = tmp_path / "beach.png"
            Image.new("RGB", (512, 512), (120, 90, 160)).save(image_path)

            with patch(
                "image_analyzer.providers.ollama.OllamaSynthesisProvider.analyze_visual_pass",
                side_effect=lambda image_path, prompt, num_predict=900: _visual_artifact_for_prompt(prompt),
            ), patch(
                "image_analyzer.providers.ollama.OllamaSynthesisProvider.generate_json",
                return_value=ProviderArtifact(provider="ollama", data={"json": {}}),
            ), patch.dict("os.environ", {"IMAGE_ANALYZER_QWEN_IMAGE_RUNNER": ""}, clear=False):
                report = run_image_flow(
                    image_path,
                    config,
                    output_root=tmp_path / "runs",
                    max_full_restarts=1,
                )

            self.assertEqual(report.generation.status, "failed_runtime")


def _fake_generation_command(*, command_template, prompt_file, negative_prompt_file, output_image, generation_config, iteration):
    del command_template, prompt_file, negative_prompt_file, generation_config, iteration
    Image.new("RGB", (512, 512), (120, 90, 160)).save(output_image)
    return GenerationResult(
        enabled=True,
        status="completed",
        message="mock generation completed",
        output_image=str(output_image),
    )


if __name__ == "__main__":
    unittest.main()
