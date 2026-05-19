from __future__ import annotations

import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from shutil import copy2
from typing import Any

from image_analyzer.config.settings import AppConfig
from image_analyzer.models.schemas import (
    CanvasSpec,
    ColorPaletteSpec,
    ComparisonIssue,
    ComparisonReport,
    GenerationConfig,
    GenerationResult,
    HybridSimilarityScore,
    IterationResult,
    LoopTermination,
    PromptCorrection,
    PromptPackage,
    RegionSpec,
    RunConfig,
    RunReport,
    SceneMap,
    SceneObject,
    ScenePassResult,
    SkySpec,
    VisualHierarchy,
)
from image_analyzer.providers.color import DominantColorProvider
from image_analyzer.providers.ollama import OllamaSynthesisProvider, configure_ollama_runtime
from image_analyzer.providers.optional_cv import TesseractOcrProvider
from image_analyzer.similarity import compute_hybrid_similarity
from image_analyzer.utils.files import dump_json, dump_text, sha256_file
from image_analyzer.utils.image_ops import load_image_size


PASS_TITLES = {
    "composition_camera": "Composition And Camera",
    "objects_positions": "Objects And Positions",
    "foreground_texture": "Foreground Texture",
    "middle_ground_horizon": "Middle Ground And Horizon",
    "sky_cloud_structure": "Sky And Cloud Structure",
    "color_lighting": "Color And Lighting",
    "negative_constraints": "Negative Constraints",
}


def run_detailed_pipeline(
    image_path: Path,
    config: AppConfig,
    *,
    output_root: Path | None = None,
    project_name: str | None = None,
    iterations: int | None = None,
    aspect_ratio: str | None = None,
    enable_generation: bool | None = None,
    enable_comparison: bool | None = None,
    target_score: float | None = None,
    max_iterations: int | None = None,
) -> RunReport:
    image_path = image_path.resolve()
    timestamp = datetime.now(timezone.utc)
    project = project_name or config.detailed.default_project_name
    run_id = f"{timestamp.strftime('%Y-%m-%d_%H%M%S')}_{_slugify(project)}_{_slugify(image_path.stem)}"
    run_root = (output_root or config.detailed.run_root_dir).resolve()
    run_dir = run_root / run_id
    paths = _build_run_paths(run_dir)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    copied_reference = paths["input"] / f"reference{image_path.suffix.lower()}"
    copy2(image_path, copied_reference)
    width, height = load_image_size(image_path)
    detected_aspect_ratio = aspect_ratio or _aspect_ratio_string(width, height) or config.detailed.default_aspect_ratio
    generation_enabled = enable_generation if enable_generation is not None else config.detailed.enable_generation_by_default
    comparison_enabled = enable_comparison if enable_comparison is not None else config.detailed.enable_comparison_by_default
    target_similarity = target_score if target_score is not None else config.detailed.target_score
    max_rounds = max_iterations if max_iterations is not None else config.detailed.max_iterations
    if iterations is not None:
        max_rounds = iterations

    run_config = RunConfig(
        project_name=project,
        reference_image=str(copied_reference),
        vlm_model=config.models.analysis_model,
        llm_model=config.models.structuring_model,
        generator_backend=config.generation.backend if generation_enabled else None,
        comparison_model=config.models.comparison_model if comparison_enabled else None,
        iterations=max_rounds,
        aspect_ratio=detected_aspect_ratio,
        enable_generation=generation_enabled,
        enable_comparison=comparison_enabled,
        target_score=target_similarity,
        max_iterations=max_rounds,
        scene_weight=config.detailed.scene_weight,
        perceptual_weight=config.detailed.perceptual_weight,
        created_at=timestamp,
    )
    dump_json(paths["logs"] / "run_config.json", run_config.model_dump(mode="json"))

    configure_ollama_runtime(
        max_loaded_models=config.models.ollama_max_loaded_models,
        num_parallel=config.models.ollama_num_parallel,
    )
    support_signals = _collect_support_signals(image_path)
    warnings: list[str] = []
    model_calls: list[dict[str, Any]] = []

    vlm_client = OllamaSynthesisProvider(
        base_url=config.models.ollama_base_url,
        model_name=config.models.analysis_model,
        timeout_sec=config.models.ollama_timeout_sec,
    )
    structuring_client = OllamaSynthesisProvider(
        base_url=config.models.ollama_base_url,
        model_name=config.models.structuring_model,
        timeout_sec=config.models.ollama_timeout_sec,
    )

    passes = _run_visual_passes(
        image_path=image_path,
        width=width,
        height=height,
        aspect_ratio=detected_aspect_ratio,
        support_signals=support_signals,
        paths=paths,
        client=vlm_client,
        model_calls=model_calls,
        warnings=warnings,
        pass_keys=config.detailed.visual_passes,
    )
    merged_scene_map = _merge_scene_map(passes, support_signals, width, height, detected_aspect_ratio)
    dump_json(paths["analysis"] / "02_structured_scene_map.json", merged_scene_map.model_dump(mode="json"))

    refine_prompt = _build_scene_refiner_prompt(merged_scene_map, passes)
    refined_artifact = structuring_client.generate_json(refine_prompt, num_predict=1400)
    warnings.extend(refined_artifact.warnings)
    model_calls.append({"phase": "scene_refiner", "model": config.models.structuring_model, "warnings": refined_artifact.warnings})
    refined_scene_map = _refine_scene_map(merged_scene_map, refined_artifact.data.get("json"))
    dump_text(paths["analysis"] / "03_scene_refiner_raw.txt", str(refined_artifact.data.get("content", "")) + "\n")
    dump_json(paths["analysis"] / "03_refined_scene_map.json", refined_scene_map.model_dump(mode="json"))

    hierarchy = _build_visual_hierarchy(refined_scene_map)
    dump_json(paths["analysis"] / "04_visual_hierarchy.json", hierarchy.model_dump(mode="json"))

    base_prompt_package = _build_prompt_package(refined_scene_map, hierarchy)
    _write_prompt_package(paths["prompts"], base_prompt_package, version=1, also_write_base=True)

    generation_config = GenerationConfig(
        backend=config.generation.backend,
        width=config.generation.width,
        height=config.generation.height,
        steps=config.generation.steps,
        cfg=config.generation.cfg,
        sampler=config.generation.sampler,
        seed=config.generation.seed,
        model=config.generation.model,
        command_template=config.generation.command_template,
    )

    iteration_results: list[IterationResult] = []
    best_iteration: IterationResult | None = None
    best_score = -1.0
    active_prompt_package = base_prompt_package

    if generation_enabled:
        comparison_enabled = True if enable_comparison is None else comparison_enabled

    for iteration_number in range(1, max_rounds + 1):
        _write_prompt_package(paths["prompts"], active_prompt_package, version=iteration_number)
        iteration_generation = _generate_image_iteration(
            iteration=iteration_number,
            paths=paths,
            prompt_package=active_prompt_package,
            generation_config=generation_config,
            enabled=generation_enabled,
        )
        comparison_report: ComparisonReport | None = None
        correction: PromptCorrection | None = None
        hybrid_score: HybridSimilarityScore | None = None

        if iteration_generation.output_image and comparison_enabled:
            comparison_artifact = vlm_client.compare_images(
                copied_reference,
                Path(iteration_generation.output_image),
                _build_comparison_prompt(refined_scene_map, hierarchy),
            )
            warnings.extend(comparison_artifact.warnings)
            model_calls.append(
                {"phase": "image_comparison", "iteration": iteration_number, "model": config.models.comparison_model, "warnings": comparison_artifact.warnings}
            )
            comparison_report = _build_comparison_report(comparison_artifact.data.get("json"), hierarchy)
            hybrid_score = compute_hybrid_similarity(
                copied_reference,
                Path(iteration_generation.output_image),
                semantic_similarity_score=comparison_report.semantic_similarity_score or comparison_report.overall_similarity_score,
                scene_weight=run_config.scene_weight,
                perceptual_weight=run_config.perceptual_weight,
                issue_severities=[issue.severity for issue in comparison_report.issues],
            )
            comparison_report.overall_similarity_score = hybrid_score.weighted_score
            comparison_report.perceptual_similarity_score = hybrid_score.perceptual.perceptual_similarity_score
            comparison_report.semantic_similarity_score = hybrid_score.semantic.semantic_similarity_score
            dump_text(paths["comparisons"] / f"comparison_v{iteration_number}.txt", str(comparison_artifact.data.get("content", "")) + "\n")
            dump_json(paths["comparisons"] / f"comparison_v{iteration_number}.json", comparison_report.model_dump(mode="json"))
            dump_json(paths["comparisons"] / f"hybrid_score_v{iteration_number}.json", hybrid_score.model_dump(mode="json"))
            correction = _build_prompt_correction(comparison_report, active_prompt_package)
            dump_json(paths["comparisons"] / f"correction_v{iteration_number}.json", correction.model_dump(mode="json"))

        iteration_result = IterationResult(
            iteration=iteration_number,
            prompt_package=active_prompt_package,
            generation=iteration_generation,
            comparison=comparison_report,
            hybrid_score=hybrid_score,
            correction=correction,
            accepted=bool(hybrid_score and hybrid_score.weighted_score >= target_similarity),
        )
        iteration_results.append(iteration_result)

        score = hybrid_score.weighted_score if hybrid_score is not None else 0.0
        if score > best_score:
            best_score = score
            best_iteration = iteration_result

        if iteration_result.accepted:
            break

        if not generation_enabled or not comparison_enabled or comparison_report is None or correction is None:
            break

        active_prompt_package = _apply_prompt_correction(active_prompt_package, correction, iteration_number + 1)

    final_generation = best_iteration.generation if best_iteration is not None else GenerationResult(
        enabled=generation_enabled,
        status="skipped",
        message="No generation iterations were executed.",
    )
    final_comparison = best_iteration.comparison if best_iteration is not None else None
    final_correction = best_iteration.correction if best_iteration is not None else None
    final_prompt_package = best_iteration.prompt_package if best_iteration is not None else base_prompt_package
    termination = _build_termination(iteration_results, target_similarity, best_score)

    report = RunReport(
        run_id=run_id,
        run_dir=str(run_dir),
        reference_image=str(copied_reference),
        scene_map=refined_scene_map,
        visual_hierarchy=hierarchy,
        prompt_package=final_prompt_package,
        passes=passes,
        generation=final_generation,
        comparison=final_comparison,
        correction=final_correction,
        iterations=iteration_results,
        termination=termination,
        warnings=sorted(set(warnings)),
    )
    dump_json(paths["reports"] / "final_prompt_package.json", _build_final_prompt_package(report, generation_config))
    dump_text(paths["prompts"] / "final_prompt.txt", final_prompt_package.final_prompt + "\n")
    dump_text(paths["reports"] / "final_report.md", _build_final_report(report))
    dump_text(paths["logs"] / "model_calls.jsonl", "\n".join(json.dumps(item) for item in model_calls) + ("\n" if model_calls else ""))
    dump_json(paths["reports"] / "run_report.json", report.model_dump(mode="json"))
    return report


def _run_visual_passes(
    *,
    image_path: Path,
    width: int,
    height: int,
    aspect_ratio: str,
    support_signals: dict[str, Any],
    paths: dict[str, Path],
    client: OllamaSynthesisProvider,
    model_calls: list[dict[str, Any]],
    warnings: list[str],
    pass_keys: tuple[str, ...],
) -> list[ScenePassResult]:
    passes: list[ScenePassResult] = []
    for index, pass_key in enumerate(pass_keys, start=1):
        prompt = _build_visual_pass_prompt(pass_key, image_path.name, width, height, aspect_ratio, support_signals)
        artifact = client.analyze_visual_pass(image_path, prompt)
        raw_response = str(artifact.data.get("content", "")).strip()
        warnings.extend(artifact.warnings)
        structured = _coerce_pass_structure(pass_key, raw_response, support_signals, width, height)
        result = ScenePassResult(
            pass_key=pass_key,
            title=PASS_TITLES.get(pass_key, pass_key.replace("_", " ").title()),
            prompt=prompt,
            raw_response=raw_response,
            structured=structured,
            warnings=list(artifact.warnings),
        )
        passes.append(result)
        dump_text(paths["analysis"] / f"pass_{index:02d}_{pass_key}.txt", raw_response + ("\n" if raw_response else ""))
        dump_json(paths["analysis"] / f"pass_{index:02d}_{pass_key}.json", result.model_dump(mode="json"))
        model_calls.append({"phase": "visual_pass", "pass_key": pass_key, "model": client.model_name, "warnings": artifact.warnings})
    return passes


def _build_run_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "run_dir": run_dir,
        "input": run_dir / "input",
        "analysis": run_dir / "analysis",
        "prompts": run_dir / "prompts",
        "generated": run_dir / "generated",
        "comparisons": run_dir / "comparisons",
        "reports": run_dir / "reports",
        "logs": run_dir / "logs",
    }


def _collect_support_signals(image_path: Path) -> dict[str, Any]:
    width, height = load_image_size(image_path)
    colors = DominantColorProvider().analyze(image_path, {}).data
    ocr = TesseractOcrProvider().analyze(image_path, {}).data
    return {
        "file_name": image_path.name,
        "sha256": sha256_file(image_path),
        "width": width,
        "height": height,
        "dominant_colors": colors.get("dominant_colors", []),
        "ocr_text": str(ocr.get("ocr_text", "")).strip(),
    }


def _build_visual_pass_prompt(
    pass_key: str,
    image_name: str,
    width: int,
    height: int,
    aspect_ratio: str,
    support_signals: dict[str, Any],
) -> str:
    tasks = {
        "composition_camera": "Describe composition, framing, horizon placement, camera angle, and lens feel.",
        "objects_positions": "List main objects with approximate size and center position percentages. Do not invent small objects.",
        "foreground_texture": "Describe the lower part of the image: materials, reflections, texture, channels, ripples, and dominant color notes.",
        "middle_ground_horizon": "Describe middle-ground structure, horizon, ocean or land band, and depth cues.",
        "sky_cloud_structure": "Describe upper image cloud structure, motion feel, light source behavior, and color separation.",
        "color_lighting": "Describe palette, saturation, contrast, lighting temperature, highlights, and reflected colors.",
        "negative_constraints": "List important things a generator must avoid changing or adding.",
    }
    return (
        f"You are analyzing {image_name} ({width}x{height}, aspect ratio {aspect_ratio}). "
        "Return a concise but detailed response grounded in the image. Estimate percentages where possible. "
        f"Task: {tasks.get(pass_key, pass_key)} Support signals: {support_signals}"
    )


def _coerce_pass_structure(pass_key: str, raw_response: str, support_signals: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    dominant_names = [str(item.get("name", "")) for item in support_signals.get("dominant_colors", []) if isinstance(item, dict)]
    default_center = {"x_percent": 50.0, "y_percent": 48.0}
    if pass_key == "composition_camera":
        return {
            "aspect_ratio": _aspect_ratio_string(width, height),
            "orientation": "landscape" if width >= height else "portrait",
            "camera_angle": _camera_angle_heuristic(raw_response),
            "lens_feel": "wide-angle landscape lens",
            "horizon_y_percent_from_top": 48.0,
            "vanishing_or_radiation_center": default_center,
        }
    if pass_key == "objects_positions":
        return {"main_objects": _objects_from_text(raw_response)}
    if pass_key == "foreground_texture":
        return {
            "region_y_percent": "60-100",
            "summary": raw_response or "Foreground surface details were not resolved.",
            "texture": _top_phrases(raw_response, fallback=["surface texture unresolved"]),
            "dominant_colors": dominant_names,
        }
    if pass_key == "middle_ground_horizon":
        return {
            "region_y_percent": "45-60",
            "summary": raw_response or "Middle-ground structure was not resolved.",
            "texture": _top_phrases(raw_response, fallback=["soft horizon band"]),
            "dominant_colors": dominant_names,
        }
    if pass_key == "sky_cloud_structure":
        return {
            "region_y_percent": "0-48",
            "cloud_direction": "radiating from the horizon center" if "radiat" in raw_response.lower() else "broad layered cloud motion",
            "upper_left": "darker cool-toned cloud mass",
            "upper_right": "warmer cloud mass",
            "center": "brightest glow near the horizon",
            "motion": "soft atmospheric motion",
            "sharpness": "natural, not hyper-sharp",
        }
    if pass_key == "color_lighting":
        return {
            "dominant": dominant_names or ["blue", "violet", "pink"],
            "saturation": "vivid but natural",
            "contrast": "moderate to high",
            "summary": raw_response,
        }
    if pass_key == "negative_constraints":
        return {"negative_constraints": _negative_constraints_from_text(raw_response)}
    return {"summary": raw_response}


def _merge_scene_map(
    passes: list[ScenePassResult],
    support_signals: dict[str, Any],
    width: int,
    height: int,
    aspect_ratio: str,
) -> SceneMap:
    pass_map = {item.pass_key: item.structured for item in passes}
    composition = pass_map.get("composition_camera", {})
    foreground = pass_map.get("foreground_texture", {})
    middle = pass_map.get("middle_ground_horizon", {})
    sky = pass_map.get("sky_cloud_structure", {})
    color_lighting = pass_map.get("color_lighting", {})
    negatives = pass_map.get("negative_constraints", {})
    objects = pass_map.get("objects_positions", {}).get("main_objects", [])
    return SceneMap(
        canvas=CanvasSpec(
            aspect_ratio=aspect_ratio,
            orientation=str(composition.get("orientation", "landscape" if width >= height else "portrait")),
            camera_angle=str(composition.get("camera_angle", "low landscape angle")),
            lens_feel=str(composition.get("lens_feel", "wide-angle landscape lens")),
            horizon_y_percent_from_top=float(composition.get("horizon_y_percent_from_top", 48.0)),
            vanishing_or_radiation_center=dict(composition.get("vanishing_or_radiation_center", {"x_percent": 50.0, "y_percent": 48.0})),
        ),
        main_objects=[SceneObject(**item) for item in objects if isinstance(item, dict)],
        foreground=RegionSpec(
            region_y_percent=str(foreground.get("region_y_percent", "60-100")),
            summary=str(foreground.get("summary", "Foreground surface details were not resolved.")),
            texture=[str(item) for item in foreground.get("texture", [])],
            dominant_colors=[str(item) for item in foreground.get("dominant_colors", [])],
        ),
        middle_ground=RegionSpec(
            region_y_percent=str(middle.get("region_y_percent", "45-60")),
            summary=str(middle.get("summary", "Middle-ground structure was not resolved.")),
            texture=[str(item) for item in middle.get("texture", [])],
            dominant_colors=[str(item) for item in middle.get("dominant_colors", [])],
        ),
        background=RegionSpec(
            region_y_percent="35-60",
            summary="Horizon-adjacent background and distant structures.",
            texture=[],
            dominant_colors=[str(item.get("name", "")) for item in support_signals.get("dominant_colors", []) if isinstance(item, dict)],
        ),
        sky=SkySpec(
            region_y_percent=str(sky.get("region_y_percent", "0-48")),
            cloud_direction=str(sky.get("cloud_direction", "broad layered cloud motion")),
            upper_left=str(sky.get("upper_left", "darker cool-toned cloud mass")),
            upper_right=str(sky.get("upper_right", "warmer cloud mass")),
            center=str(sky.get("center", "brightest glow near the horizon")),
            motion=str(sky.get("motion", "soft atmospheric motion")),
            sharpness=str(sky.get("sharpness", "natural, not hyper-sharp")),
        ),
        color_palette=ColorPaletteSpec(
            dominant=[str(item) for item in color_lighting.get("dominant", [])],
            saturation=str(color_lighting.get("saturation", "moderate")),
            contrast=str(color_lighting.get("contrast", "moderate")),
        ),
        negative_constraints=[str(item) for item in negatives.get("negative_constraints", [])],
        support_signals=support_signals,
    )


def _build_scene_refiner_prompt(scene_map: SceneMap, passes: list[ScenePassResult]) -> str:
    return (
        "Refine the following scene-map JSON without inventing unsupported objects. "
        "Keep the structure strict. Fill missing numeric estimates conservatively. "
        f"Scene map: {scene_map.model_dump(mode='json')} "
        f"Pass evidence: {[item.structured for item in passes]}"
    )


def _refine_scene_map(scene_map: SceneMap, refined_payload: Any) -> SceneMap:
    if isinstance(refined_payload, dict):
        try:
            merged = scene_map.model_dump(mode="json")
            merged = _deep_merge(merged, refined_payload)
            return SceneMap.model_validate(merged)
        except Exception:
            return scene_map
    return scene_map


def _build_visual_hierarchy(scene_map: SceneMap) -> VisualHierarchy:
    main_object_lines = [
        f"{item.name} around x={item.center_x_percent or 'unknown'}%, y={item.center_y_percent or 'unknown'}%"
        for item in scene_map.main_objects
        if item.must_preserve or item.importance == "high"
    ]
    must_match = [
        scene_map.canvas.camera_angle,
        f"horizon around {scene_map.canvas.horizon_y_percent_from_top}% from top",
        scene_map.foreground.summary,
        scene_map.sky.cloud_direction,
        *main_object_lines,
    ]
    should_match = [
        scene_map.middle_ground.summary,
        f"palette: {', '.join(scene_map.color_palette.dominant[:6])}",
    ]
    return VisualHierarchy(
        must_match=[item for item in must_match if item],
        should_match=[item for item in should_match if item],
        can_vary_slightly=[
            "micro-texture details",
            "small cloud cluster edges",
            "fine surface ripple exact shapes",
        ],
        must_avoid=scene_map.negative_constraints,
    )


def _build_prompt_package(scene_map: SceneMap, hierarchy: VisualHierarchy) -> PromptPackage:
    object_lines = []
    for item in scene_map.main_objects:
        position = []
        if item.center_x_percent is not None:
            position.append(f"x={item.center_x_percent}%")
        if item.center_y_percent is not None:
            position.append(f"y={item.center_y_percent}%")
        if item.width_percent is not None:
            position.append(f"w={item.width_percent}%")
        if item.height_percent is not None:
            position.append(f"h={item.height_percent}%")
        object_lines.append(
            f"{item.name}, role {item.role}, {' '.join(position) if position else 'position estimated'}, "
            f"{item.shape}, {item.surface}, color {item.color}"
        )
    base_prompt = (
        f"Realistic image in {scene_map.canvas.aspect_ratio} {scene_map.canvas.orientation} format. "
        f"Camera angle: {scene_map.canvas.camera_angle}. Lens feel: {scene_map.canvas.lens_feel}. "
        f"Horizon near {scene_map.canvas.horizon_y_percent_from_top}% from top. "
        f"Foreground: {scene_map.foreground.summary} "
        f"Middle ground: {scene_map.middle_ground.summary} "
        f"Sky: {scene_map.sky.center}; {scene_map.sky.cloud_direction}."
    )
    precision_prompt = (
        f"Preserve these main objects: {'; '.join(object_lines) or 'no dominant objects resolved'}. "
        f"Foreground textures: {', '.join(scene_map.foreground.texture) or 'not resolved'}. "
        f"Palette: {', '.join(scene_map.color_palette.dominant)}. "
        f"Saturation {scene_map.color_palette.saturation}, contrast {scene_map.color_palette.contrast}."
    )
    negative_prompt = ", ".join(scene_map.negative_constraints) or "no extra invented elements"
    generator_prompt = f"{base_prompt}\n\n{precision_prompt}"
    final_prompt = (
        f"{generator_prompt}\n\nMust match: {'; '.join(hierarchy.must_match)}. "
        f"Should match: {'; '.join(hierarchy.should_match)}. "
        f"Avoid: {'; '.join(hierarchy.must_avoid)}."
    )
    return PromptPackage(
        base_prompt=base_prompt,
        precision_prompt=precision_prompt,
        negative_prompt=negative_prompt,
        generator_prompt=generator_prompt,
        final_prompt=final_prompt,
        notes=["Multi-pass VLM analysis was used to derive this prompt package."],
    )


def _write_prompt_package(prompt_dir: Path, package: PromptPackage, *, version: int, also_write_base: bool = False) -> None:
    dump_text(prompt_dir / f"{version:02d}_generator_prompt_v{version}.txt", package.generator_prompt + "\n")
    dump_text(prompt_dir / f"{version:02d}_negative_prompt_v{version}.txt", package.negative_prompt + "\n")
    if also_write_base:
        dump_text(prompt_dir / "01_base_prompt.txt", package.base_prompt + "\n")
        dump_text(prompt_dir / "02_precision_prompt.txt", package.precision_prompt + "\n")
        dump_text(prompt_dir / "03_negative_prompt.txt", package.negative_prompt + "\n")
        dump_text(prompt_dir / "04_generator_prompt_v1.txt", package.generator_prompt + "\n")


def _generate_image_iteration(
    *,
    iteration: int,
    paths: dict[str, Path],
    prompt_package: PromptPackage,
    generation_config: GenerationConfig,
    enabled: bool,
) -> GenerationResult:
    metadata_path = paths["logs"] / f"generation_v{iteration}.json"
    prompt_file = paths["prompts"] / f"{iteration:02d}_generator_prompt_v{iteration}.txt"
    negative_prompt_file = paths["prompts"] / f"{iteration:02d}_negative_prompt_v{iteration}.txt"
    output_image = paths["generated"] / f"generated_v{iteration}.png"
    generation_payload = generation_config.model_dump(mode="json") | {
        "prompt_file": str(prompt_file),
        "negative_prompt_file": str(negative_prompt_file),
        "output_image": str(output_image),
        "iteration": iteration,
    }
    dump_json(metadata_path, generation_payload)
    if not enabled:
        return GenerationResult(
            enabled=False,
            status="skipped",
            message="Generation disabled. Prompt package is ready for a Qwen-image runner.",
            metadata_path=str(metadata_path),
            seed=generation_config.seed + (iteration - 1),
        )

    command_template = os.environ.get("IMAGE_ANALYZER_QWEN_IMAGE_COMMAND") or generation_payload.get("command_template") or ""
    command_template = str(command_template) or ""
    result = _execute_generation_command(
        command_template=command_template,
        prompt_file=prompt_file,
        negative_prompt_file=negative_prompt_file,
        output_image=output_image,
        generation_config=generation_config,
        iteration=iteration,
    )
    result.metadata_path = str(metadata_path)
    result.seed = generation_config.seed + (iteration - 1)
    return result


def _execute_generation_command(
    *,
    command_template: str,
    prompt_file: Path,
    negative_prompt_file: Path,
    output_image: Path,
    generation_config: GenerationConfig,
    iteration: int,
) -> GenerationResult:
    if not command_template:
        return GenerationResult(
            enabled=True,
            status="failed_config",
            message="Generation requested but no qwen-image command template is configured.",
        )
    command = command_template.format(
        prompt_file=prompt_file,
        negative_prompt_file=negative_prompt_file,
        output_image=output_image,
        width=generation_config.width,
        height=generation_config.height,
        steps=generation_config.steps,
        cfg=generation_config.cfg,
        sampler=generation_config.sampler,
        seed=generation_config.seed + (iteration - 1),
        model=generation_config.model,
    )
    completed = subprocess.run(shlex.split(command), capture_output=True, text=True)
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or f"Generator exited with {completed.returncode}"
        return GenerationResult(enabled=True, status="failed_runtime", message=message)
    if not output_image.exists():
        return GenerationResult(enabled=True, status="failed_runtime", message="Generator command completed but no output image was created.")
    return GenerationResult(
        enabled=True,
        status="completed",
        message=completed.stdout.strip() or "Generation completed.",
        output_image=str(output_image),
    )


def _build_comparison_prompt(scene_map: SceneMap, hierarchy: VisualHierarchy) -> str:
    return (
        "Compare the generated image against the reference image. Focus on composition, object scale and placement, "
        "horizon, lighting, palette, and negative constraints. Return strict JSON with fields: "
        "`overall_similarity_score`, `semantic_similarity_score`, `issues`, and `negative_prompt_additions`. "
        f"Reference expectations: {scene_map.model_dump(mode='json')} hierarchy: {hierarchy.model_dump(mode='json')}"
    )


def _build_comparison_report(payload: Any, hierarchy: VisualHierarchy) -> ComparisonReport:
    if isinstance(payload, dict):
        issues = payload.get("issues") or []
        try:
            return ComparisonReport.model_validate(
                {
                    "overall_similarity_score": float(payload.get("overall_similarity_score", payload.get("semantic_similarity_score", 0.0))),
                    "semantic_similarity_score": float(payload.get("semantic_similarity_score", payload.get("overall_similarity_score", 0.0))),
                    "perceptual_similarity_score": float(payload.get("perceptual_similarity_score", 0.0)),
                    "issues": issues,
                    "negative_prompt_additions": payload.get("negative_prompt_additions", []),
                }
            )
        except Exception:
            pass
    fallback_issues = [
        ComparisonIssue(
            category="constraints",
            issue="No structured comparison returned; preserve existing constraints.",
            severity="medium",
            reference="scene hierarchy",
            generated="unknown",
            prompt_fix="reinforce existing must-match constraints",
        )
    ]
    return ComparisonReport(
        overall_similarity_score=0.0,
        semantic_similarity_score=0.0,
        perceptual_similarity_score=0.0,
        issues=fallback_issues,
        negative_prompt_additions=hierarchy.must_avoid,
    )


def _build_prompt_correction(comparison_report: ComparisonReport, prompt_package: PromptPackage) -> PromptCorrection:
    changes_to_prompt = []
    for issue in comparison_report.issues:
        if not issue.prompt_fix:
            continue
        changes_to_prompt.append(
            {
                "old_instruction": prompt_package.generator_prompt,
                "new_instruction": issue.prompt_fix,
                "reason": issue.issue,
            }
        )
    return PromptCorrection(
        changes_to_prompt=changes_to_prompt,
        changes_to_negative_prompt=list(dict.fromkeys(comparison_report.negative_prompt_additions)),
    )


def _apply_prompt_correction(prompt_package: PromptPackage, correction: PromptCorrection, next_iteration: int) -> PromptPackage:
    correction_lines = [item["new_instruction"] for item in correction.changes_to_prompt if item.get("new_instruction")]
    generator_prompt = prompt_package.generator_prompt
    if correction_lines:
        generator_prompt = f"{generator_prompt}\n\nCorrection round {next_iteration}: " + "; ".join(correction_lines)
    negative_bits = [item.strip() for item in prompt_package.negative_prompt.split(",") if item.strip()]
    negative_bits.extend(correction.changes_to_negative_prompt)
    deduped_negative = ", ".join(dict.fromkeys(negative_bits))
    final_prompt = (
        f"{generator_prompt}\n\nAdditional constraints: {', '.join(correction.changes_to_negative_prompt) if correction.changes_to_negative_prompt else 'none'}."
    )
    return PromptPackage(
        base_prompt=prompt_package.base_prompt,
        precision_prompt=prompt_package.precision_prompt,
        negative_prompt=deduped_negative or prompt_package.negative_prompt,
        generator_prompt=generator_prompt,
        final_prompt=final_prompt,
        notes=prompt_package.notes + [f"Prompt corrected for iteration {next_iteration}."],
    )


def _build_termination(iterations: list[IterationResult], target_score: float, best_score: float) -> LoopTermination:
    if not iterations:
        return LoopTermination(reason="no_iterations", best_iteration=None, best_score=0.0, threshold_reached=False)
    for item in iterations:
        if item.accepted and item.hybrid_score is not None:
            return LoopTermination(
                reason="threshold_met",
                best_iteration=item.iteration,
                best_score=item.hybrid_score.weighted_score,
                threshold_reached=True,
            )
    return LoopTermination(
        reason="max_iterations" if iterations[-1].generation.enabled else iterations[-1].generation.status,
        best_iteration=max(iterations, key=lambda item: item.hybrid_score.weighted_score if item.hybrid_score else 0.0).iteration,
        best_score=max(best_score, 0.0),
        threshold_reached=best_score >= target_score,
    )


def _build_final_prompt_package(report: RunReport, generation_config: GenerationConfig) -> dict[str, Any]:
    best_iteration = report.termination.best_iteration if report.termination else None
    best_result = next((item for item in report.iterations if item.iteration == best_iteration), None)
    return {
        "reference_image": report.reference_image,
        "final_image": best_result.generation.output_image if best_result is not None else report.generation.output_image,
        "final_prompt": report.prompt_package.final_prompt,
        "negative_prompt": report.prompt_package.negative_prompt,
        "scene_map": report.scene_map.model_dump(mode="json"),
        "visual_hierarchy": report.visual_hierarchy.model_dump(mode="json"),
        "generator_settings": generation_config.model_dump(mode="json"),
        "best_iteration": best_iteration,
        "best_score": report.termination.best_score if report.termination else 0.0,
        "termination_reason": report.termination.reason if report.termination else "unknown",
        "warnings": report.warnings,
    }


def _build_final_report(report: RunReport) -> str:
    termination = report.termination or LoopTermination(reason="unknown")
    return "\n".join(
        [
            f"# Closed-Loop Run Report: {report.run_id}",
            "",
            f"- Reference image: `{report.reference_image}`",
            f"- Run directory: `{report.run_dir}`",
            f"- Best iteration: `{termination.best_iteration}`",
            f"- Best score: `{termination.best_score:.4f}`",
            f"- Termination: `{termination.reason}`",
            f"- Warnings: {', '.join(report.warnings) if report.warnings else 'none'}",
            "",
            "## Must Match",
            *[f"- {item}" for item in report.visual_hierarchy.must_match],
            "",
            "## Final Prompt",
            "",
            report.prompt_package.final_prompt,
        ]
    ) + "\n"


def _aspect_ratio_string(width: int, height: int) -> str:
    if width == 0 or height == 0:
        return "unknown"
    from math import gcd

    divisor = gcd(width, height)
    return f"{width // divisor}:{height // divisor}"


def _camera_angle_heuristic(raw_response: str) -> str:
    lowered = raw_response.lower()
    if "low angle" in lowered or "close to the ground" in lowered:
        return "very low angle close to the surface"
    if "overhead" in lowered:
        return "overhead angle"
    return "eye-level to low landscape angle"


def _objects_from_text(raw_response: str) -> list[dict[str, Any]]:
    lowered = raw_response.lower()
    if "rock" in lowered:
        return [
            {
                "name": "single dark rock",
                "role": "foreground anchor",
                "center_x_percent": 78.0,
                "center_y_percent": 78.0,
                "width_percent": 9.0,
                "height_percent": 7.0,
                "shape": "rounded oval dome",
                "color": "dark cool-toned rock",
                "surface": "wet glossy surface",
                "importance": "high",
                "must_preserve": True,
            }
        ]
    return []


def _top_phrases(raw_response: str, *, fallback: list[str]) -> list[str]:
    parts = [item.strip(" .") for item in raw_response.replace("\n", " ").split(",")]
    results = [item for item in parts if item][:4]
    return results or fallback


def _negative_constraints_from_text(raw_response: str) -> list[str]:
    candidates = [item.strip(" .") for item in raw_response.replace("\n", ",").split(",")]
    results = [item for item in candidates if item.lower().startswith("no ") or item.lower().startswith("not ")]
    return results or ["no extra invented subjects", "not hyper-saturated neon", "no major crop change"]


def _slugify(value: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")
    return safe or "run"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
