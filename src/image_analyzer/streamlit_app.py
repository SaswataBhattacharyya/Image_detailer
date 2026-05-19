from __future__ import annotations

import tempfile
from pathlib import Path

from image_analyzer.cli.batch import resolve_batch_inputs
from image_analyzer.config.settings import load_settings
from image_analyzer.detailed_pipeline import run_batch_flow, run_image_flow
from image_analyzer.models.schemas import RunReport


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Image Recreation Agent", layout="wide")
    st.title("Image Recreation Agent")
    st.caption("One unified OpenClaw-style flow: interrogate the image with a VLM, build scene memory, generate, score, and restart until the similarity threshold is met or retries are exhausted.")

    config = load_settings(PROJECT_ROOT)

    with st.sidebar:
        mode = st.radio("Mode", ["Single image", "Batch folder"], index=0)
        output_root_raw = st.text_input("Run folder root", value=str(config.detailed.run_root_dir))
        project_name = st.text_input("Project name", value=config.detailed.default_project_name)
        target_score = st.slider("Similarity threshold", min_value=0.50, max_value=0.99, value=float(config.detailed.target_score), step=0.01)
        max_full_restarts = st.number_input("Max full restarts", min_value=1, max_value=10, value=int(config.detailed.max_full_restarts), step=1)
        max_question_rounds = st.number_input("Max question rounds", min_value=1, max_value=20, value=int(config.detailed.max_question_rounds), step=1)

    output_root = Path(output_root_raw).expanduser()
    if mode == "Single image":
        _render_single_mode(output_root, project_name, target_score, int(max_full_restarts), int(max_question_rounds))
    else:
        _render_batch_mode(output_root, project_name)


def _render_single_mode(
    output_root: Path,
    project_name: str,
    target_score: float,
    max_full_restarts: int,
    max_question_rounds: int,
) -> None:
    import streamlit as st

    path_value = st.text_input("Image path", value="")
    uploaded_file = st.file_uploader("Or upload one image", type=["png", "jpg", "jpeg", "webp", "bmp"])
    run = st.button("Run recreation flow", type="primary")

    if not run:
        return

    image_path = _resolve_single_input(path_value, uploaded_file)
    if image_path is None:
        st.error("Provide a valid image path or upload an image.")
        return

    output_root.mkdir(parents=True, exist_ok=True)
    with st.spinner("Running unified image recreation flow..."):
        report = run_image_flow(
            image_path,
            load_settings(PROJECT_ROOT),
            output_root=output_root,
            project_name=project_name,
            target_score=target_score,
            max_full_restarts=max_full_restarts,
            max_question_rounds=max_question_rounds,
        )
    _render_report(report, batch_index=None)


def _render_batch_mode(output_root: Path, project_name: str) -> None:
    import streamlit as st

    batch_path_value = st.text_input("Input folder or manifest path", value="")
    run = st.button("Run batch recreation flow", type="primary")

    if not run:
        return

    if not batch_path_value.strip():
        st.error("Provide an input folder or manifest path for batch processing.")
        return

    input_path = Path(batch_path_value).expanduser()
    try:
        image_paths = resolve_batch_inputs(input_path)
    except Exception as exc:
        st.error(f"Unable to resolve batch inputs: {exc}")
        return

    if not image_paths:
        st.warning("No supported images were found in the provided location.")
        return

    output_root.mkdir(parents=True, exist_ok=True)
    with st.spinner("Running unified batch recreation flow..."):
        reports = run_batch_flow(
            image_paths,
            load_settings(PROJECT_ROOT),
            output_root=output_root,
            project_name=project_name,
        )

    st.success(f"Finished {len(reports)} image runs.")
    for index, report in enumerate(reports, start=1):
        with st.expander(f"{index}. {Path(report.reference_image).name} | {round(report.termination.best_score * 100.0, 2)}%", expanded=index == 1):
            _render_report(report, batch_index=index)


def _resolve_single_input(path_value: str, uploaded_file: object) -> Path | None:
    if path_value.strip():
        candidate = Path(path_value).expanduser()
        return candidate if candidate.exists() else None
    if uploaded_file is None:
        return None

    upload_dir = Path(tempfile.mkdtemp(prefix="image-analyzer-upload-"))
    target = upload_dir / str(getattr(uploaded_file, "name", "uploaded-image.png"))
    target.write_bytes(uploaded_file.getbuffer())
    return target


def _render_report(report: RunReport, batch_index: int | None) -> None:
    import streamlit as st

    if batch_index is None:
        st.subheader("Current Run")

    score_percent = round(report.termination.best_score * 100.0, 2)
    restart_count = len(report.iterations)
    left, right = st.columns([1.0, 1.1])
    run_dir = Path(report.run_dir)
    model_calls_path = run_dir / "logs" / "model_calls.jsonl"
    gap_history_path = run_dir / "logs" / "gap_history.json"
    latest_hybrid = report.iterations[-1].hybrid_score if report.iterations else None

    with left:
        st.image(report.reference_image, caption="Reference image", use_container_width=True)
        if report.generation.output_image and Path(report.generation.output_image).exists():
            st.image(report.generation.output_image, caption="Generated image", use_container_width=True)
        else:
            st.info("No generated image was produced for this run.")

    with right:
        metric_a, metric_b, metric_c = st.columns(3)
        metric_a.metric("Similarity", f"{score_percent}%")
        metric_b.metric("Restarts", str(restart_count))
        metric_c.metric("Status", report.termination.reason if report.termination else "unknown")
        st.write(f"Run directory: `{report.run_dir}`")
        st.write(f"Termination: `{report.termination.reason}`")
        st.subheader("Logs")
        if report.passes:
            pass_lines = []
            for item in report.passes:
                preview = (item.raw_response or "").strip().replace("\n", " ")
                pass_lines.append(f"{item.pass_key} | {preview[:220]}")
            st.code("\n\n".join(pass_lines), language="text")
        else:
            st.info("No pass logs were recorded.")

        if report.warnings:
            st.warning("\n".join(report.warnings))

        st.subheader("Current Result")
        st.write(f"Similarity result: **{score_percent}%**")
        if latest_hybrid is not None:
            st.caption(
                f"Perceptual similarity: {round(latest_hybrid.perceptual.perceptual_similarity_score * 100.0, 2)}%"
            )
        st.write(report.prompt_package.final_prompt)

    tab_prompt, tab_memory, tab_questions, tab_logs, tab_similarity, tab_json = st.tabs(
        ["Prompt", "Scene Memory", "Question Loop", "Run Logs", "Similarity", "Structured Output"]
    )
    with tab_prompt:
        st.code(report.prompt_package.generator_prompt, language="text")
        st.caption("Negative prompt")
        st.code(report.prompt_package.negative_prompt, language="text")
        if report.visual_hierarchy.must_avoid:
            st.caption("Critical constraints")
            st.write("\n".join(f"- {item}" for item in report.visual_hierarchy.must_match + report.visual_hierarchy.must_avoid))
    with tab_memory:
        if report.final_scene_memory is not None:
            st.json(report.final_scene_memory.model_dump(mode="json"))
        else:
            st.info("No final scene memory recorded.")
    with tab_questions:
        if report.question_history:
            for item in report.question_history:
                with st.expander(f"Round {item.iteration}: {item.topic}", expanded=False):
                    st.write(f"Question: {item.question}")
                    st.write(item.answer)
        else:
            st.info("No follow-up questions were recorded.")
    with tab_logs:
        if model_calls_path.exists():
            st.caption("Model calls")
            st.code(model_calls_path.read_text(encoding="utf-8"), language="json")
        else:
            st.info("No model call log file found.")
        if report.passes:
            st.caption("Pass responses")
            for item in report.passes:
                with st.expander(item.title, expanded=item.pass_key == "overview"):
                    st.caption(item.prompt)
                    st.code(item.raw_response or "(empty response)", language="text")
                    if item.warnings:
                        st.warning("\n".join(item.warnings))
        if gap_history_path.exists():
            st.caption("Gap history")
            st.code(gap_history_path.read_text(encoding="utf-8"), language="json")
    with tab_similarity:
        st.write(f"Final similarity: **{score_percent}%**")
        if report.comparison is not None:
            st.json(report.comparison.model_dump(mode="json"))
        if latest_hybrid is not None:
            st.json(latest_hybrid.model_dump(mode="json"))
        else:
            st.info("No similarity artifact was recorded.")
    with tab_json:
        st.json(report.model_dump(mode="json"))


if __name__ == "__main__":
    main()
