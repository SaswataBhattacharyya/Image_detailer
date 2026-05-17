from __future__ import annotations

import tempfile
from pathlib import Path

from image_analyzer.cli.batch import resolve_batch_inputs
from image_analyzer.config.settings import load_settings
from image_analyzer.models.schemas import AnalysisEvent, AnalysisResult
from image_analyzer.pipeline import analyze_image


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Image Analyzer", layout="wide")
    st.title("Image Analyzer")
    st.caption("Measured-first image analysis with live orchestration events and layered output files.")

    config = load_settings(PROJECT_ROOT)

    with st.sidebar:
        mode = st.radio("Mode", ["Single image", "Batch folder"], index=0)
        output_root_raw = st.text_input("Output folder", value=str(config.paths.artifact_dir))
        save_debug = st.checkbox("Save debug module outputs", value=config.pipeline.save_debug_by_default)
        st.markdown(
            "\n".join(
                [
                    "`summary`",
                    "`detailed_description`",
                    "`colors_and_materials`",
                    "`composition_and_camera`",
                    "`emotion_style_or_intent`",
                    "`ocr_and_context`",
                ]
            )
        )

    output_root = Path(output_root_raw).expanduser()
    if mode == "Single image":
        _render_single_mode(output_root, save_debug)
    else:
        _render_batch_mode(output_root, save_debug)


def _render_single_mode(output_root: Path, save_debug: bool) -> None:
    import streamlit as st

    path_value = st.text_input("Image path", value="")
    uploaded_file = st.file_uploader("Or upload one image", type=["png", "jpg", "jpeg", "webp", "bmp"])
    run = st.button("Analyze image", type="primary")

    if not run:
        return

    image_path = _resolve_single_input(path_value, uploaded_file)
    if image_path is None:
        st.error("Provide a valid image path or upload an image.")
        return

    _run_analysis([image_path], output_root, save_debug, batch_mode=False)


def _render_batch_mode(output_root: Path, save_debug: bool) -> None:
    import streamlit as st

    batch_path_value = st.text_input("Input folder or manifest path", value="")
    run = st.button("Analyze batch", type="primary")

    if not run:
        return

    if not batch_path_value.strip():
        st.error("Provide an input folder or manifest path for batch analysis.")
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

    _run_analysis(image_paths, output_root, save_debug, batch_mode=True)


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


def _run_analysis(image_paths: list[Path], output_root: Path, save_debug: bool, batch_mode: bool) -> None:
    import streamlit as st

    output_root.mkdir(parents=True, exist_ok=True)
    config = load_settings(PROJECT_ROOT)

    left_col, right_col = st.columns([1.1, 1.4])
    progress_bar = st.progress(0.0)
    batch_status = st.empty()
    image_slot = left_col.empty()
    latest_event_slot = left_col.empty()
    event_log_slot = right_col.empty()
    result_slot = right_col.empty()

    collected_events: list[AnalysisEvent] = []
    completed_results: list[AnalysisResult] = []

    def on_event(event: AnalysisEvent) -> None:
        collected_events.append(event)
        latest_event_slot.info(f"{event.stage} [{event.status}] {event.message}")
        event_lines = [
            f"{item.timestamp.isoformat()} | {item.stage} | {item.status} | {item.message}"
            for item in collected_events[-20:]
        ]
        event_log_slot.code("\n".join(event_lines), language="text")

    total = len(image_paths)
    for index, image_path in enumerate(image_paths, start=1):
        batch_status.write(f"Processing {index}/{total}: `{image_path.name}`")
        image_slot.image(str(image_path), caption=image_path.name, use_container_width=True)
        result = analyze_image(image_path, config, output_root=output_root, save_debug=save_debug, event_callback=on_event)
        completed_results.append(result)
        progress_bar.progress(index / total)
        result_slot.empty()
        _render_result(result_slot.container(), result, output_root)

    if batch_mode:
        st.success(f"Finished batch analysis for {len(completed_results)} images.")
    else:
        st.success("Finished image analysis.")


def _render_result(container, result: AnalysisResult, output_root: Path) -> None:
    import streamlit as st

    bundle_dir = output_root / Path(result.image.file_name).stem
    container.subheader("Current Result")
    container.write(f"Bundle: `{bundle_dir}`")
    container.write(result.summary.long_description)

    layer_tab, data_tab, event_tab = container.tabs(["Layered Descriptions", "Structured Output", "Events"])
    with layer_tab:
        for layer in result.description_layers:
            with st.expander(f"{layer.title} ({layer.file_name})", expanded=layer.key == "summary"):
                st.write(layer.text)
    with data_tab:
        st.json(result.model_dump())
    with event_tab:
        st.json([event.model_dump(mode="json") for event in result.orchestration_events])


if __name__ == "__main__":
    main()
