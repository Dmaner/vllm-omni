# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SenseNova FP16/BF16 vs online-FP8 output-quality evaluation.

Opt in with SENSENOVA_RUN_FP8_EVAL=1.
Quality thresholds must be supplied explicitly; none are imposed by default.
"""

import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest
import torch
from PIL import Image, ImageDraw
from tabulate import tabulate  # type: ignore[import-untyped]

from tests.e2e.accuracy.helpers import compute_image_ssim_psnr
from tests.helpers.runtime import OmniRunner
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.diffusion.utils.image_output import extract_images_from_outputs
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion, pytest.mark.cuda, pytest.mark.gpu, pytest.mark.cards_1]

MODEL_ID = "SenseNova/SenseNova-U1.5-8B-MoT"
BASELINE_DTYPES = {"fp16": "float16", "bf16": "bfloat16"}
ATTENTION_BACKEND = "TORCH_SDPA"
WIDTH = HEIGHT = 1024
NUM_STEPS = 50
UNDERSTANDING_SEED = 42
GENERATION_ARGS = {
    "think": False,
    "cfg_scale": 4.0,
    "cfg_norm": "none",
    "cfg_interval": [0.0, 1.0],
    "timestep_shift": 3.0,
    "t_eps": 0.02,
    "batch_size": 1,
}
UNDERSTANDING_ARGS = {"max_tokens": 768, "do_sample": False, "temperature": 0.0}
GENERATION_CASES = [
    {"id": "apple_s42", "prompt": "A red apple on a wooden table, natural daylight.", "seed": 42},
    {"id": "ducks_s42", "prompt": "Three yellow rubber ducks in a row on a blue table.", "seed": 42},
    {
        "id": "butterfly_s123",
        "prompt": "A butterfly resting on a purple flower, macro photography, detailed wings and pollen.",
        "seed": 123,
    },
]


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _generation_cases() -> list[dict]:
    source = os.environ.get("SENSENOVA_FP8_CASES")
    cases = json.loads(Path(source).read_text()) if source else GENERATION_CASES
    assert isinstance(cases, list) and cases, "Generation cases must be a nonempty list"
    identifiers = set()
    for case in cases:
        assert isinstance(case, dict) and {"id", "prompt", "seed"} <= case.keys(), case
        assert isinstance(case["id"], str) and re.fullmatch(r"[A-Za-z0-9_-]+", case["id"]), case
        assert case["id"] not in identifiers and type(case["seed"]) is int, case
        assert isinstance(case["prompt"], str) and case["prompt"].strip(), case
        assert type(case.get("allow_uniform", False)) is bool, case
        identifiers.add(case["id"])
    return cases


def _quality_thresholds() -> dict[str, float]:
    path = os.environ.get("SENSENOVA_FP8_THRESHOLDS")
    values = json.loads(Path(path).read_text()) if path else {}
    allowed = {"ssim_min", "psnr_min_db", "lpips_max", "understanding_min_accuracy"}
    assert isinstance(values, dict) and not values.keys() - allowed, f"Allowed threshold keys: {sorted(allowed)}"
    assert all(type(value) in (int, float) and math.isfinite(value) for value in values.values()), values
    assert 0 <= values.get("understanding_min_accuracy", 0) <= 1, values
    return values


def _understanding_cases(directory: Path) -> list[dict]:
    square = Image.new("RGB", (512, 512), "white")
    ImageDraw.Draw(square).rectangle((128, 128, 383, 383), fill="red")
    square.save(directory / "red_square.png")
    circles = Image.new("RGB", (512, 512), "white")
    draw = ImageDraw.Draw(circles)
    for bounds in ((48, 176, 207, 335), (304, 176, 463, 335)):
        draw.ellipse(bounds, fill="blue")
    circles.save(directory / "blue_circles.png")
    questions = [
        ("square_color", "red_square.png", "What color is the square? Answer with only the color name.", ["red"]),
        (
            "square_shape",
            "red_square.png",
            "What shape is the red object? Answer with only the shape name.",
            ["square"],
        ),
        ("circles_color", "blue_circles.png", "What color are the circles? Answer with only the color name.", ["blue"]),
        ("circles_count", "blue_circles.png", "How many circles are shown? Answer with only the number.", ["2", "two"]),
        ("arithmetic", None, "What is two plus two? Answer with only the number.", ["4", "four"]),
    ]
    return [
        {"id": name, "image": image, "prompt": prompt, "answers": answers} for name, image, prompt, answers in questions
    ]


def _final_answer(text: str) -> str:
    thinking = False
    for match in re.finditer(r"</?think>", text):
        opening = match.group() == "<think>"
        if opening == thinking:
            return ""
        thinking = opening
    return "" if thinking else text.rsplit("</think>", 1)[-1].strip()


def _normalize_answer(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", text.casefold()).split())


def _run_model(
    model: str, config: dict, output_dir: Path, *, quantization: str | None
) -> tuple[list[Image.Image], list[dict]]:
    output_dir.mkdir()
    images, answers = [], []
    with OmniRunner(
        model,
        dtype=config["dtype"],
        mode="text-to-image",
        quantization=quantization,
        diffusion_attention_backend=config["attention_backend"],
        enforce_eager=config["enforce_eager"],
        parallel_config=DiffusionParallelConfig(tensor_parallel_size=config["tensor_parallel_size"]),
    ) as runner:
        for case in config["generation_cases"]:
            params = OmniDiffusionSamplingParams(
                width=config["width"],
                height=config["height"],
                num_inference_steps=config["steps"],
                seed=case["seed"],
                extra_args=config["generation_args"].copy(),
            )
            outputs = list(runner.omni.generate({"prompt": case["prompt"], "modalities": ["image"]}, params))
            generated = extract_images_from_outputs(outputs)
            assert len(generated) == 1 and generated[0].size == (config["width"], config["height"]), case["id"]
            image = generated[0].convert("RGB")
            image.save(output_dir / f"{case['id']}.png")
            images.append(image)
        for case in config["understanding_cases"]:
            prompt = {"prompt": case["prompt"], "modalities": ["text"]}
            if case["image"]:
                with Image.open(output_dir.parent / case["image"]) as image:
                    prompt["multi_modal_data"] = {"image": [image.convert("RGB")]}
            params = OmniDiffusionSamplingParams(
                seed=config["understanding_seed"], extra_args=config["understanding_args"].copy()
            )
            texts = []
            for output in runner.omni.generate(prompt, params):
                text = (getattr(output, "multimodal_output", None) or {}).get("text")
                if isinstance(text, str):
                    texts.append(text)
            assert len(texts) == 1, f"{case['id']}: expected one text output"
            answer = _final_answer(texts[0])
            normalized = _normalize_answer(answer)
            answers.append(
                {"id": case["id"], "text": texts[0], "answer": answer, "exact_match": normalized in case["answers"]}
            )
            _write_json(output_dir / "answers.json", {"answers": answers})
    return images, answers


def _image_metrics(
    cases: list[dict], baseline_images: list[Image.Image], fp8_images: list[Image.Image], use_lpips: bool
) -> list[dict]:
    """Measure valid RGB pairs; a uniform image needs explicit case permission."""
    assert len(cases) == len(baseline_images) == len(fp8_images), "Image sequence lengths differ"
    rows, lpips_pairs = [], []
    for case, baseline, fp8 in zip(cases, baseline_images, fp8_images):
        baseline, fp8 = baseline.convert("RGB"), fp8.convert("RGB")
        row = {
            **case,
            "ssim": None,
            "psnr_db": None,
            "psnr_positive_infinity": False,
            "lpips256": None,
            "invalid_outputs": [],
        }
        rows.append(row)
        if baseline.size != fp8.size:
            row["invalid_outputs"].append(f"{case['id']}: image dimensions differ")
        for label, image in (("baseline", baseline), ("fp8", fp8)):
            if case.get("allow_uniform") is not True and image.getcolors(maxcolors=1) is not None:
                row["invalid_outputs"].append(f"{case['id']}: {label} image is uniform")
        if row["invalid_outputs"]:
            continue
        ssim, psnr = compute_image_ssim_psnr(prediction=fp8, reference=baseline, compare_mode="RGB")
        identical = baseline.tobytes() == fp8.tobytes()
        row["ssim"] = ssim if math.isfinite(ssim) else None
        row["psnr_db"] = psnr if math.isfinite(psnr) else None
        row["psnr_positive_infinity"] = psnr == math.inf and identical
        if row["ssim"] is None or (row["psnr_db"] is None and not row["psnr_positive_infinity"]):
            row["invalid_outputs"].append(f"{case['id']}: invalid SSIM or PSNR")
        if not row["invalid_outputs"]:
            lpips_pairs.append((row, baseline, fp8))
    if use_lpips and lpips_pairs:
        from benchmarks.diffusion.quantization_quality import compute_lpips_images

        scores = compute_lpips_images([p[1] for p in lpips_pairs], [p[2] for p in lpips_pairs], net="alex")
        assert len(scores) == len(lpips_pairs), "LPIPS result count differs from valid image pairs"
        for (row, _, _), score in zip(lpips_pairs, scores):
            if math.isfinite(score):
                row["lpips256"] = score
            else:
                row["invalid_outputs"].append(f"{row['id']}: non-finite LPIPS")
    return rows


def _understanding_metrics(cases: list[dict], baseline_rows: list[dict], fp8_rows: list[dict]) -> list[dict]:
    """Keep both answers and exact-match results; blank answers are invalid."""
    assert len(cases) == len(baseline_rows) == len(fp8_rows), "Understanding sequence lengths differ"
    rows = []
    for case, baseline, fp8 in zip(cases, baseline_rows, fp8_rows):
        assert case["id"] == baseline["id"] == fp8["id"], "Understanding case IDs differ"
        row = {
            "id": case["id"],
            "answers": case["answers"],
            "baseline_answer": baseline["answer"],
            "fp8_answer": fp8["answer"],
            "baseline_exact_match": baseline["exact_match"],
            "fp8_exact_match": fp8["exact_match"],
            "answers_equal": _normalize_answer(baseline["answer"]) == _normalize_answer(fp8["answer"]),
            "invalid_outputs": [],
        }
        for label in ("baseline", "fp8"):
            if not row[f"{label}_answer"].strip():
                row["invalid_outputs"].append(f"{case['id']}: {label} answer is empty")
        rows.append(row)
    return rows


def _quality_gate(image_rows: list[dict], text_rows: list[dict], thresholds: dict[str, float]) -> dict:
    """Check output validity first, then only thresholds supplied by the caller."""
    invalid = [issue for row in image_rows + text_rows for issue in row["invalid_outputs"]]
    if "lpips_max" in thresholds and any(row["lpips256"] is None for row in image_rows):
        invalid.append("LPIPS threshold requires an LPIPS measurement for every image pair")
    if "understanding_min_accuracy" in thresholds and not text_rows:
        invalid.append("Understanding threshold requires understanding samples")
    image_keys = {"ssim_min", "psnr_min_db", "lpips_max"}
    if image_keys.intersection(thresholds) and not image_rows:
        invalid.append("Image thresholds require image samples")
    if invalid:
        return {"status": "invalid", "failures": [], "invalid_outputs": invalid}
    assert not set(thresholds) - image_keys - {"understanding_min_accuracy"}, "Unknown quality threshold"
    assert all(type(v) in (int, float) and math.isfinite(v) for v in thresholds.values()), "Invalid threshold"
    if "understanding_min_accuracy" in thresholds:
        assert 0 <= thresholds["understanding_min_accuracy"] <= 1, "Accuracy threshold must be in [0, 1]"
    failures = []
    for row in image_rows:
        for metric, key in (("ssim", "ssim_min"), ("psnr_db", "psnr_min_db"), ("lpips256", "lpips_max")):
            if key not in thresholds:
                continue
            value = math.inf if metric == "psnr_db" and row["psnr_positive_infinity"] else row[metric]
            outside = value > thresholds[key] if key == "lpips_max" else value < thresholds[key]
            if outside:
                failures.append(f"{row['id']}: {metric}={value}, {key}={thresholds[key]}")
    if "understanding_min_accuracy" in thresholds:
        for label in ("baseline", "fp8"):
            accuracy = sum(row[f"{label}_exact_match"] for row in text_rows) / len(text_rows)
            if accuracy < thresholds["understanding_min_accuracy"]:
                failures.append(f"{label}: understanding accuracy={accuracy}")
    return {
        "status": ("failed" if failures else "passed") if thresholds else "not_configured",
        "failures": failures,
        "invalid_outputs": [],
    }


def _print_results(report: dict) -> None:
    baseline = report["config"]["baseline"].upper()
    print(f"\nSenseNova generation: FP8 vs {baseline}")
    print(
        tabulate(
            [
                [
                    row["id"],
                    "invalid" if row["invalid_outputs"] else "valid",
                    row["ssim"],
                    "inf" if row["psnr_positive_infinity"] else row["psnr_db"],
                    row["lpips256"],
                ]
                for row in report["generation"]
            ],
            headers=["Case", "Output", "SSIM ↑", "PSNR (dB) ↑", "LPIPS@256 ↓"],
            floatfmt=".6f",
            tablefmt="grid",
        )
    )
    print("\nSenseNova understanding")
    print(
        tabulate(
            [
                [
                    row["id"],
                    "/".join(row["answers"]),
                    row["baseline_answer"],
                    row["fp8_answer"],
                    row["baseline_exact_match"],
                    row["fp8_exact_match"],
                ]
                for row in report["understanding"]
            ],
            headers=["Case", "Expected", f"{baseline} answer", "FP8 answer", f"{baseline} correct", "FP8 correct"],
            tablefmt="grid",
        )
    )
    for label, display in (("baseline", baseline), ("fp8", "FP8")):
        correct = sum(row[f"{label}_exact_match"] for row in report["understanding"])
        total = len(report["understanding"])
        print(f"{display}: {correct}/{total}, exact match={correct / total:.2%}")
    print(f"Quality gate: {report['gate']}")


@pytest.mark.skipif(os.environ.get("SENSENOVA_RUN_FP8_EVAL") != "1", reason="Set SENSENOVA_RUN_FP8_EVAL=1")
def test_sensenova_u15_fp8_accuracy(accuracy_artifact_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert torch.cuda.is_available(), "This evaluation requires a CUDA GPU"
    baseline = os.environ.get("SENSENOVA_BASELINE_DTYPE", "fp16").lower()
    assert baseline in BASELINE_DTYPES, "SENSENOVA_BASELINE_DTYPE must be fp16 or bf16"
    thresholds = _quality_thresholds()
    use_lpips = os.environ.get("SENSENOVA_FP8_LPIPS") == "1"
    assert use_lpips or "lpips_max" not in thresholds, "lpips_max requires SENSENOVA_FP8_LPIPS=1"
    if use_lpips:
        import lpips

        lpips.LPIPS(net="alex").eval()  # Prepare metric weights before loading the model.
    model = os.environ.get("SENSENOVA_U15_MODEL", MODEL_ID)
    revision = os.environ.get("SENSENOVA_U15_REVISION")
    if not Path(model).is_dir():
        from vllm_omni.transformers_utils.repo_utils import hf_api

        model = hf_api().snapshot_download(repo_id=model, revision=revision)
    else:
        assert revision is None, "Use a fixed local snapshot without SENSENOVA_U15_REVISION"
    name = "sensenova-u15-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = Path(os.environ.get("SENSENOVA_FP8_OUTPUT_DIR", str(accuracy_artifact_root / name))).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    assert not any(output_dir.iterdir()), f"Refusing to overwrite nonempty directory: {output_dir}"
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    monkeypatch.setenv("DIFFUSION_ATTENTION_BACKEND", ATTENTION_BACKEND)
    monkeypatch.setenv("VLLM_OMNI_SENSENOVA_PAGED_DECODE", "0")
    config = {
        "model": model,
        "baseline": baseline,
        "dtype": BASELINE_DTYPES[baseline],
        "attention_backend": ATTENTION_BACKEND,
        "enforce_eager": True,
        "tensor_parallel_size": 1,
        "batch_size": 1,
        "paged_decode": False,
        "width": WIDTH,
        "height": HEIGHT,
        "steps": NUM_STEPS,
        "generation_args": GENERATION_ARGS,
        "understanding_seed": UNDERSTANDING_SEED,
        "understanding_args": UNDERSTANDING_ARGS,
        "generation_cases": _generation_cases(),
        "understanding_cases": _understanding_cases(output_dir),
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "disabled_kernels": os.environ.get("VLLM_DISABLED_KERNELS"),
        "lpips": use_lpips,
    }
    _write_json(output_dir / "config.json", config)
    baseline_images, baseline_answers = _run_model(model, config, output_dir / baseline, quantization=None)
    fp8_images, fp8_answers = _run_model(model, config, output_dir / "fp8", quantization="fp8")
    generation = _image_metrics(config["generation_cases"], baseline_images, fp8_images, use_lpips)
    understanding = _understanding_metrics(config["understanding_cases"], baseline_answers, fp8_answers)
    gate = _quality_gate(generation, understanding, thresholds)
    report = {
        "config": config,
        "generation": generation,
        "understanding": understanding,
        "thresholds": thresholds,
        "gate": gate,
    }
    _write_json(output_dir / "metrics.json", report)
    _print_results(report)
    print(f"Metrics: {output_dir / 'metrics.json'}")
    assert gate["status"] not in ("invalid", "failed"), gate
