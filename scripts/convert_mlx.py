"""learning-unit/L1-30B-A5B를 MLX 형식으로 변환(양자화)합니다.

이 모델의 model_type은 "gravity_moe"라 MLX가 직접 알지 못합니다. 하지만 모델 코드가
transformers의 DeepSeek V3 구현을 그대로 상속하고 가중치 이름도 같아서, 설정 파일의
model_type만 "deepseek_v3"로 바꾸면 MLX의 DeepSeek V3 구현으로 변환할 수 있습니다.
(원본을 CPU에서 실행한 결과와 비교해 MLX bf16 99.1%, 8비트 96.3%의 다음 토큰 예측 일치를 확인)

원본 파일은 복사하지 않고, 임시 폴더에 링크 + 수정한 config.json만 만들어 변환합니다.

실행:
    uv run scripts/convert_mlx.py            # 8비트 → models/L1-30B-A5B-mlx-8bit
    uv run scripts/convert_mlx.py --bits 4   # 4비트 → models/L1-30B-A5B-mlx-4bit
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download
from mlx_lm import convert

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app import MODEL_NAME, MODEL_REVISION  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bits", type=int, default=8, choices=[4, 8], help="양자화 비트 수 (기본 8)")
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--output", type=Path, help="저장 위치 (기본 models/L1-30B-A5B-mlx-<bits>bit)")
    args = parser.parse_args()
    output = args.output or ROOT / "models" / f"L1-30B-A5B-mlx-{args.bits}bit"
    if output.exists():
        sys.exit(f"이미 존재합니다: {output}\n다시 변환하려면 먼저 지우세요.")

    # 원본 모델 위치 (캐시에 없으면 내려받음). 변환에 필요한 파일만 대상으로 함
    # (README, LICENSE 등은 transformers 실행 시 받지 않으므로 캐시에 없음)
    src = Path(snapshot_download(
        MODEL_NAME, revision=MODEL_REVISION, allow_patterns=["*.json", "*.safetensors", "*.jinja"]
    ))
    print(f"원본: {src}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for f in src.iterdir():
            if f.name == "config.json" or f.suffix == ".py":
                continue
            (tmp / f.name).symlink_to(f.resolve())
        config = json.loads((src / "config.json").read_text())
        config["model_type"] = "deepseek_v3"
        config["architectures"] = ["DeepseekV3ForCausalLM"]
        config.pop("auto_map", None)
        (tmp / "config.json").write_text(json.dumps(config, indent=2))

        output.parent.mkdir(parents=True, exist_ok=True)
        convert(
            hf_path=str(tmp),
            mlx_path=str(output),
            quantize=True,
            q_bits=args.bits,
            q_group_size=args.group_size,
        )
    print(f"완료: {output}")


if __name__ == "__main__":
    main()
