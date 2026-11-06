"""실제 모델 스모크 테스트 — 모델을 로드해 짧은 질문 하나에 답하게 하고 결과와 속도를 출력합니다.

모델을 메모리에 올리므로 app.py가 실행 중이면 먼저 종료하세요.
(MLX 8비트 약 32GB, transformers 약 59GB)

실행 예:
    uv run tests/smoke_test.py                          # MLX
    uv run tests/smoke_test.py --backend transformers   # 원본 + transformers (느림)
    uv run tests/smoke_test.py --prompt "후기 감속이 무엇인가요?" --max-new-tokens 1500
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import app  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["mlx", "transformers"], default="mlx")
    parser.add_argument("--mlx-model", type=Path, default=app.MLX_MODEL_DIR)
    parser.add_argument("--prompt", default="감기와 독감의 차이를 두 문장으로 알려주세요.")
    parser.add_argument("--max-new-tokens", type=int, default=800,
                        help="추론 포함 생성 한도 (작을수록 빨리 끝남)")
    args = parser.parse_args()

    t0 = time.time()
    app.backend = app.load_backend(args.backend, args.mlx_model)
    print(f"[로딩] {time.time() - t0:.0f}초, 백엔드={app.backend.label}")

    app.MAX_NEW_TOKENS = args.max_new_tokens
    history = [  # 이전 턴(추론 메시지 포함)이 있는 상황도 함께 확인
        {"role": "user", "content": [{"type": "text", "text": "안녕하세요"}], "metadata": None},
        {"role": "assistant", "content": [{"type": "text", "text": "인사에 답한다."}],
         "metadata": {"title": "추론 과정", "status": "done"}},
        {"role": "assistant", "content": [{"type": "text", "text": "안녕하세요! 무엇을 도와드릴까요?"}],
         "metadata": None},
    ]

    # 생성 토큰 수를 세기 위해 backend.generate를 감쌈
    count = {}
    original = app.backend.generate

    def counting_generate(*a, **kw):
        count["n"] = original(*a, **kw)
        return count["n"]

    app.backend.generate = counting_generate

    t1 = time.time()
    first_chunk = answer_start = None
    final = []
    for out in app.medical_chat(args.prompt, history):
        now = time.time() - t1
        first_chunk = first_chunk or now
        if len(out) > 1 and answer_start is None:
            answer_start = now
        final = out
    elapsed = time.time() - t1

    thought = final[0]
    answer = final[1].content if len(final) > 1 else ""
    print(f"[생성] {count.get('n', 0)}토큰, 총 {elapsed:.1f}초 (약 {count.get('n', 0) / elapsed:.1f} 토큰/초) / "
          f"첫 출력 {first_chunk:.1f}초 / 답변 시작 {answer_start and f'{answer_start:.1f}초'}")
    print(f"[추론 과정] {thought.metadata} ({len(thought.content)}자)\n{thought.content[:300]}...\n")
    print(f"[답변]\n{answer or '(없음)'}\n")

    checks = {
        "추론 블록이 완료 상태": thought.metadata.get("status") == "done",
        "답변이 생성됨": bool(answer),
        "답변에 </think> 태그가 섞이지 않음": "</think>" not in answer,
    }
    if args.backend == "transformers":
        checks["모델이 MPS(GPU)에 올라감"] = "mps" in app.backend.label
    for name, ok in checks.items():
        print(f"{'✅' if ok else '❌'} {name}")
    if "최대 생성 길이" in answer:
        print("ℹ️  한도에 걸려 답변이 잘렸습니다. --max-new-tokens를 늘려 다시 시도해 보세요.")
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
