"""L1-30B-A5B Gradio 채팅 UI.

실행:
    uv run app.py                          # MLX 백엔드 (기본, 빠름)
    uv run app.py --backend transformers   # 원본 모델 + transformers (느림, 변환 불필요)
"""

import argparse
import queue
import time
from pathlib import Path
from threading import Event, Thread

import gradio as gr

MODEL_NAME = "learning-unit/L1-30B-A5B"
# trust_remote_code=True는 저장소의 Python 코드를 로컬에서 실행합니다.
# 저장소 내용을 확인한 뒤 커밋 해시로 고정하는 것을 권장합니다. (None이면 최신 버전 사용)
MODEL_REVISION = None
# scripts/convert_mlx.py로 변환한 MLX 모델 위치
MLX_MODEL_DIR = Path(__file__).resolve().parent / "models" / "L1-30B-A5B-mlx-8bit"

SYSTEM_PROMPT = "You are Chain-of-Evidence, a medical AI assistant developed by Lunit."
MAX_HISTORY_TURNS = 10  # 최근 10턴(질문+답변)만 유지
THINK_END = "</think>"
# 추론 토큰도 포함되는 생성 한도. 모델 카드 예시(max_tokens=32768)에 맞춤. 실행 시 --max-new-tokens로 변경 가능.
# 한도까지 생성하면 MLX(초당 약 60토큰)로 약 9분, transformers(초당 약 2.7토큰)로 몇 시간이 걸릴 수 있음.
# 이 모델의 MLA 어텐션은 토큰당 KV 캐시가 작아서, 32K 토큰이어도 추가 메모리는 약 2GB 수준.
MAX_NEW_TOKENS = 32768
TEMPERATURE = 0.7
REPETITION_PENALTY = 1.05


class MLXBackend:
    """Apple MLX로 변환한 모델. transformers보다 약 25배 빠르고 메모리도 절반입니다."""

    def __init__(self, model_dir):
        model_dir = Path(model_dir)
        if not (model_dir / "config.json").exists():
            raise SystemExit(
                f"MLX 모델이 없습니다: {model_dir}\n"
                "먼저 변환하세요:  uv run scripts/convert_mlx.py\n"
                "또는 변환 없이 실행:  uv run app.py --backend transformers"
            )
        from mlx_lm import load, stream_generate
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        self.model, self.tokenizer = load(str(model_dir))
        self.stream_generate = stream_generate
        self.make_sampler = make_sampler
        self.make_logits_processors = make_logits_processors
        self.label = f"MLX ({model_dir.name})"

    def generate(self, messages, max_new_tokens, emit, should_stop):
        """토큰을 생성하며 텍스트 조각마다 emit을 호출하고, 생성한 토큰 수를 반환."""
        ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=False)
        n_generated = 0
        stream = self.stream_generate(
            self.model, self.tokenizer, ids,
            max_tokens=max_new_tokens,
            sampler=self.make_sampler(temp=TEMPERATURE),
            logits_processors=self.make_logits_processors(repetition_penalty=REPETITION_PENALTY),
        )
        try:
            for response in stream:
                emit(response.text)
                n_generated = response.generation_tokens
                if should_stop():
                    break
        finally:
            stream.close()
        return n_generated


class TransformersBackend:
    """원본 모델을 transformers + PyTorch로 실행. 변환이 필요 없지만 느립니다 (Mac 기준 초당 약 2.5토큰)."""

    def __init__(self):
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            StoppingCriteria,
            StoppingCriteriaList,
            TextStreamer,
        )

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, revision=MODEL_REVISION, trust_remote_code=True
        )
        if torch.backends.mps.is_available():
            # 1) device_map="auto"로 바로 GPU에 올리면 전문가(expert) 가중치를 합치는 변환 중에 원본과 결과가
            #    동시에 GPU에 올라가 모델 크기의 2배(약 120GB)가 필요해 OOM이 납니다.
            #    그래서 CPU에서 로딩·변환을 끝낸 뒤 GPU로 옮깁니다. (GPU 사용량은 약 59GB)
            # 2) Mac GPU + bf16에서 PyTorch의 sdpa 어텐션은 이 모델(쿼리·키 192차원, 값 128차원)을 잘못 계산합니다.
            #    원본을 CPU에서 실행한 결과와 다음 토큰 예측이 79%만 일치했고, eager로 바꾸면 정확해집니다.
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                revision=MODEL_REVISION,
                dtype=torch.bfloat16,
                attn_implementation="eager",
                trust_remote_code=True,
            ).to("mps")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                revision=MODEL_REVISION,
                dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
        self.model.eval()
        self.label = f"transformers ({self.model.device})"

        class CallbackStreamer(TextStreamer):
            """디코딩된 텍스트 조각을 콜백으로 전달 (한글 등 여러 토큰에 걸친 글자도 완성된 뒤 전달)."""

            def __init__(self, tokenizer, emit):
                super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
                self.emit = emit

            def on_finalized_text(self, text, stream_end=False):
                if text:
                    self.emit(text)

        class StopWhen(StoppingCriteria):
            def __init__(self, should_stop):
                self.should_stop = should_stop

            def __call__(self, input_ids, scores, **kwargs):
                return torch.full(
                    (input_ids.shape[0],), self.should_stop(), dtype=torch.bool, device=input_ids.device
                )

        self.CallbackStreamer = CallbackStreamer
        self.stopping = lambda should_stop: StoppingCriteriaList([StopWhen(should_stop)])

    def generate(self, messages, max_new_tokens, emit, should_stop):
        # 템플릿 적용과 토큰화를 한 번에 처리 (특수 토큰 중복 방지)
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.model.device)
        output = self.model.generate(
            **inputs,
            streamer=self.CallbackStreamer(self.tokenizer, emit),
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=TEMPERATURE,
            repetition_penalty=REPETITION_PENALTY,
            stopping_criteria=self.stopping(should_stop),
        )
        return output.shape[1] - inputs["input_ids"].shape[1]


def load_backend(name, mlx_model_dir=MLX_MODEL_DIR):
    if name == "mlx":
        return MLXBackend(mlx_model_dir)
    if name == "transformers":
        return TransformersBackend()
    raise ValueError(f"알 수 없는 백엔드: {name}")


backend = None  # main()에서 load_backend()로 설정


def to_text(content):
    """Gradio 6 메시지 content([{"type": "text", "text": ...}, ...])에서 텍스트만 추출."""
    if isinstance(content, str):
        return content
    return "".join(c.get("text", "") for c in content if c.get("type") == "text")


def is_thought(m):
    """화면에 '추론 과정'으로 표시했던 메시지인지 확인 (모델 입력에서는 제외)."""
    return bool((m.get("metadata") or {}).get("title"))


def medical_chat(message, history):
    history = [m for m in history if not is_thought(m)][-MAX_HISTORY_TURNS * 2:]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += [{"role": m["role"], "content": to_text(m["content"])} for m in history]
    messages.append({"role": "user", "content": message})

    # 생성은 별도 스레드에서 하고, 텍스트 조각을 큐로 받아 화면에 표시합니다.
    chunks = queue.Queue()
    stop_event = Event()
    result = {}

    def run_generate():
        try:
            result["n_generated"] = backend.generate(messages, MAX_NEW_TOKENS, chunks.put, stop_event.is_set)
        except Exception as e:
            result["error"] = e
        finally:
            chunks.put(None)  # 종료 신호 (에러가 나도 아래 루프가 영원히 기다리지 않도록)

    thread = Thread(target=run_generate, daemon=True)
    thread.start()

    # chat template이 프롬프트 끝에 "<think>"를 넣으므로, 모델은 추론 → "</think>" → 답변 순으로 출력합니다.
    # 추론은 접을 수 있는 블록에, 답변은 일반 메시지로 나눠 생성되는 대로 표시합니다.
    raw = ""
    start = time.time()
    think_time = None

    def render(notice=""):
        thinking, sep, answer = raw.partition(THINK_END)
        # <|observation|>은 종료 토큰이지만 일반 토큰이라 skip_special_tokens로 지워지지 않음
        answer = answer.replace("<|observation|>", "").strip()
        thought = gr.ChatMessage(
            content=thinking.strip(),
            metadata={
                "title": "추론 과정",
                "status": "done" if (sep or notice) else "pending",
                "duration": round(think_time or (time.time() - start), 1),
            },
        )
        answer = f"{answer}\n\n{notice}".strip()
        return [thought, gr.ChatMessage(content=answer)] if answer else [thought]

    try:
        while (chunk := chunks.get()) is not None:
            raw += chunk
            if think_time is None and THINK_END in raw:
                think_time = time.time() - start  # 추론이 끝난 시점에서 시간 고정
            yield render()
    finally:
        # 중지 버튼을 누르면 Gradio가 이 generator를 닫으므로, 백그라운드 생성도 함께 멈춤
        stop_event.set()

    thread.join()
    if "error" in result:
        raise gr.Error(f"생성 중 오류가 발생했습니다: {result['error']}")
    if result["n_generated"] >= MAX_NEW_TOKENS:
        yield render(
            f"⚠️ 최대 생성 길이({MAX_NEW_TOKENS:,} 토큰, 추론 포함)에 도달해 답변이 잘렸습니다. "
            "--max-new-tokens 옵션으로 한도를 늘려 다시 실행하거나 질문 범위를 좁혀 주세요."
        )


def main():
    global backend, MAX_NEW_TOKENS
    parser = argparse.ArgumentParser(description="L1-30B-A5B 채팅 UI")
    parser.add_argument(
        "--backend", choices=["mlx", "transformers"], default="mlx",
        help="mlx: 변환한 모델로 빠르게 실행 (기본) / transformers: 원본 모델로 실행 (느림)",
    )
    parser.add_argument("--mlx-model", type=Path, default=MLX_MODEL_DIR, help="MLX 모델 폴더")
    parser.add_argument(
        "--max-new-tokens", type=int, default=MAX_NEW_TOKENS,
        help=f"추론을 포함한 최대 생성 토큰 수 (기본 {MAX_NEW_TOKENS})",
    )
    args = parser.parse_args()

    MAX_NEW_TOKENS = args.max_new_tokens
    backend = load_backend(args.backend, args.mlx_model)
    print(f"백엔드: {backend.label} / 최대 생성 길이: {MAX_NEW_TOKENS:,} 토큰")
    # 실행 시 브라우저(http://127.0.0.1:7860)에 채팅 UI가 열립니다.
    gr.ChatInterface(
        medical_chat,
        title="L1-30B-A5B 채팅",
        description=f"비공식 로컬 실행 도구 · 백엔드: {backend.label} · 최대 생성 길이: {MAX_NEW_TOKENS:,} 토큰",
    ).launch()


if __name__ == "__main__":
    main()
