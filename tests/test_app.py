"""app.py 단위 테스트 — 실제 모델 없이 가짜 모델/토크나이저로 몇 초 안에 실행됩니다.

실행: uv run python -m unittest discover -s tests -v
"""

import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import gradio as gr
import torch
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import app  # noqa: E402  (모델은 main()에서만 로드하므로 import만으로는 아무것도 불러오지 않음)


def text_msg(role, text, thought=False):
    """Gradio 6가 history로 넘기는 형태의 메시지."""
    return {
        "role": role,
        "content": [{"type": "text", "text": text}],
        "metadata": {"title": "추론 과정", "status": "done"} if thought else None,
    }


def describe(messages):
    """[ChatMessage, ...] → [("thought"|"answer", status, content), ...]"""
    out = []
    for m in messages:
        meta = m.metadata or {}
        out.append(("thought" if meta.get("title") else "answer", meta.get("status"), m.content))
    return out


# ---------------------------------------------------------------------------
# 1. 화면 동작 (백엔드와 무관한 medical_chat 로직)
# ---------------------------------------------------------------------------
class FakeBackend:
    """미리 정한 텍스트 조각을 내보내는 가짜 백엔드."""

    label = "fake"

    def __init__(self, chunks=(), error=None, delay=0.0):
        self.chunks = list(chunks)
        self.error = error
        self.delay = delay
        self.emitted = 0
        self.last_messages = None

    def generate(self, messages, max_new_tokens, emit, should_stop):
        self.last_messages = messages
        if self.error:
            raise self.error
        for chunk in self.chunks[:max_new_tokens]:
            emit(chunk)
            self.emitted += 1
            time.sleep(self.delay)
            if should_stop():
                break
        return self.emitted


class MedicalChatTest(unittest.TestCase):
    def run_chat(self, backend, message="질문", history=()):
        with mock.patch.object(app, "backend", backend):
            return list(app.medical_chat(message, list(history)))

    def test_to_text(self):
        self.assertEqual(app.to_text("plain"), "plain")
        content = [{"type": "text", "text": "가"}, {"type": "file", "file": {}}, {"type": "text", "text": "나"}]
        self.assertEqual(app.to_text(content), "가나")

    def test_splits_thinking_and_answer(self):
        final = describe(self.run_chat(FakeBackend(["생각 ", "생각", "</think>", "답변 ", "답변"]))[-1])
        self.assertEqual(final[0][:2], ("thought", "done"))
        self.assertEqual(final[0][2], "생각 생각")
        self.assertEqual(final[1], ("answer", None, "답변 답변"))

    def test_thinking_is_pending_until_think_end(self):
        first = describe(self.run_chat(FakeBackend(["생각", "</think>", "답변"]))[0])
        self.assertEqual(first, [("thought", "pending", "생각")])

    def test_observation_token_removed_from_answer(self):
        final = describe(self.run_chat(FakeBackend(["생각", "</think>", "답변", "<|observation|>"]))[-1])
        self.assertEqual(final[1][2], "답변")

    def test_history_excludes_thoughts_and_converts_content(self):
        backend = FakeBackend(["생각", "</think>", "답변"])
        history = [
            text_msg("user", "안녕하세요"),
            text_msg("assistant", "인사에 답한다", thought=True),
            text_msg("assistant", "안녕하세요!"),
        ]
        self.run_chat(backend, "새 질문", history)
        self.assertEqual(
            backend.last_messages,
            [
                {"role": "system", "content": app.SYSTEM_PROMPT},
                {"role": "user", "content": "안녕하세요"},
                {"role": "assistant", "content": "안녕하세요!"},
                {"role": "user", "content": "새 질문"},
            ],
        )

    def test_history_trimmed_to_max_turns(self):
        backend = FakeBackend(["생각", "</think>", "답변"])
        history = []
        for i in range(app.MAX_HISTORY_TURNS + 5):
            history += [text_msg("user", f"q{i}"), text_msg("assistant", f"a{i}")]
        self.run_chat(backend, "새 질문", history)
        sent = backend.last_messages
        self.assertEqual(len(sent), 1 + app.MAX_HISTORY_TURNS * 2 + 1)  # system + 최근 N턴 + 새 질문
        self.assertEqual(sent[1]["content"], "q5")  # 앞의 5턴은 잘림

    def test_truncation_notice_when_max_tokens_reached(self):
        backend = FakeBackend(["생각", "</think>"] + ["답변 "] * 100)
        with mock.patch.object(app, "MAX_NEW_TOKENS", 20):
            final = describe(self.run_chat(backend)[-1])
        self.assertEqual(backend.emitted, 20)
        self.assertIn("최대 생성 길이", final[-1][2])

    def test_no_truncation_notice_on_normal_finish(self):
        final = describe(self.run_chat(FakeBackend(["생각", "</think>", "답변"]))[-1])
        self.assertNotIn("최대 생성 길이", final[-1][2])

    def test_closing_generator_stops_generation(self):
        """Gradio 중지 버튼 = generator.close() → 백그라운드 생성도 멈춰야 함."""
        backend = FakeBackend(["생각", "</think>"] + ["답변 "] * 1000, delay=0.005)
        with mock.patch.object(app, "backend", backend):
            chat = app.medical_chat("질문", [])
            for _ in range(5):
                next(chat)
            chat.close()
            time.sleep(0.2)
        self.assertLess(backend.emitted, 100)

    def test_generation_error_raises_gradio_error(self):
        with self.assertRaises(gr.Error) as ctx:
            self.run_chat(FakeBackend(error=RuntimeError("boom")))
        self.assertIn("boom", str(ctx.exception))


# ---------------------------------------------------------------------------
# 2. MLX 백엔드
# ---------------------------------------------------------------------------
class FakeMLXTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(kwargs)
        return [1, 2, 3]


class MLXBackendTest(unittest.TestCase):
    def make_backend(self, responses):
        self.tmp = tempfile.TemporaryDirectory()
        model_dir = Path(self.tmp.name) / "L1-30B-A5B-mlx-8bit"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")
        self.tokenizer = FakeMLXTokenizer()
        self.stream_kwargs = {}
        self.yielded = 0
        self.closed = False

        def fake_stream_generate(model, tokenizer, prompt, **kwargs):
            self.stream_kwargs = dict(kwargs, prompt=prompt)
            try:
                for i, text in enumerate(responses, 1):
                    self.yielded = i
                    yield SimpleNamespace(text=text, generation_tokens=i)
            finally:
                self.closed = True

        with mock.patch("mlx_lm.load", return_value=("model", self.tokenizer)), \
                mock.patch("mlx_lm.stream_generate", fake_stream_generate):
            backend = app.MLXBackend(model_dir)
        self.addCleanup(self.tmp.cleanup)
        return backend

    def test_label_shows_model_dir(self):
        self.assertEqual(self.make_backend([]).label, "MLX (L1-30B-A5B-mlx-8bit)")

    def test_generate_emits_text_and_returns_token_count(self):
        backend = self.make_backend(["생각", "</think>", "답변"])
        emitted = []
        n = backend.generate([{"role": "user", "content": "q"}], 100, emitted.append, lambda: False)
        self.assertEqual(emitted, ["생각", "</think>", "답변"])
        self.assertEqual(n, 3)
        self.assertEqual(self.stream_kwargs["max_tokens"], 100)
        self.assertEqual(self.stream_kwargs["prompt"], [1, 2, 3])
        # 토큰 ID 리스트로 받도록 요청했는지 (transformers 5는 기본이 dict 반환)
        self.assertEqual(self.tokenizer.calls[0]["return_dict"], False)
        self.assertTrue(self.tokenizer.calls[0]["add_generation_prompt"])

    def test_generate_stops_and_closes_stream(self):
        backend = self.make_backend(["답변 "] * 100)
        emitted = []
        backend.generate([], 100, emitted.append, lambda: len(emitted) >= 3)
        self.assertEqual(len(emitted), 3)
        self.assertEqual(self.yielded, 3)
        self.assertTrue(self.closed)

    def test_missing_model_dir_explains_how_to_convert(self):
        with self.assertRaises(SystemExit) as ctx:
            app.MLXBackend(Path(tempfile.gettempdir()) / "no-such-mlx-model")
        self.assertIn("convert_mlx.py", str(ctx.exception))
        self.assertIn("--backend transformers", str(ctx.exception))


# ---------------------------------------------------------------------------
# 3. transformers 백엔드
# ---------------------------------------------------------------------------
VOCAB = {10: "생각 ", 11: "</think>", 12: "답변 "}


class FakeBatch(dict):
    def to(self, device):
        return self


class FakeHFTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return FakeBatch(input_ids=torch.tensor([[1, 2, 3]]))

    def decode(self, ids, **kwargs):
        return "".join(VOCAB.get(int(i), "") for i in ids)


class FakeHFModel:
    """streamer와 stopping_criteria를 실제 generate처럼 다루는 가짜 모델."""

    device = "cpu"

    def __init__(self, plan=()):
        self.plan = list(plan)
        self.generated = 0

    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self

    def generate(self, input_ids, streamer, max_new_tokens, stopping_criteria, **kwargs):
        streamer.put(input_ids)  # 프롬프트 (skip_prompt=True라 출력되지 않아야 함)
        ids = input_ids
        for token in self.plan[:max_new_tokens]:
            ids = torch.cat([ids, torch.tensor([[token]])], dim=1)
            streamer.put(torch.tensor([token]))
            self.generated += 1
            if stopping_criteria(ids, None).all():
                break
        streamer.end()
        return ids


class TransformersBackendTest(unittest.TestCase):
    def make_backend(self, plan=(), mps=True):
        self.model = FakeHFModel(plan)
        with mock.patch.object(transformers.AutoTokenizer, "from_pretrained", return_value=FakeHFTokenizer()), \
                mock.patch.object(transformers.AutoModelForCausalLM, "from_pretrained",
                                  return_value=self.model) as load_model, \
                mock.patch.object(torch.backends.mps, "is_available", return_value=mps):
            backend = app.TransformersBackend()
        self.load_kwargs = load_model.call_args.kwargs
        return backend

    def test_mps_uses_eager_attention(self):
        """Mac GPU + bf16 + sdpa 조합은 이 모델을 잘못 계산하므로 eager여야 함 (회귀 방지)."""
        self.make_backend(mps=True)
        self.assertEqual(self.load_kwargs["attn_implementation"], "eager")
        self.assertNotIn("device_map", self.load_kwargs)  # CPU에 먼저 로딩 후 .to("mps") (OOM 방지)

    def test_generate_streams_text_and_counts_tokens(self):
        backend = self.make_backend([10, 10, 11, 12, 12])
        emitted = []
        n = backend.generate([], 100, emitted.append, lambda: False)
        self.assertEqual("".join(emitted), "생각 생각 </think>답변 답변 ")
        self.assertEqual(n, 5)

    def test_generate_stops_when_requested(self):
        backend = self.make_backend([12] * 100)
        emitted = []
        backend.generate([], 100, emitted.append, lambda: self.model.generated >= 3)
        self.assertEqual(self.model.generated, 3)


class LoadBackendTest(unittest.TestCase):
    def test_unknown_backend(self):
        with self.assertRaises(ValueError):
            app.load_backend("nope")


class MainTest(unittest.TestCase):
    """실행 옵션이 백엔드와 생성 한도에 반영되는지 (UI는 띄우지 않음)."""

    def run_main(self, argv):
        with mock.patch.object(sys, "argv", ["app.py", *argv]), \
                mock.patch.object(app, "load_backend", return_value=FakeBackend()) as load, \
                mock.patch.object(app.gr, "ChatInterface") as ui, \
                mock.patch.object(app, "MAX_NEW_TOKENS", app.MAX_NEW_TOKENS), \
                mock.patch.object(app, "backend", None):
            app.main()
            return load.call_args.args, app.MAX_NEW_TOKENS, ui.call_args.kwargs

    def test_defaults(self):
        load_args, max_tokens, ui_kwargs = self.run_main([])
        self.assertEqual(load_args, ("mlx", app.MLX_MODEL_DIR))
        self.assertEqual(max_tokens, 32768)
        self.assertIn("32,768", ui_kwargs["description"])

    def test_options(self):
        load_args, max_tokens, _ = self.run_main(["--backend", "transformers", "--max-new-tokens", "1000"])
        self.assertEqual(load_args[0], "transformers")
        self.assertEqual(max_tokens, 1000)


if __name__ == "__main__":
    unittest.main()
