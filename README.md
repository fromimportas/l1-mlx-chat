# L1-30B-A5B 로컬 채팅 UI

Hugging Face의 의료 특화 언어 모델 [`learning-unit/L1-30B-A5B`](https://huggingface.co/learning-unit/L1-30B-A5B)를 Mac에서 직접 실행하고, Gradio 채팅 화면으로 질문할 수 있게 해 주는 프로젝트입니다.

- 모델의 **추론 과정**과 **최종 답변**을 나눠서 보여 줍니다. 추론 과정은 접을 수 있는 블록에 표시됩니다.
- 답변은 생성되는 대로 실시간으로 표시됩니다.
- **중지** 버튼을 누르면 모델 생성도 바로 멈춥니다.
- 실행 방식(백엔드)을 두 가지 중에서 고를 수 있습니다. 기본은 빠른 **MLX**입니다.

> ⚠️ 연구·평가용입니다. 모델 답변을 실제 진료 판단에 그대로 쓰면 안 됩니다.

> 이 프로젝트는 모델 개발사(Lunit, Trillion Labs 및 컨소시엄)와 무관한 **비공식** 프로젝트입니다. 모델은 [Apache 2.0](https://huggingface.co/learning-unit/L1-30B-A5B/blob/main/LICENSE)으로 공개되어 있으며, 이 저장소에는 모델 가중치가 포함되어 있지 않습니다. 코드의 시스템 프롬프트("…developed by Lunit.")는 [모델 카드](https://huggingface.co/learning-unit/L1-30B-A5B)가 필수 설정으로 안내하는 문장을 그대로 쓴 것입니다.

## 백엔드 비교

| | **MLX (기본)** | transformers |
|---|---|---|
| 모델 | 원본을 8비트로 변환한 모델 | 원본 모델 그대로 (bf16) |
| 생성 속도 (M5 Max) | **초당 약 60토큰** | 초당 약 2.7토큰 |
| 메모리 | 약 32GB | 약 59GB |
| 정확도 (원본을 CPU에서 실행한 결과와 다음 토큰 예측 일치율) | 96.3% | 97.7% |
| 준비 | 변환 필요 (한 번, 수 분) | 없음 |

두 백엔드 모두 정확도는 정상 범위입니다. 100%가 아닌 것은 각각 8비트 양자화와 bf16 정밀도로 생기는 작은 오차 때문입니다. 특별한 이유가 없으면 MLX를 쓰세요. transformers는 변환 결과가 의심스러울 때 원본과 비교하는 용도로 남겨 두었습니다.

## 요구 사항

| 항목 | 내용 |
|---|---|
| 기기 | Apple Silicon Mac. **메모리 128GB(M5 Max)에서 확인했습니다.** MLX 백엔드는 약 32GB를 써서 64GB Mac에서도 돌아갈 것으로 보이지만 확인하지는 않았습니다. transformers 백엔드는 약 59GB를 씁니다 |
| 디스크 | 약 90GB. 원본 모델 약 55GB(`~/.cache/huggingface/hub`)와 MLX 변환 모델 약 29GB(`models/`)입니다 |
| Python | **3.12.** 시스템 Python 3.15에는 torch 휠이 없을 수 있어서 3.12를 씁니다 |
| 도구 | [uv](https://docs.astral.sh/uv/) (`brew install uv`) |

## 설치

uv 프로젝트로 구성되어 있습니다. 아래 명령 하나로 Python 3.12 준비, 가상환경 `.venv` 생성, 패키지 설치가 모두 끝납니다.

```bash
uv sync
```

- 의존성은 [pyproject.toml](pyproject.toml)에, 확인한 정확한 버전은 [uv.lock](uv.lock)에 기록되어 있습니다. `uv sync`는 lock 파일의 버전 그대로 설치합니다.
- Python 버전은 [.python-version](.python-version)(3.12)을 따릅니다. 없으면 uv가 자동으로 내려받습니다.
- 주요 패키지는 gradio 6.31, mlx-lm 0.32, torch 2.15, transformers 5.22입니다. `mlx-lm`은 Apple Silicon Mac에서만 설치됩니다.

**패키지 추가와 업데이트**

```bash
uv add <패키지>          # pyproject.toml과 uv.lock에 추가하고 설치
uv lock --upgrade        # 의존성을 최신 버전으로 갱신 (갱신 후 테스트 필요)
```

`.venv`에는 `pip`이 들어 있지 않으므로 `pip install`은 쓰지 마세요.

### MLX 모델 변환 (처음 한 번)

```bash
uv run scripts/convert_mlx.py            # 8비트 → models/L1-30B-A5B-mlx-8bit
```

- 원본 모델이 캐시에 없으면 먼저 내려받습니다(약 51GB). 받는 데 수십 분에서 몇 시간이 걸릴 수 있습니다.
- 변환 자체는 몇 분이면 끝납니다. 원본 파일이 메모리 캐시에 남아 있으면 몇 초 만에 끝나기도 합니다.
- `--bits 4`로 4비트 모델(약 16GB)도 만들 수 있습니다. 더 빠르지만 정확도는 검증하지 않았습니다.

**변환 원리:** 이 모델의 종류(`gravity_moe`)는 MLX가 직접 지원하지 않습니다. 하지만 모델 코드가 transformers의 DeepSeek V3 구현을 그대로 상속하고 가중치 이름도 같아서, 설정 파일의 모델 종류만 `deepseek_v3`로 바꿔 MLX의 DeepSeek V3 구현으로 변환합니다. 원본 파일은 복사하지 않고 링크로만 씁니다.

## 실행

```bash
uv run app.py                          # MLX (기본)
uv run app.py --backend transformers   # 원본 모델 + transformers
uv run app.py --mlx-model models/L1-30B-A5B-mlx-4bit   # 다른 MLX 모델
uv run app.py --max-new-tokens 65536   # 최대 생성 길이 변경 (기본 32,768)
```

- 로딩이 끝나면 브라우저에서 http://127.0.0.1:7860 에 접속합니다. 화면 위쪽에 사용 중인 백엔드가 표시됩니다.
- 로딩 시간: MLX는 몇 초, transformers는 약 30초입니다.
- MLX 모델이 없으면 변환 방법을 안내하고 종료합니다.
- `uv run`은 `.venv`의 환경으로 실행하고, 실행 전에 패키지가 lock 파일과 맞는지도 확인합니다.

`HF_TOKEN`이 없다는 경고가 나와도 실행에는 문제가 없습니다. 토큰을 설정하면 다운로드가 더 빨라질 수 있습니다.

## 설정

[app.py](app.py) 위쪽에 있는 상수로 조정합니다.

| 상수 | 기본값 | 설명 |
|---|---|---|
| `MAX_NEW_TOKENS` | `32768` | 한 번에 생성할 최대 토큰 수입니다. **추론에 쓴 토큰도 포함됩니다.** 모델 카드 예시와 같은 값이고, 실행할 때 `--max-new-tokens`로 바꿀 수 있습니다. 이 값에 도달하면 답변 끝에 "⚠️ 최대 생성 길이…" 안내가 붙습니다. 한도까지 생성하면 MLX로 약 9분, transformers로는 몇 시간이 걸릴 수 있습니다 |
| `MAX_HISTORY_TURNS` | `10` | 모델에 넘길 최근 대화 턴 수입니다. 이전 턴의 추론 과정은 넘기지 않습니다 |
| `TEMPERATURE` / `REPETITION_PENALTY` | `0.7` / `1.05` | 샘플링 설정입니다. 두 백엔드에 똑같이 적용됩니다 |
| `MLX_MODEL_DIR` | `models/L1-30B-A5B-mlx-8bit` | MLX 백엔드의 기본 모델 폴더입니다 |
| `MODEL_REVISION` | `None` | 원본 모델 저장소의 커밋 해시입니다. `None`이면 항상 최신 버전을 받습니다 (아래 보안 참고) |
| `SYSTEM_PROMPT` | Chain-of-Evidence… | 시스템 프롬프트입니다 |

## 테스트

테스트는 세 단계로 나뉩니다. 코드를 고친 뒤에는 **1번은 항상**, 나머지는 필요할 때 실행하세요.

### 1. 단위 테스트 (모델 불필요, 몇 초)

가짜 모델과 토크나이저로 `app.py`의 로직만 확인합니다. 모델을 내려받거나 메모리에 올리지 않습니다.

```bash
uv run python -m unittest discover -s tests -v
```

[tests/test_app.py](tests/test_app.py)에는 테스트 20개가 있습니다.

**화면 동작** (`MedicalChatTest`, 백엔드와 무관)

| 테스트 | 확인 내용 |
|---|---|
| `test_splits_thinking_and_answer` | `</think>`를 기준으로 추론 과정과 답변이 나뉘는지 |
| `test_thinking_is_pending_until_think_end` | 추론이 진행되는 동안 블록이 "진행 중" 상태로 표시되는지 |
| `test_observation_token_removed_from_answer` | 종료 토큰 `<|observation|>`이 답변에서 지워지는지 |
| `test_history_excludes_thoughts_and_converts_content` | Gradio 6 형식의 대화 기록을 모델 입력으로 바르게 바꾸고, 추론 메시지는 빼는지 |
| `test_history_trimmed_to_max_turns` | 대화 기록이 최근 `MAX_HISTORY_TURNS` 턴으로 잘리는지 |
| `test_truncation_notice_when_max_tokens_reached` | 생성 한도에 도달하면 안내 문구가 붙는지 |
| `test_no_truncation_notice_on_normal_finish` | 정상적으로 끝났을 때는 안내 문구가 붙지 않는지 |
| `test_closing_generator_stops_generation` | 중지 버튼을 누르면 백그라운드 생성도 멈추는지 |
| `test_generation_error_raises_gradio_error` | 생성 중 에러가 나면 화면이 멈추지 않고 에러를 보여 주는지 |
| `test_to_text` | Gradio 메시지 content에서 텍스트만 꺼내는지 |

**MLX 백엔드** (`MLXBackendTest`)

| 테스트 | 확인 내용 |
|---|---|
| `test_generate_emits_text_and_returns_token_count` | 텍스트 조각을 전달하고 생성 토큰 수를 반환하는지, 토큰 ID 목록으로 프롬프트를 만드는지 |
| `test_generate_stops_and_closes_stream` | 중지 요청 시 생성을 멈추고 스트림을 닫는지 |
| `test_missing_model_dir_explains_how_to_convert` | 변환 모델이 없을 때 변환 방법을 안내하는지 |
| `test_label_shows_model_dir` | 화면에 표시할 백엔드 이름 |

**transformers 백엔드** (`TransformersBackendTest`)

| 테스트 | 확인 내용 |
|---|---|
| `test_mps_uses_eager_attention` | Mac GPU에서 `eager` 어텐션을 쓰고, CPU에 먼저 로딩하는지 (아래 문제 해결의 두 버그가 되살아나지 않도록 막음) |
| `test_generate_streams_text_and_counts_tokens` | 텍스트 조각을 전달하고 생성 토큰 수를 반환하는지 |
| `test_generate_stops_when_requested` | 중지 요청 시 생성을 멈추는지 |

`LoadBackendTest`는 알 수 없는 백엔드 이름을 거부하는지, `MainTest`는 실행 옵션(`--backend`, `--max-new-tokens`)이 제대로 반영되는지 확인합니다.

### 2. 실제 모델 스모크 테스트

실제 모델을 올려 짧은 질문 하나에 답하게 하고, 결과와 속도를 출력합니다.

> ⚠️ **실행하기 전에 `app.py`를 꺼 주세요.** 모델이 두 개 올라가면 메모리가 부족할 수 있습니다.

```bash
uv run tests/smoke_test.py                          # MLX (약 10초)
uv run tests/smoke_test.py --backend transformers   # transformers (약 3~5분)
# 질문이나 생성 한도를 바꿀 때
uv run tests/smoke_test.py --prompt "후기 감속이 무엇인가요?" --max-new-tokens 1500
```

아래 항목을 점검하고, 하나라도 실패하면 종료 코드 1로 끝납니다.

- 추론 블록이 완료 상태인지
- 답변이 생성됐는지
- 답변에 `</think>` 태그가 섞이지 않았는지
- (transformers만) 모델이 MPS(GPU)에 올라갔는지

생성 한도가 작아서 답변 전에 추론이 끝나지 않으면 "답변이 생성됨"이 실패할 수 있습니다. 이때는 `--max-new-tokens`를 늘려서 다시 실행하세요.

### 3. 화면 동작 확인 (직접)

`uv run app.py`로 띄운 뒤 브라우저에서 다음을 확인합니다.

- [ ] 화면 위쪽에 백엔드 이름이 표시된다
- [ ] 질문하면 "추론 과정" 블록에 진행 표시가 돌다가, 추론이 끝나면 블록이 접히고 걸린 시간이 표시된다
- [ ] 답변이 블록 아래에 실시간으로 표시된다
- [ ] 연달아 질문할 때 앞 대화를 이해한다 (예: "후기 감속이 뭐예요?" → "그게 반복되면 어떻게 하나요?")
- [ ] 생성 중에 중지 버튼을 누르면 생성이 멈추고, 다음 질문이 바로 시작된다

## 문제 해결

| 증상 | 원인과 해결 |
|---|---|
| (transformers) 로딩 중 `MPS ... Insufficient Memory` (OOM) | `device_map="auto"`로 GPU에 바로 올리면, 전문가 가중치를 합치는 동안 메모리가 모델 크기의 2배(약 124GB)까지 필요해집니다. 그래서 `app.py`는 Mac에서 **CPU에 먼저 로딩한 뒤 `.to("mps")`로 옮깁니다.** 그래도 OOM이 나면 다른 모델 프로세스(실행 중인 `app.py`, LM Studio 등)가 떠 있는지 확인하세요 |
| (transformers) 답변 품질이 이상하다 | **Mac GPU + bf16 + PyTorch `sdpa` 어텐션** 조합은 이 모델을 잘못 계산합니다. 이 모델은 어텐션의 쿼리·키 크기(192)와 값 크기(128)가 다릅니다. 원본을 CPU에서 실행한 결과와 다음 토큰 예측이 79%만 일치했습니다. 그래서 `app.py`는 `attn_implementation="eager"`를 씁니다(일치율 97.7%). **이 설정을 지우지 마세요** |
| MLX 모델이 없다고 나온다 | `uv run scripts/convert_mlx.py`로 먼저 변환하세요 |
| 답변이 중간에 끊긴다 | 답변 끝에 "⚠️ 최대 생성 길이…" 안내가 있으면 한도(추론 포함)에 도달한 것입니다. `--max-new-tokens`로 한도를 늘리거나 질문 범위를 좁히세요. 안내 없이 끊겼다면 다른 원인이니 터미널 로그를 확인하세요 |
| 너무 느리다 | transformers 백엔드를 쓰고 있는지 확인하세요. Mac에서는 초당 약 2.7토큰입니다. MLX 백엔드(기본)는 초당 약 60토큰입니다 |
| `import gradio`가 이상하게 동작한다 | 프로젝트 폴더에 `gradio.py`라는 파일을 만들면, 설치된 gradio 패키지 대신 그 파일이 import됩니다. 그런 이름의 파일을 만들지 마세요 |
| "A new version of ... was downloaded" 경고 | 아래 보안 항목을 참고하세요 |

## 보안: `trust_remote_code`

transformers 백엔드는 `trust_remote_code=True`가 필요합니다. 즉, 모델 저장소의 Python 파일(`configuration_gravity_moe.py`, `modeling_gravity_moe.py`)이 내 컴퓨터에서 실행됩니다. 2026-11-06에 받은 버전(커밋 `f30fdbc73fbd3dec1f4dfb59b6f5ad495d6b2c4a`)은 transformers의 DeepSeek V3 구현을 상속만 하는 짧은 코드였고, 문제 될 내용은 없었습니다. MLX 백엔드와 변환 스크립트는 이 Python 파일을 실행하지 않습니다.

저장소가 나중에 바뀌어도 검토한 버전만 쓰려면 `app.py`에서 다음과 같이 고정하세요. 변환 스크립트도 이 값을 따릅니다.

```python
MODEL_REVISION = "f30fdbc73fbd3dec1f4dfb59b6f5ad495d6b2c4a"
```

## 파일 구조

```
.
├── app.py                  # Gradio 채팅 UI + 백엔드(MLX, transformers)
├── scripts/
│   └── convert_mlx.py      # 원본 모델 → MLX 변환
├── models/                 # 변환한 MLX 모델 (git 제외)
├── pyproject.toml          # 프로젝트 정보와 의존성
├── uv.lock                 # 고정된 패키지 버전 (uv sync가 사용)
├── .python-version         # Python 버전 (3.12)
├── .gitignore
├── LICENSE
├── tests/
│   ├── test_app.py         # 단위 테스트 (가짜 모델)
│   └── smoke_test.py       # 실제 모델 스모크 테스트
└── README.md
```

## 라이선스

이 저장소의 코드는 [MIT 라이선스](LICENSE)를 따릅니다. 모델 가중치는 포함되어 있지 않으며, 모델은 [Apache 2.0](https://huggingface.co/learning-unit/L1-30B-A5B/blob/main/LICENSE)으로 배포됩니다.
