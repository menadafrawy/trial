# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Scriptify — Arabic handwriting synthesis. An LSTM + soft-attention model (Graves 2013) outputs GMM stroke distributions; the API serves inference; the React frontend renders animated strokes.

## Commands

### Backend (FastAPI)
```bash
cd api/src
python main.py                      # runs uvicorn on http://localhost:8000
# or: uvicorn main:app --reload
```
The API **must** be launched from `api/src/` because `MODEL_DIR` is resolved as `../../ml/packaged_models` relative to that directory.

### Frontend (React + Vite)
```bash
cd frontend
npm install
npm run dev          # http://localhost:5173
npm run type-check   # TypeScript check (no emit)
npm run lint         # ESLint with auto-fix
npm run build        # production build
```

For local development the frontend needs `frontend/.env.local`:
```
VITE_API_BASE_URL=http://localhost:8000
```
Without this, `generateHandwriting()` targets the HuggingFace Space URL instead of the local backend. Note: feedback endpoints (`generateCandidates`, `saveFeedback`, etc.) always hardcode `http://127.0.0.1:8000` regardless of the env var.

## Architecture

### Request flow
```
User types text
  → App.tsx (bias/settings state)
  → useHandwritingAPI hook (loading/error state)
  → services/api.ts generateHandwriting()
  → POST /generate  (FastAPI)
  → text normalised (normalise_arabic)
  → NUM_SAMPLES=10 candidates generated in parallel via scripted model
  → best candidate scored by _candidate_score() / ARABIC_EXPECTED_SEGMENTS
  → trim_stroke_noise → filter_spatially_distant_strokes
  → absolute coords returned
  → HandwritingCanvas animates strokes
```

### Model loading (startup)
`main.py` loads two files from `ml/packaged_models/`:
- `handwriting_model.scripted.pt` — TorchScript model used for inference (`scripted_model.sample(...)`)
- `handwriting_model.pt` — metadata dict containing `config_full` (alphabet, hyperparams). This is the source of truth for `alphabet_map`, `max_text_len`, `output_mixture_components`, `lstm_size`, `attention_mixture_components`.

If either file is missing the API raises `FileNotFoundError` at startup — check `/health` to confirm.

A fine-tuned model trained on 188 accepted human feedback samples exists at `ml/packaged_models/handwriting_model_finetuned_188samples.pt`. To serve it, rename/copy it to `handwriting_model.pt` and its scripted counterpart to `handwriting_model.scripted.pt`, then restart the API.

### Arabic text handling
`normalise_arabic()` in `inference_utils.py` maps character variants (أ إ آ → ا, ى → ي, ة → ه) before encoding, because these variants are absent from the training alphabet. `ARABIC_EXPECTED_SEGMENTS` in `main.py` maps isolated letters to expected pen-down segment counts, used to select the best candidate from the batch.

### Feedback / fine-tuning loop
- `POST /feedback/candidates` — generates N candidate strokes for human review (used by `FeedbackPanel` in the frontend)
- `POST /feedback/save` — stores accepted/rejected sample to `ml/data/feedback/feedback_data.json` via `FeedbackStore`
- `POST /feedback/fine-tune` — triggers background fine-tuning on accepted samples (min 5 required); calls `_reload_scripted_model()` on completion to hot-swap the model without restarting
- `GET /feedback/stats` — returns per-character accepted counts and fine-tune status
- Kaggle fine-tuning workflow lives in `ml/kaggle/fine_tune_kaggle.py`

### Packaging a new model
After training:
```bash
cd ml
python -m src.package_model --pkg_name handwriting_model
# produces ml/packaged_models/handwriting_model.pt and handwriting_model.scripted.pt
```

## Known issues

- The `Slider` component's `inverted` prop reverses the visual direction for animation speed (higher slider value = slower speed internally, `inverted=true` flips the display label).
- HuggingFace Spaces auto-sleep; cold starts take 30–60 seconds, which may hit the 180 s client timeout.
