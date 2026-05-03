from typing import Optional
from fastapi import FastAPI, HTTPException, status 
from fastapi.middleware.cors import CORSMiddleware 
from pydantic import BaseModel, Field
import torch 
import torch.nn.functional as F 
from pathlib import Path 
import logging 
import time 
from contextlib import asynccontextmanager
from inference_utils import construct_alphabet_list, convert_offsets_to_absolute_coords, encode_text, get_alphabet_map, trim_stroke_noise, filter_spatially_distant_strokes, normalise_arabic, is_degenerate_strokes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) 

MODEL_DIR = Path("../../ml/packaged_models") 
SCRIPTED_MODEL_NAME = "handwriting_model.scripted.pt" 
METADATA_MODEL_NAME = "handwriting_model.pt" 

scripted_model: Optional[torch.jit.ScriptModule] = None 
model_metadata: Optional[dict] = None 
device: Optional[torch.device] = None 
alphabet_map: Optional[dict[str, int]] = None 
ALPHABET_LIST: Optional[list[str]] = None 
ALPHABET_SIZE: Optional[int] = None 
max_text_len: Optional[int] = None 
output_mixture_components: Optional[int] = None # To store num_mixtures for GMM sampling
lstm_size: Optional[int] = None 
attention_mixture_components: Optional[int] = None 

# Patience for early stopping in generate_strokes
PATIENCE_PEN_UP_EOS = 15
MIN_MOVEMENT_THRESHOLD = 0.02 


class HandwritingRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=40, description="Text to generate handwriting for")
    max_length: int = Field(default=1000, ge=50, le=1200, description="Maximum number of stroke points")
    bias: float = Field(default=2, ge=0.2, le=3.0, description="Sampling bias for generation")
class HandwritingResponse(BaseModel):
    success: bool = True
    input_text: str
    generation_time_ms: float
    num_points: int
    strokes: list[list[float]]
    message: str = "Successfully generated handwriting."

class HealthResponse(BaseModel):
    status: str 
    model_loaded: bool 
    device: str 
    model_metadata_keys: Optional[list[str]] = None 

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    global scripted_model, model_metadata, device, alphabet_map, max_text_len, ALPHABET_LIST, output_mixture_components, lstm_size, attention_mixture_components, ALPHABET_SIZE
    logger.info("Attempting to load model resources during startup") 
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        scripted_model_path = MODEL_DIR / SCRIPTED_MODEL_NAME
        metadata_model_path = MODEL_DIR / METADATA_MODEL_NAME
        
        if  not scripted_model_path.exists():
            logger.error(f"Traced model not found at {scripted_model_path}")
            raise FileNotFoundError(f"Traced model not found at {scripted_model_path}")
        if not metadata_model_path or not metadata_model_path.exists():
            logger.error(f"Metadata model file not found at {metadata_model_path}")
            raise FileNotFoundError(f"Metadata model file not found at {metadata_model_path}")

        # Load the traced model
        scripted_model = torch.jit.load(scripted_model_path, map_location=device)
        if scripted_model:
            scripted_model.eval()
            logger.info(f"Traced model loaded successfully from {scripted_model_path}")

        # Load the metadata
        model_metadata = torch.load(metadata_model_path, map_location='cpu', weights_only=False)
        if model_metadata:
            logger.info(f"Model metadata loaded successfully from {metadata_model_path}")
            logger.info(f"Model metadata keys: {list(model_metadata.keys())}")
            
            config_full = model_metadata['config_full']
            if not config_full or not isinstance(config_full, dict):
                raise ValueError(f"Key `config_full` not found or not a dict")
            
            dataset_config = config_full['dataset']
            model_params = config_full['model_params']

            if not dataset_config or not isinstance(dataset_config, dict):
                raise ValueError(f"Key `dataset` not found or not a dict in config_full")
            alphabet_str = dataset_config['alphabet_string']
            max_text_len = dataset_config['max_text_len']
            output_mixture_components = model_params['output_mixture_components']

            lstm_size = model_params['lstm_size']
            attention_mixture_components = model_params['attention_mixture_components']

            ALPHABET_LIST = construct_alphabet_list(alphabet_str)
            ALPHABET_SIZE = len(ALPHABET_LIST)
            alphabet_map = get_alphabet_map(ALPHABET_LIST) 
            
            logger.info(f"Alphabet created. Size: {len(ALPHABET_LIST)}")
            logger.info("Model resources are loaded and ready")
        else:
            raise ValueError(f"Failed to load content frm metadata file")

    except Exception as e:
        logger.error(f"Error loading model resources: {e}", exc_info=True)
        scripted_model = None
        model_metadata = None
        raise
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down API and cleaning up resources")
    scripted_model = None 
    model_metadata = None 

app = FastAPI(
    title="Scriptify API", 
    description="API to generate handwriting from text using a PyTorch model.", 
    version="0.1.0",
    lifespan=lifespan
)

# add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173","http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.get("/", tags=["General"])
async def read_root():
    return {"message": "Welcome to the Scriptify Handwriting Generation API!"}

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    global scripted_model, model_metadata, device, alphabet_map, max_text_len, ALPHABET_LIST 
    
    is_healthy = all([scripted_model, model_metadata, device, alphabet_map, max_text_len, ALPHABET_LIST])
    
    return HealthResponse(
        status="healthy" if is_healthy else "unhealthy",
        model_loaded=bool(scripted_model),
        device=str(device) if device else "unknown",
        model_metadata_keys=list(model_metadata.keys()) if model_metadata else None,
    )

NUM_SAMPLES = 10  # generate N candidates per request, return the best one

# Expected pen-down segment count per isolated Arabic letter.
# Used to pick the best generation candidate (prefer count closest to expected).
ARABIC_EXPECTED_SEGMENTS: dict[str, int] = {
    'أ': 1, 'إ': 1, 'آ': 1, 'ا': 1,
    'ب': 2,
    'ت': 3, 'ث': 4,
    'ج': 2, 'ح': 1, 'خ': 2,
    'د': 1, 'ذ': 2,
    'ر': 1, 'ز': 2,
    'س': 1, 'ش': 4,
    'ص': 1, 'ض': 2,
    'ط': 1, 'ظ': 2,
    'ع': 1, 'غ': 2,
    'ف': 2, 'ق': 3,
    'ك': 1, 'ل': 1, 'م': 1,
    'ن': 2, 'ه': 1,
    'و': 1,
    'ى': 1, 'ي': 3, 'ة': 2,
}

def text_to_tensor(text: str, device: torch.device, batch_size: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert text to tensor format expected by the model"""
    global alphabet_map, max_text_len
    if alphabet_map is None:
        raise ValueError("Alphabet map not initialized during api startup")
    if max_text_len is None:
        raise ValueError("`max_text_len` is not initialized during api startup")
    padded_encoded_np, true_length = encode_text(
        text=text,
        char_to_index_map=alphabet_map,
        max_length=max_text_len
    )
    # Repeat for batch
    import numpy as np
    padded_batch = np.repeat(padded_encoded_np, batch_size, axis=0)
    char_seq = torch.from_numpy(padded_batch).to(device=device, dtype=torch.long)
    char_len = torch.tensor([true_length] * batch_size, device=device, dtype=torch.long)
    return char_seq, char_len


def _count_pen_segments(strokes: list[list[float]]) -> int:
    """Count the number of pen-down segments (higher = richer output)."""
    count = 0
    in_seg = False
    for s in strokes:
        if s[2] < 0.5:
            if not in_seg:
                count += 1
                in_seg = True
        else:
            in_seg = False
    return count


def _candidate_score(strokes: list[list[float]], target_segs: int | None) -> float:
    """Score a candidate stroke sequence. Higher = better.

    If target_segs is known: reward closeness to expected count and penalise excess.
    Otherwise fall back to raw segment count.
    """
    actual = _count_pen_segments(strokes)
    if target_segs is None:
        return float(actual)
    # reward each matched segment, penalise each extra one
    return float(min(actual, target_segs)) - 0.5 * max(0, actual - target_segs)


def generate_strokes(
    char_seq: torch.Tensor,
    char_lengths: torch.Tensor,
    max_gen_len: int,
    api_bias: float,
    current_device: torch.device,
    input_text: str = "",
) -> list[list[float]]:
    """Generate strokes using batch sampling, returning the richest candidate."""
    global scripted_model
    if scripted_model is None:
        raise ValueError("Scripted model not initialized.")

    with torch.no_grad():
        try:
            stroke_tensors = scripted_model.sample(
                char_seq,
                char_lengths,
                max_length=max_gen_len,
                bias=api_bias
            )

            # Each element of stroke_tensors is one sample: shape (T_i, 3)
            candidates: list[list[list[float]]] = []
            for tensor in stroke_tensors:
                if tensor.dim() == 2 and tensor.shape[-1] == 3:
                    candidates.append(tensor.cpu().numpy().tolist())

            if not candidates:
                return []

            # Convert to absolute coords, denoise, then pick by expected segment count
            target_segs = ARABIC_EXPECTED_SEGMENTS.get(input_text.strip()) if input_text else None
            best_trimmed: list[list[float]] = []
            best_score = float("-inf")
            for cand in candidates:
                abs_coords = convert_offsets_to_absolute_coords(cand)
                trimmed = trim_stroke_noise(abs_coords)
                trimmed = filter_spatially_distant_strokes(trimmed)
                if is_degenerate_strokes(trimmed):
                    continue  # skip bias-collapse straight-line candidates
                score = _candidate_score(trimmed, target_segs)
                if score > best_score:
                    best_score = score
                    best_trimmed = trimmed

            return best_trimmed

        except Exception as e:
            logger.error(f"Error in model sampling: {e}", exc_info=True)
            return []
            
@app.post("/generate", response_model=HandwritingResponse, tags=["Generation"])
async def generate_handwriting_endpoint(request: HandwritingRequest):
    if not all([scripted_model, model_metadata, device, alphabet_map, max_text_len]):
        logger.error("API not fully initialized. Check /health endpoint.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model or required resources not loaded."
        )
    
    assert device is not None, "Device is None inside generate_handwriting"
    start_time = time.time()
    
    try:
        normalised_text = normalise_arabic(request.text)
        target_segs = ARABIC_EXPECTED_SEGMENTS.get(normalised_text.strip())
        logger.info(
            f"GENERATE | raw='{request.text}' (U+{' '.join(f'{ord(c):04X}' for c in request.text)}) "
            f"-> normalised='{normalised_text}' | expected_segments={target_segs}"
        )
        char_seq_tensor, char_lengths_tensor = text_to_tensor(normalised_text, device, batch_size=NUM_SAMPLES)

        relative_stroke_offsets = generate_strokes(
            char_seq_tensor, char_lengths_tensor, request.max_length, request.bias, device,
            input_text=normalised_text,
        )

        if not relative_stroke_offsets:
            return HandwritingResponse(
                success=False,
                input_text=request.text,
                strokes=[],
                num_points=0,
                generation_time_ms=(time.time() - start_time) * 1000,
                message="No strokes generated."
            )

        final_strokes = relative_stroke_offsets
        generation_time_ms = (time.time() - start_time) * 1000
        actual_segs = sum(
            1 for i, s in enumerate(final_strokes)
            if s[2] < 0.5 and (i == 0 or final_strokes[i-1][2] >= 0.5)
        )
        logger.info(
            f"RESULT  | pts={len(final_strokes)} segments={actual_segs} "
            f"expected={target_segs} time={generation_time_ms:.0f}ms"
        )

        return HandwritingResponse(
            input_text=request.text,
            strokes=final_strokes,
            num_points=len(final_strokes),
            generation_time_ms=generation_time_ms
        )
    except ValueError as ve:
        logger.error(f"ValueError during generation for '{request.text}': {ve}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error for '{request.text}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred.")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server for Scriptify API...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, app_dir=".")