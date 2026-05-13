export interface HandwritingRequest {
  text: string;
  bias: number;
}

export interface HandwritingResponse {
  success: boolean;
  strokes: number[][];
  message?: string;
}

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "https://h3nock-scriptify-api.hf.space/";
const DEFAULT_TIMEOUT = 180000;

class ApiError extends Error {
  status?: number;

  constructor(message: string, status?: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

const createFetchWithTimeout = (timeoutMs: number = DEFAULT_TIMEOUT) => {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  return {
    fetch: (url: string, options: RequestInit = {}) =>
      fetch(url, { ...options, signal: controller.signal }),
    cleanup: () => clearTimeout(timeoutId),
  };
};

const getErrorMessage = (error: unknown, status?: number): string => {
  if (status) {
    const statusMessages: Record<number, string> = {
      404: "Service not found. Please try again later.",
      500: "Service error. Please try again in a moment.",
      503: "Service temporarily unavailable. Please try again later.",
    };
    return statusMessages[status] || "Request failed. Please try again.";
  }

  if (error instanceof Error) {
    if (error.name === "AbortError") {
      return "Request timed out. The server may be slow or unavailable.";
    }
    if (error.name === "TypeError" || error.message.includes("NetworkError")) {
      return "Unable to connect. Please try again later.";
    }
  }

  return "An unexpected error occurred. Please try again.";
};

export const generateHandwriting = async (
  text: string,
  bias: number = 2
): Promise<HandwritingResponse> => {
  const { fetch: fetchWithTimeout, cleanup } = createFetchWithTimeout();

  try {
    const response = await fetchWithTimeout(`${API_BASE_URL}/generate`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify({
        text: text.trim(),
        bias,
      }),
    });

    cleanup();

    if (!response.ok) {
      throw new ApiError(
        getErrorMessage(null, response.status),
        response.status
      );
    }

    const data = await response.json();

    return {
      success: true,
      strokes: data.strokes,
      message: data.message,
    };
  } catch (error) {
    cleanup();
    console.error("API Error:", error);
    throw new ApiError(getErrorMessage(error));
  }
};

// ── Feedback API ──────────────────────────────────────────────────────────────

export interface CandidatesResponse {
  candidates: number[][][];
  text: string;
}

export interface FeedbackStats {
  total_accepted: number;
  per_character: Record<string, number>;
  fine_tune_status: { running: boolean; progress: string; last_result: unknown };
}

export const generateCandidates = async (
  text: string,
  bias: number,
  n: number = 6
): Promise<CandidatesResponse> => {
  const LOCAL_URL = "http://127.0.0.1:8000";
  const response = await fetch(`${LOCAL_URL}/feedback/candidates`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: text.trim(), bias, n }),
  });
  if (!response.ok) throw new Error(`Candidates request failed: ${response.status}`);
  return response.json();
};

export const saveFeedback = async (
  text: string,
  strokes: number[][],
  bias: number,
  accepted: boolean
): Promise<{ id: string; total_accepted: number }> => {
  const LOCAL_URL = "http://127.0.0.1:8000";
  const response = await fetch(`${LOCAL_URL}/feedback/save`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, strokes, bias, accepted }),
  });
  if (!response.ok) throw new Error(`Save feedback failed: ${response.status}`);
  return response.json();
};

export const triggerFineTune = async (
  n_epochs: number = 3,
  lr: number = 1e-5
): Promise<{ started: boolean; message: string }> => {
  const LOCAL_URL = "http://127.0.0.1:8000";
  const response = await fetch(`${LOCAL_URL}/feedback/fine-tune`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ n_epochs, lr }),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(err.detail || "Fine-tune failed");
  }
  return response.json();
};

export const getFeedbackStats = async (): Promise<FeedbackStats> => {
  const LOCAL_URL = "http://127.0.0.1:8000";
  const response = await fetch(`${LOCAL_URL}/feedback/stats`);
  if (!response.ok) throw new Error("Stats request failed");
  return response.json();
};

export const checkAPIHealth = async (): Promise<boolean> => {
  try {
    const response = await fetch(`${API_BASE_URL}/health`, {
      method: "GET",
      headers: {
        accept: "application/json",
      },
    });
    return response.ok;
  } catch (error) {
    console.error("Health check failed:", error);
    return false;
  }
};
