import React, { useState, useCallback, useEffect } from "react";
import HandwritingCanvas from "../HandwritingCanvas/HandwritingCanvas";
import type { FeedbackStats } from "../../services/api";
import {
  generateCandidates,
  saveFeedback,
  triggerFineTune,
  getFeedbackStats,
} from "../../services/api";
import "./FeedbackPanel.css";

interface CandidateState {
  strokes: number[][];
  decision: "none" | "accepted" | "rejected";
}

const FeedbackPanel: React.FC = () => {
  const [inputText, setInputText] = useState("");
  const [bias, setBias] = useState(1.5);
  const [candidates, setCandidates] = useState<CandidateState[]>([]);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState<number | null>(null);
  const [stats, setStats] = useState<FeedbackStats | null>(null);
  const [ftStatus, setFtStatus] = useState<string>("");
  const [error, setError] = useState("");

  const refreshStats = useCallback(async () => {
    try {
      const s = await getFeedbackStats();
      setStats(s);
    } catch {
      /* ignore */
    }
  }, []);

  useEffect(() => {
    refreshStats();
  }, [refreshStats]);

  // Poll fine-tune status while running
  useEffect(() => {
    if (!stats?.fine_tune_status.running) return;
    const id = setInterval(refreshStats, 2000);
    return () => clearInterval(id);
  }, [stats?.fine_tune_status.running, refreshStats]);

  const handleGenerate = async () => {
    if (!inputText.trim()) return;
    setLoading(true);
    setError("");
    setCandidates([]);
    try {
      const res = await generateCandidates(inputText.trim(), bias, 6);
      if (res.candidates.length === 0) {
        setError("No valid candidates generated. Try a different bias or character.");
        return;
      }
      setCandidates(res.candidates.map((s) => ({ strokes: s, decision: "none" })));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to generate candidates.");
    } finally {
      setLoading(false);
    }
  };

  const handleDecision = async (
    idx: number,
    accepted: boolean
  ) => {
    const cand = candidates[idx];
    if (!cand || cand.decision !== "none") return;
    setSaving(idx);
    try {
      await saveFeedback(inputText.trim(), cand.strokes, bias, accepted);
      setCandidates((prev) =>
        prev.map((c, i) =>
          i === idx ? { ...c, decision: accepted ? "accepted" : "rejected" } : c
        )
      );
      await refreshStats();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to save feedback.");
    } finally {
      setSaving(null);
    }
  };

  const handleFineTune = async () => {
    try {
      const res = await triggerFineTune(15, 1e-4);
      setFtStatus(res.message);
      await refreshStats();
    } catch (e) {
      setFtStatus(e instanceof Error ? e.message : "Failed to start fine-tune.");
    }
  };

  const totalAccepted = stats?.total_accepted ?? 0;
  const ftRunning = stats?.fine_tune_status.running ?? false;
  const ftProgress = stats?.fine_tune_status.progress ?? "";

  return (
    <div className="feedback-panel">
      <div className="feedback-header">
        <h2>Feedback Mode</h2>
        <p className="feedback-subtitle">
          Generate multiple candidates, accept the ones that look correct. After collecting
          samples, fine-tune the model to reinforce good outputs.
        </p>
      </div>

      {/* Controls */}
      <div className="feedback-controls">
        <input
          className="feedback-input"
          type="text"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="Enter a single Arabic character…"
          maxLength={4}
          dir="rtl"
          onKeyDown={(e) => e.key === "Enter" && handleGenerate()}
        />
        <div className="bias-control">
          <label>Bias: {bias.toFixed(1)}</label>
          <input
            type="range"
            min={0.5}
            max={3}
            step={0.1}
            value={bias}
            onChange={(e) => setBias(parseFloat(e.target.value))}
          />
        </div>
        <button
          className="btn-generate-candidates"
          onClick={handleGenerate}
          disabled={loading || !inputText.trim()}
        >
          {loading ? "Generating…" : "Generate Candidates"}
        </button>
      </div>

      {error && <p className="feedback-error">{error}</p>}

      {/* Candidate grid */}
      {candidates.length > 0 && (
        <div className="candidates-grid">
          {candidates.map((cand, idx) => (
            <div
              key={idx}
              className={`candidate-card ${cand.decision !== "none" ? `decision-${cand.decision}` : ""}`}
            >
              <div className="candidate-canvas-wrapper">
                <HandwritingCanvas
                  strokes={cand.strokes}
                  canvasWidth={360}
                  canvasHeight={220}
                  strokeColor="#1a1a1a"
                  strokeWidth={3}
                  animationSpeed={5}
                />
              </div>
              <div className="candidate-actions">
                {cand.decision === "none" ? (
                  <>
                    <button
                      className="btn-accept"
                      onClick={() => handleDecision(idx, true)}
                      disabled={saving !== null}
                    >
                      {saving === idx ? "…" : "✓ Accept"}
                    </button>
                    <button
                      className="btn-reject"
                      onClick={() => handleDecision(idx, false)}
                      disabled={saving !== null}
                    >
                      {saving === idx ? "…" : "✗ Reject"}
                    </button>
                  </>
                ) : (
                  <span className={`decision-badge badge-${cand.decision}`}>
                    {cand.decision === "accepted" ? "✓ Accepted" : "✗ Rejected"}
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Stats + Fine-tune */}
      <div className="feedback-footer">
        <div className="feedback-stats">
          <span className="stat-total">
            <strong>{totalAccepted}</strong> accepted samples total
          </span>
          {stats && Object.keys(stats.per_character).length > 0 && (
            <div className="per-char-stats">
              {Object.entries(stats.per_character)
                .sort((a, b) => b[1] - a[1])
                .map(([char, count]) => (
                  <span key={char} className="char-badge">
                    {char}: {count}
                  </span>
                ))}
            </div>
          )}
        </div>

        <div className="fine-tune-section">
          {ftRunning ? (
            <div className="ft-running">
              <div className="ft-spinner" />
              <span>{ftProgress}</span>
            </div>
          ) : (
            <button
              className="btn-fine-tune"
              onClick={handleFineTune}
              disabled={totalAccepted < 5 || ftRunning}
              title={totalAccepted < 5 ? `Need at least 5 accepted samples (have ${totalAccepted})` : ""}
            >
              Fine-tune Model ({totalAccepted} samples)
            </button>
          )}
          {ftStatus && !ftRunning && (
            <p className="ft-status">{ftStatus}</p>
          )}
          {!!stats?.fine_tune_status.last_result && !ftRunning && (
            <p className="ft-result">
              Last run: {(stats.fine_tune_status.last_result as { epochs: number; losses: number[] }).epochs} epochs,
              final loss{" "}
              {(stats.fine_tune_status.last_result as { losses: number[] }).losses.slice(-1)[0]}
            </p>
          )}
        </div>
      </div>
    </div>
  );
};

export default FeedbackPanel;
