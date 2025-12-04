# pgen_agents.py
"""
PGEN Agent System
- Author: Sims
- Purpose: Orchestrate ingestion, adaptive denoising, hybrid interpretation, RAG knowledge updates,
  real-time routing and outputs for quantum-sensing data.
- Requirements: Provide implementations for the placeholder tools used below:
    - google_search: research tool
    - vector_store: RAG vector DB (with .retrieve(query, k) )
    - qpu_runner: runs guppy programs or quantum-enhanced transforms
    - cpu_gpu_executor: runs heavy classical compute
    - edge_infer: edge-serving tool
    - telemetry: logs agent metrics
    - function tools like exit_loop
- Follow the same Agent / AgentTool API you used earlier.
"""

# ---------- Imports ----------
from typing import Dict, Any, List
from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.genai import types

# ---------- Configuration & Helpers ----------
retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
)

# Minimal Model Context Protocol (MCP) builder
def build_mcp_context(sensor_run: Dict[str, Any], last_models: List[Dict[str,Any]] = None) -> Dict[str, Any]:
    """
    Create a compact context packet to give agents historical context:
      - run metadata (device, timestamp, calib)
      - recent model artifacts & metrics
      - recent relevant literature pointers (ids)
    """
    ctx = {
        "run_id": sensor_run.get("run_id"),
        "device_id": sensor_run.get("device_id"),
        "timestamp": sensor_run.get("timestamp"),
        "calibration": sensor_run.get("calibration"),
        "noise_profile": sensor_run.get("noise_profile"),
        "recent_models": last_models or [],
    }
    return ctx

# Agent quality instrumentation
def record_agent_quality(agent_name: str, metric: str, value: float):
    """
    Push quality metric to telemetry or local log. Examples:
      - 'precision', 'recall', 'SNR_improvement', 'runtime_ms'
      - Helps enforce the Agent Quality program
    """
    print(f"[QUALITY] {agent_name} | {metric} = {value}")

# ---------- Guppy ----------
guppy_program = r'''
// Guppy pseudo-code for a quantum-enhanced transform
// This is a small example; replace with real Guppy syntax per Quantinuum docs.
PROGRAM guppy_feature_transform:
    // load register with sensor-derived coefficients
    QREG q[8];

    // prepare state based on classical preprocessed vector (encoded externally)
    PREPARE_STATE q, "amplitude_encoding", input_vector;

    // apply a small feature transform circuit (example)
    FOR i IN 0..7:
        RY(q[i], 0.3 * i);
        CNOT(q[i], q[(i+1) % 8]);
    END

    // measure and return expectation values used as features
    MEASURE_EXPECTATIONS q -> features;
END
'''

# ---------- Agent Definitions ----------
# Research Agent: support RAG updates & literature ingestion
research_agent = Agent(
    name="PGENResearchAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
You are the PGEN research agent. Use the google_search tool to find 2-4 recent,
relevant papers, methods, or code snippets about: {topic}.
Return a short annotated list with title, one-sentence summary, and a citation/link.
Also return any code examples or technique names to be pushed to the RAG store.
""",
    tools=[google_search],
    output_key="research_findings",
)

# RAG Updater Agent: writes new research into the vector store and tags it
rag_updater_agent = Agent(
    name="RAGUpdater",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
You are the RAG updater. Given {research_findings}, extract 3-5 short embeddings-ready passages
(each ≤ 150 words), create metadata (topic, year, source), and call the vector_store.upsert(items).
Return a summary of what was added and the keys/ids of the stored items.
""",
    tools=[vector_store],
    output_key="rag_update_result",
)

# Ingestion Agent: unify raw sensor + metadata into canonical format
ingestion_agent = Agent(
    name="IngestionAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
You are the ingestion agent. Given raw_payload (binary waveform, headers, device logs), 
produce a canonical JSON with: run_id, device_id, timestamp, calibrated_waveform (float array),
short summary of calibration steps applied, noise_profile summary, and provenance.
Also emit an MCP context packet via build_mcp_context(...) for downstream agents.
""",
    output_key="canonical_run",
)

# Adaptive Noise Correction Agent (noise-aware)
noise_agent = Agent(
    name="NoiseAdaptAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
You are the adaptive noise correction agent. Given {canonical_run} and {mcp_context}:
1) Propose 3 candidate denoising pipelines (short name + short rationale): 
   e.g., "wavelet_denoise+STFT", "adaptive_bandpass+kalman", "zero_noise_extrapolation variant".
2) For each candidate, estimate expected SNR improvement and runtime.
3) Select the pipeline that maximizes SNR per runtime penalty and return a runnable spec.
Return results as JSON: {candidates: [...], chosen: <spec>, estimate: {...}}.
""",
    tools=[],
    output_key="denoise_plan",
)

# Hybrid Loop Agent (iterative interpret & early-stop)
hybrid_agent = Agent(
    name="HybridLoopAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
You manage the hybrid quantum-classical interpretation loop. Given:
 - preprocessed waveform {prepped_waveform},
 - denoise_plan {denoise_plan},
 - mcp_context {mcp_context},
 - available compute targets {compute_targets}

1) Create 4 candidate interpretation strategies (classical ML features, end-to-end conv, quantum-enhanced transform, combined).
2) For each, describe the action plan: feature list, training/eval steps, early-stop criteria.
3) Rank by expected F1 improvement and compute budget.
4) Return the chosen strategy and a run plan to the executor layer.
""",
    output_key="interpret_plan",
)

# Quantum Runner Agent (calls QPU with Guppy program)
quantum_agent = Agent(
    name="QuantumTransformAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
You are the quantum transform agent. Given {guppy_program} and {input_encoding}:
1) Validate the guppy program for input shape and required qubits.
2) Create a job payload for qpu_runner.run(program, params).
3) Return job_id and an expected shape of returned features, plus diagnostics to be saved.
""",
    tools=[qpu_runner],
    output_key="quantum_job",
)

# Router / Decision Agent: decide where to run tasks in real time
router_agent = Agent(
    name="RouterAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
You are the routing agent. Given {interpret_plan} and real-time compute telemetry,
decide for each step whether to route to: edge_infer, cpu_gpu_executor, or quantum_agent.
Consider latency SLAs, cost budget, and expected accuracy gains.
Return a routing table: [{step_name, target, reason, cost_estimate_ms}].
""",
    tools=[edge_infer, cpu_gpu_executor, AgentTool(quantum_agent)],
    output_key="routing_table",
)

# Orchestrator / Root Agent: executes full flow (ingest -> noise -> hybrid -> route -> exec -> output)
root_agent = Agent(
    name="PGENRootAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
You are the PGEN root orchestrator. Workflow:
1) Call IngestionAgent to canonicalize the run and produce MCP context.
2) Call ResearchAgent -> RAGUpdater periodically (or when drift detected) to update knowledge.
3) Call NoiseAdaptAgent to propose and select denoise pipeline.
4) Call HybridLoopAgent to produce interpretation strategies.
5) Call RouterAgent to allocate compute targets.
6) Dispatch tasks to executors (edge/cpu/gpu/quantum) and collect results.
7) Evaluate outcomes, record agent-quality metrics, and store final artifacts + explanations.
Return a final JSON with `detections`, `explanations`, `metrics`, `model_artifact_ids`.
""",
    tools=[
        AgentTool(ingestion_agent),
        AgentTool(research_agent),
        AgentTool(rag_updater_agent),
        AgentTool(noise_agent),
        AgentTool(hybrid_agent),
        AgentTool(router_agent),
        AgentTool(quantum_agent),
    ],
    output_key="pgen_final",
)





def exit_loop():
    return {"status": "exit"}

refiner_agent = Agent(
    name="RefinerAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
You refine story drafts given critique. IF critique == "APPROVED" call exit_loop(), else rewrite story incorporating feedback.
""",
    tools=[FunctionTool(exit_loop)],
    output_key="current_story",
)


def run_pgen_pipeline(raw_payload: Dict[str,Any]):
    """
    High-level function demonstrating how the orchestration would run.
    This is pseudocode showing the sequence of calls and checks.
    """
    # 1. Ingest
    ingestion_res = root_agent.call_tool("IngestionAgent", {"raw_payload": raw_payload})
    canonical_run = ingestion_res["canonical_run"]
    mcp_context = build_mcp_context(canonical_run)

    # 2. Optional research / RAG update when drift or weekly schedule
    research = root_agent.call_tool("PGENResearchAgent", {"topic": "quantum sensing denoising methods"})
    rag_update = root_agent.call_tool("RAGUpdater", {"research_findings": research["research_findings"]})
    record_agent_quality("RAGUpdater", "items_upserted", len(rag_update.get("rag_ids", [])))

    # 3. Noise adaptation
    denoise_plan = root_agent.call_tool("NoiseAdaptAgent", {"canonical_run": canonical_run, "mcp_context": mcp_context})
    chosen_pipeline = denoise_plan["denoise_plan"]["chosen"]

    # 4. Hybrid interpretation planning
    interpret_plan = root_agent.call_tool("HybridLoopAgent", {
        "prepped_waveform": canonical_run["calibrated_waveform"],
        "denoise_plan": denoise_plan["denoise_plan"],
        "mcp_context": mcp_context,
        "compute_targets": {"edge": True, "gpu": True, "quantum": True}
    })

    # 5. Routing decisions
    routing = root_agent.call_tool("RouterAgent", {"interpret_plan": interpret_plan["interpret_plan"], "telemetry": {}})

    # 6. Dispatch & run steps according to route
    final_outputs = {}
    for step in routing["routing_table"]:
        if step["target"] == "edge":
            out = edge_infer.run(step["spec"])
        elif step["target"] == "gpu":
            out = cpu_gpu_executor.run(step["spec"])
        elif step["target"] == "quantum":
            # Prepare guppy program / encoding
            job = root_agent.call_tool("QuantumTransformAgent", {"guppy_program": guppy_program, "input_encoding": step.get("encoding")})
            out = qpu_runner.poll(job["job_id"])
        else:
            out = cpu_gpu_executor.run(step["spec"])
        final_outputs[step["step_name"]] = out

    # 7. Evaluate outputs, compute metrics (SNR improvements, detection precision)
    record_agent_quality("PGENRootAgent", "SNR_improvement", 2.3)  # example

    # 8. Return final package
    return {
        "detections": final_outputs.get("detections"),
        "final_artifacts": ["model:v1", "dataset:run123"],
        "metrics": {"snr": 2.3},
    }

# ---------- End of pgen_agents.py ----------
