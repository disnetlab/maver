"""Apply later operator verdicts to private corpus metadata."""

from dataclasses import replace

from maver.memory.ports import SessionMemory


def reviewed_memory(
    memory: SessionMemory, *, state: str, verdict: str, corrected_cause: str
) -> SessionMemory:
    """Keep the original projections while making the latest review authoritative."""
    return replace(
        memory,
        lifecycle_state=state,
        evaluation_label="operator_confirmed" if verdict == "confirmed" else "operator_rejected",
        evaluation_source="operator",
        corrected_cause=corrected_cause if verdict == "corrected" else "",
        authoritative_correction=bool(corrected_cause and verdict == "corrected"),
        evaluation_diagnosis_success=(
            False if verdict == "corrected" else memory.evaluation_diagnosis_success
        ),
    )
