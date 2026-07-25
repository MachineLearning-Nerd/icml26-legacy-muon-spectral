# Claim 6 — Newton-Schulz


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_e2167565c166", "created_at": "2026-07-21T17:21:50+00:00", "title": "C6: Newton-Schulz computes Ψ_{1/2}, Ψ_{1/4} (no SVD) — VERIFIED"}
-->
Algorithm 1 coupled Newton-Schulz computes Ψ_{1/2} and Ψ_{1/4} using only matrix multiplications (no explicit SVD). Implementation: polar Newton-Schulz for P=UV^T + Denman-Beavers on the SPD factor H=PᵀO, giving Ψ_{1/2}=P·H^{1/2}=UΣ^{1/2}V^T. **VERIFIED** vs SVD-based Ψ_p to ≤1e-15 (cond 5–50).
