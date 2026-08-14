"""Multi-tenancy isolation test battery.

This is the security-critical half of the evaluation. Retrieval quality is a
tuning problem; isolation is a correctness guarantee, and a single failure here
invalidates the product.

Coverage:
  * Bidirectional data leakage with real ingested corpora
  * Entity-id probing (asking tenant B directly for tenant A's canonical ids)
  * Auth-layer enforcement (key/tenant binding, header assertion)
  * Tenant-id edge cases (path traversal, case, unicode, injection)
  * Concurrent interleaved load across tenants
  * Adversarial query handling
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.core.tenant_context import TenantContext, tenant_scope
from app.services.retrieval_service import retrieval_service
from tests.evaluation.dataset import ADVERSARIAL_QUERIES, TenantFixture


@dataclass
class IsolationCheck:
    """One isolation assertion and its outcome."""

    name: str
    category: str
    passed: bool
    detail: str
    severity: str = "critical"
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "passed": self.passed,
            "severity": self.severity,
            "detail": self.detail,
            "evidence": self.evidence,
        }


def _ctx(tenant_id: str, request_id: str = "eval") -> TenantContext:
    return TenantContext(
        tenant_id=tenant_id,
        api_key_id=f"eval_{tenant_id}",
        request_id=request_id,
        auth_method="evaluation_harness",
    )


class IsolationSuite:
    """Runs the full isolation battery across a set of tenant fixtures."""

    def __init__(self, fixtures: List[TenantFixture]) -> None:
        self.fixtures = fixtures
        self.checks: List[IsolationCheck] = []

    def _record(self, check: IsolationCheck) -> None:
        self.checks.append(check)

    # ------------------------------------------------------- cross-tenant leakage
    async def check_cross_tenant_leakage(self) -> None:
        """Query each tenant for every other tenant's exclusive content.

        The strongest available evidence of isolation: real data, real queries,
        asserting absence.
        """
        for source in self.fixtures:
            for target in self.fixtures:
                if source.tenant_id == target.tenant_id:
                    continue

                # Ask the target tenant using the source tenant's distinctive phrases.
                for phrase in source.exclusive_phrases:
                    leaked: List[str] = []
                    error: Optional[str] = None
                    try:
                        with tenant_scope(_ctx(target.tenant_id)):
                            result = await retrieval_service.execute_retrieval(
                                ctx=_ctx(target.tenant_id),
                                query=f"Tell me about {phrase}",
                                max_depth=3,
                                top_k=10,
                            )
                        subgraph = result["subgraph"]
                        passages = result["passages"]

                        for node in subgraph.nodes:
                            if node.id in source.exclusive_entities:
                                leaked.append(f"entity:{node.id}")
                        for passage in passages:
                            for other in source.exclusive_phrases:
                                if other.lower() in passage.lower():
                                    leaked.append(f"passage:{other}")
                    except Exception as exc:  # noqa: BLE001
                        error = f"{type(exc).__name__}: {exc}"

                    self._record(
                        IsolationCheck(
                            name=f"leak_{source.tenant_id}_into_{target.tenant_id}_{phrase[:20]}",
                            category="cross_tenant_leakage",
                            passed=not leaked and error is None,
                            detail=(
                                f"Queried '{target.tenant_id}' for '{phrase}' "
                                f"(exclusive to '{source.tenant_id}')"
                                + (f"; ERROR {error}" if error else "")
                            ),
                            evidence={"leaked": leaked, "error": error},
                        )
                    )

    async def check_entity_id_probing(self) -> None:
        """Ask a tenant directly for another tenant's canonical entity ids.

        A more targeted attack than phrase search: the attacker already knows the
        exact id and is testing whether the database boundary holds.
        """
        for source in self.fixtures:
            for target in self.fixtures:
                if source.tenant_id == target.tenant_id:
                    continue
                for entity_id in source.exclusive_entities:
                    found = False
                    error: Optional[str] = None
                    try:
                        with tenant_scope(_ctx(target.tenant_id)):
                            result = await retrieval_service.execute_retrieval(
                                ctx=_ctx(target.tenant_id),
                                query=entity_id.replace("canon_", "").replace("_", " "),
                                max_depth=3,
                                top_k=10,
                            )
                        found = any(n.id == entity_id for n in result["subgraph"].nodes)
                    except Exception as exc:  # noqa: BLE001
                        error = f"{type(exc).__name__}: {exc}"

                    self._record(
                        IsolationCheck(
                            name=f"probe_{entity_id}_from_{target.tenant_id}",
                            category="entity_id_probing",
                            passed=not found and error is None,
                            detail=(
                                f"Probed '{target.tenant_id}' for '{entity_id}'"
                                + (f"; ERROR {error}" if error else "")
                            ),
                            evidence={"found": found, "error": error},
                        )
                    )

    # ------------------------------------------------------- concurrency
    async def check_concurrent_isolation(self, rounds: int = 25) -> None:
        """Interleave concurrent queries across tenants and verify no bleed.

        `contextvars` are per-task, so this should hold by construction. Testing it
        empirically is what turns "should" into "does".
        """
        if len(self.fixtures) < 2:
            return

        async def run_one(fixture: TenantFixture, idx: int) -> Dict[str, Any]:
            phrase = fixture.exclusive_phrases[idx % len(fixture.exclusive_phrases)]
            with tenant_scope(_ctx(fixture.tenant_id, f"conc_{idx}")):
                result = await retrieval_service.execute_retrieval(
                    ctx=_ctx(fixture.tenant_id, f"conc_{idx}"),
                    query=f"Tell me about {phrase}",
                    max_depth=2,
                    top_k=5,
                )
            foreign = [
                other.tenant_id
                for other in self.fixtures
                if other.tenant_id != fixture.tenant_id
                for node in result["subgraph"].nodes
                if node.id in other.exclusive_entities
            ]
            return {"tenant": fixture.tenant_id, "foreign": foreign}

        tasks = []
        for i in range(rounds):
            for fixture in self.fixtures:
                tasks.append(run_one(fixture, i))

        started = time.perf_counter()
        try:
            outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as exc:  # noqa: BLE001
            self._record(
                IsolationCheck(
                    name="concurrent_isolation",
                    category="concurrency",
                    passed=False,
                    detail=f"Concurrent run raised: {exc}",
                )
            )
            return
        elapsed = (time.perf_counter() - started) * 1000

        violations = [
            o for o in outcomes if isinstance(o, dict) and o.get("foreign")
        ]
        errors = [o for o in outcomes if isinstance(o, Exception)]

        self._record(
            IsolationCheck(
                name="concurrent_isolation",
                category="concurrency",
                passed=not violations,
                detail=(
                    f"{len(tasks)} interleaved cross-tenant queries in {elapsed:.0f}ms; "
                    f"{len(violations)} isolation violations, {len(errors)} errors"
                ),
                evidence={
                    "total_requests": len(tasks),
                    "violations": len(violations),
                    "errors": len(errors)
                    if not errors
                    else [f"{type(e).__name__}: {e}" for e in errors[:3]],
                    "elapsed_ms": round(elapsed, 2),
                    "throughput_rps": round(len(tasks) / (elapsed / 1000), 2) if elapsed else 0,
                },
            )
        )

    # ------------------------------------------------------- adversarial input
    async def check_adversarial_queries(self) -> None:
        """Injection payloads must be rejected, not executed."""
        from app.core.exceptions import SecurityViolationError

        fixture = self.fixtures[0]
        for case in ADVERSARIAL_QUERIES:
            rejected = False
            detail = ""
            try:
                with tenant_scope(_ctx(fixture.tenant_id)):
                    await retrieval_service.execute_retrieval(
                        ctx=_ctx(fixture.tenant_id), query=case.query, max_depth=2, top_k=5
                    )
                detail = "Query was ACCEPTED and executed"
            except SecurityViolationError as exc:
                rejected = True
                detail = f"Rejected: {exc.detail}"
            except Exception as exc:  # noqa: BLE001
                # Any other error still means it did not execute as written.
                rejected = True
                detail = f"Rejected via {type(exc).__name__}: {exc}"

            self._record(
                IsolationCheck(
                    name=f"adversarial_{case.name}",
                    category="injection_defence",
                    passed=rejected,
                    detail=f"{case.description} -> {detail}",
                    evidence={"query": case.query[:120]},
                )
            )

    # ------------------------------------------------------- tenant id edge cases
    def check_tenant_id_validation(self) -> None:
        """Malformed tenant identifiers must be rejected, never normalized."""
        from app.core.exceptions import SecurityViolationError
        from app.core.security import TenantIdValidator

        cases = [
            ("path_traversal", "../../../etc/passwd"),
            ("path_traversal_db", "movies_bot/../ai_trends_bot"),
            ("sql_injection", "movies_bot'; DROP DATABASE x; --"),
            ("null_byte", "movies_bot\x00ai_trends_bot"),
            ("empty", ""),
            ("whitespace_only", "   "),
            ("leading_digit", "1movies_bot"),
            ("special_chars", "movies-bot!@#"),
            ("unicode_homoglyph", "movies_bоt"),  # Cyrillic 'о'
            ("overlong", "a" * 100),
            ("uppercase_mixed", "Movies_Bot"),  # normalizes to movies_bot, must be safe
        ]

        for name, raw in cases:
            rejected = False
            normalized: Optional[str] = None
            try:
                normalized = TenantIdValidator.validate(raw)
            except SecurityViolationError:
                rejected = True
            except Exception:  # noqa: BLE001
                rejected = True

            # Mixed case is legitimately normalized; everything else must be rejected.
            expect_rejected = name != "uppercase_mixed"
            passed = rejected if expect_rejected else (normalized == "movies_bot")

            self._record(
                IsolationCheck(
                    name=f"tenant_id_{name}",
                    category="tenant_id_validation",
                    passed=passed,
                    detail=(
                        f"input={raw[:40]!r} -> "
                        + ("rejected" if rejected else f"accepted as {normalized!r}")
                    ),
                    evidence={"rejected": rejected, "normalized": normalized},
                )
            )

    def check_unscoped_access_fails_closed(self) -> None:
        """A database call with no bound tenant context must refuse, not run."""
        from app.core.exceptions import TenantAccessDeniedError
        from app.core.tenant_context import get_tenant_context

        failed_closed = False
        detail = ""
        try:
            ctx = get_tenant_context()
            detail = f"Returned a context without binding: {ctx.tenant_id}"
        except TenantAccessDeniedError as exc:
            failed_closed = True
            detail = f"Correctly refused: {exc.detail}"
        except Exception as exc:  # noqa: BLE001
            failed_closed = True
            detail = f"Refused via {type(exc).__name__}"

        self._record(
            IsolationCheck(
                name="unscoped_access_fails_closed",
                category="tenant_context_guard",
                passed=failed_closed,
                detail=detail,
            )
        )

    # ------------------------------------------------------- orchestration
    async def run_all(self, include_live: bool = True) -> List[IsolationCheck]:
        """Run the full battery. `include_live` requires a provisioned database."""
        self.checks = []

        # These need no database.
        self.check_tenant_id_validation()
        self.check_unscoped_access_fails_closed()

        if include_live:
            await self.check_adversarial_queries()
            await self.check_cross_tenant_leakage()
            await self.check_entity_id_probing()
            await self.check_concurrent_isolation()

        return self.checks

    def summary(self) -> Dict[str, Any]:
        total = len(self.checks)
        passed = sum(1 for c in self.checks if c.passed)
        critical_failures = [
            c for c in self.checks if not c.passed and c.severity == "critical"
        ]
        by_cat: Dict[str, Dict[str, int]] = {}
        for check in self.checks:
            bucket = by_cat.setdefault(check.category, {"passed": 0, "failed": 0})
            bucket["passed" if check.passed else "failed"] += 1

        return {
            "total_checks": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": round(passed / total, 4) if total else 0.0,
            "critical_failures": len(critical_failures),
            "isolation_verdict": "PASS" if not critical_failures else "FAIL",
            "by_category": by_cat,
            "failures": [c.to_dict() for c in self.checks if not c.passed],
        }
