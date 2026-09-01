"""Run the bounded harness evolution reference example."""

from harness_lab import (
    EditSurface,
    SurfacePatch,
    attributed_proposal,
    build_demo_registry,
)


def main() -> None:
    registry = build_demo_registry()
    proposals = [
        attributed_proposal(
            registry,
            name="fix observed failures",
            failure_type="missing-persistence-and-retry",
            target_surface=EditSurface.WORKFLOW,
            patches=(
                SurfacePatch(
                    EditSurface.WORKFLOW,
                    add=frozenset({"retry_transient"}),
                ),
                SurfacePatch(
                    EditSurface.TOOL,
                    add=frozenset({"write_artifact"}),
                ),
            ),
            expected_fixes=("persist-artifact", "recover-transient-tool-error"),
            at_risk=("tool-cost",),
        )
    ]

    first = proposals[0]
    first_decision = registry.propose_evaluate_accept(first)

    proposals = [
        attributed_proposal(
            registry,
            name="remove safety for speed",
            failure_type="latency",
            target_surface=EditSurface.WORKFLOW,
            patches=(
                SurfacePatch(
                    EditSurface.WORKFLOW,
                    remove=frozenset({"ask_before_destructive"}),
                ),
            ),
            expected_fixes=("latency",),
            at_risk=("safe-shell",),
        ),
        attributed_proposal(
            registry,
            name="turn off verifier",
            failure_type="evaluation-failure",
            target_surface=EditSurface.TOOL,
            patches=(),
            expected_fixes=("evaluation-failure",),
            attempted_control_plane_updates=("verifier",),
        ),
    ]

    print(f"immutable control plane: {registry.control_plane_digest}")
    print(
        f"{'ACCEPT' if first_decision.accepted else 'REJECT':6} {first.name:24} "
        f"in={first_decision.held_in_before:.2f}->{first_decision.held_in_after} "
        f"out={first_decision.held_out_before:.2f}->{first_decision.held_out_after} "
        f"reasons={list(first_decision.reasons)}"
    )
    for proposal in proposals:
        decision = registry.propose_evaluate_accept(proposal)
        status = "ACCEPT" if decision.accepted else "REJECT"
        print(
            f"{status:6} {proposal.name:24} "
            f"in={decision.held_in_before:.2f}->{decision.held_in_after} "
            f"out={decision.held_out_before:.2f}->{decision.held_out_after} "
            f"reasons={list(decision.reasons)}"
        )
    print(f"active harness: v{registry.active.version} {registry.active.digest}")
    print(f"stored accepted versions: {len(registry.versions)}")
    print(f"audited decisions: {len(registry.decisions)}")


if __name__ == "__main__":
    main()
