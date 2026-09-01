"""Run the minimal harness evolution example."""

from harness_lab import Proposal, build_demo_registry


def main() -> None:
    registry = build_demo_registry()
    proposals = [
        Proposal(
            name="fix observed failures",
            add_features=frozenset({"write_artifact", "retry_transient"}),
        ),
        Proposal(
            name="remove safety for speed",
            remove_features=frozenset({"ask_before_destructive"}),
            add_features=frozenset({"fast_shell"}),
        ),
        Proposal(
            name="turn off verifier",
            immutable_updates={"verifier": "disabled"},
        ),
    ]

    print(f"immutable evaluator: {registry.evaluator_digest}")
    print(f"initial harness: v{registry.active.version} {registry.active.digest}")
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


if __name__ == "__main__":
    main()
