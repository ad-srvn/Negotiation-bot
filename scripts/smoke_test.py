from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.domain import make_mechanism, make_ufuns
from src.negotiators import strategy_registry


def main() -> None:
    registry = strategy_registry()
    session = make_mechanism(n_steps=20)
    buyer_ufun, seller_ufun = make_ufuns("A", session.outcome_space)
    session.add(registry["BoulwareTBNegotiator"](name="buyer"), ufun=buyer_ufun)
    session.add(registry["LinearTBNegotiator"](name="seller"), ufun=seller_ufun)
    state = session.run()
    print("Smoke test complete.")
    print(f"Agreement: {state.agreement}")
    print(f"Rounds: {state.step}")


if __name__ == "__main__":
    main()
