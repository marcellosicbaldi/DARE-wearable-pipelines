"""Walking-bout event counts, separate from averages of stride parameters."""
import numpy as np


def add_bout_event_counts(bouts, initial_contacts, turns):
    """Count distinct ICs between a bout's endpoint ICs, including both ends.

    Turns belong to the bout containing their midpoint (half-open interval).
    This avoids counting one turn twice when it overlaps adjacent bouts.
    All event positions must be in the same sample coordinates as the bouts.
    """
    out = bouts.copy()
    contacts = np.sort(np.unique(initial_contacts["ic"].to_numpy(dtype=float)))
    starts = out.start.to_numpy(dtype=float)
    ends = out.end.to_numpy(dtype=float)
    out["n_raw_initial_contacts"] = (
        np.searchsorted(contacts, ends, side="right")
        - np.searchsorted(contacts, starts, side="left")
    ).astype(int)
    centers = np.sort(((turns.start + turns.end) / 2).to_numpy(dtype=float))
    out["n_turns"] = (np.searchsorted(centers, ends, side="left")
                      - np.searchsorted(centers, starts, side="left")).astype(int)
    return out
