"""Geometries with methyl-type tops at chosen turns, for the top-collapse tests.

Built from ideal bond lengths and angles rather than optimized, so that each
methyl's turn is exactly the one asked for.
"""

import numpy as np


def methyl(anchor: np.ndarray, c: np.ndarray, phase: float) -> np.ndarray:
    """Three tetrahedral hydrogens on ``c``, bonded to ``anchor``, turned ``phase`` degrees.

    Phase 0 puts one hydrogen out of the xy plane, perpendicular to it; 90, in it.
    """
    axis = (c - anchor) / np.linalg.norm(c - anchor)
    u = np.array([0.0, 0.0, 1.0]) - axis[2] * axis
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    beta = np.radians(180.0 - 109.47)
    turns = np.radians(phase + np.array([0.0, 120.0, 240.0]))
    return np.array(
        [
            c + 1.09 * (np.cos(beta) * axis + np.sin(beta) * (np.cos(t) * u + np.sin(t) * v))
            for t in turns
        ]
    )


def ring(
    methyls: dict[int, float], carboxyl: bool = False, twist: float = 0.0
) -> tuple[np.ndarray, list[str]]:
    """A benzene ring in the xy plane with methyls (ring position: phase) and an optional
    COOH at position 3, para to a methyl at 0, turned ``twist`` degrees out of the plane."""
    angles = np.radians(np.arange(6) * 60.0)
    ring = np.stack([1.39 * np.cos(angles), 1.39 * np.sin(angles), np.zeros(6)], axis=1)
    out, symbols = list(ring), ["C"] * 6
    for k in range(6):
        radial = ring[k] / np.linalg.norm(ring[k])
        if k in methyls:
            c = ring[k] + 1.51 * radial
            out += [c, *methyl(ring[k], c, methyls[k])]
            symbols += ["C", "H", "H", "H"]
        elif carboxyl and k == 3:
            c = ring[k] + 1.49 * radial
            o1 = c + 1.21 * np.array([np.cos(np.radians(240.0)), np.sin(np.radians(240.0)), 0.0])
            o2 = c + 1.35 * np.array([np.cos(np.radians(120.0)), np.sin(np.radians(120.0)), 0.0])
            h = o2 + 0.97 * np.array([-1.0, 0.0, 0.0])
            t = np.radians(twist)  # about the ring-carboxyl bond, the x axis
            turn = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
            out += [c, *((np.array([o1, o2, h]) - c) @ turn.T + c)]
            symbols += ["C", "O", "O", "H"]
        else:
            out.append(ring[k] + 1.08 * radial)
            symbols.append("H")
    return np.array(out), symbols


def acetonitrile(phase: float) -> tuple[np.ndarray, list[str]]:
    n, c1, c2 = np.array([0.0, 0, 0]), np.array([1.16, 0, 0]), np.array([2.62, 0, 0])
    return np.vstack([n, c1, c2, methyl(c1, c2, phase)]), ["N", "C", "C", "H", "H", "H"]


def ethane(phase_a: float, phase_b: float) -> tuple[np.ndarray, list[str]]:
    a, b = np.array([0.0, 0, 0]), np.array([1.53, 0, 0])
    return (
        np.vstack([a, b, methyl(b, a, phase_a), methyl(a, b, phase_b)]),
        ["C", "C", "H", "H", "H", "H", "H", "H"],
    )


def methanol(phase: float) -> tuple[np.ndarray, list[str]]:
    """CH3-O-H in the xy plane, the methyl turned ``phase`` degrees (as `methyl`)."""
    c, o = np.array([0.0, 0, 0]), np.array([1.43, 0, 0])
    h = o + 0.96 * np.array([np.cos(np.radians(71.0)), np.sin(np.radians(71.0)), 0.0])
    return np.vstack([c, o, h, methyl(o, c, phase)]), ["C", "O", "H", "H", "H", "H"]


def trihalo_ethane(halogen: str, bond: float) -> tuple[np.ndarray, list[str]]:
    """CH3-CX3, staggered, with C-X bonds ``bond`` Angstrom long."""
    a, b = np.array([0.0, 0, 0]), np.array([1.52, 0, 0])
    xs = b + (methyl(a, b, 60.0) - b) * bond / 1.09
    return np.vstack([a, b, methyl(b, a, 0.0), xs]), ["C", "C", "H", "H", "H"] + [halogen] * 3
