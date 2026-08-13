"""Build a local rough-terrain MJCF for the MuJoCo G1 env (heightfield ground)."""
import struct
import os
import re

import numpy as np
from scipy.ndimage import gaussian_filter

REPO = r"D:\GR00T-WholeBodyControl"
SCENE_DIR = os.path.join(
    REPO, "gear_sonic/data/robot_model/model_data/g1"
)
OUT = os.path.join(SCENE_DIR, "scene_43dof_rough.xml")
SRC = os.path.join(SCENE_DIR, "scene_43dof.xml")

# terrain: 40x40 m at 0.1 m resolution, amplitude +-0.06 m (like Isaac 0.06)
SIZE_M = 40.0
RES = 0.1
AMP = 0.06
SEED = 0
BASE_Z = 0.5  # MuJoCo 3.x requires hfield size[3] (base_z) strictly positive


def build(amp: float = AMP, seed: int = SEED, coarse: float = 0.4, sigma: float = 1.2):
    """Generate a rough hfield (+-amp), write the .bin and scene XML.

    Returns the raw height array (meters), whose physical heights are exactly
    reproduced by the hfield surface (geom pos offsets the required base_z).
    """
    rng = np.random.default_rng(seed)
    n_low = int(SIZE_M / coarse)
    low = rng.uniform(-amp, amp, (n_low, n_low))
    n = int(SIZE_M / RES)
    xx = np.linspace(0, 1, n)
    yy = np.linspace(0, 1, n)
    from scipy.interpolate import RectBivariateSpline

    spl = RectBivariateSpline(np.linspace(0, 1, n_low), np.linspace(0, 1, n_low), low)
    h = spl(xx, yy)
    h = gaussian_filter(h, sigma=sigma)  # slight smoothing
    h -= h.mean()
    # force the center cell to ~0 so the robot spawns on level ground
    h[n // 2 - 2 : n // 2 + 2, n // 2 - 2 : n // 2 + 2] = 0.0
    np.save(r"C:\Users\zyz\Documents\gr00t\apt_g1\outputs\rough_h.npy", h)

    hmin = float(h.min())
    hmax = float(h.max())
    data = ((h - hmin) / (hmax - hmin)).astype(np.float64)

    data_file = "rough_hfield.bin"
    mesh_dir = os.path.join(SCENE_DIR, "meshes")
    # MuJoCo 3.x binary hfield: int32 nrow, int32 ncol, then float32[nrow*ncol]
    # (heights normalized to [0, 1]; physical height is set by size[2]).
    os.makedirs(mesh_dir, exist_ok=True)
    with open(os.path.join(mesh_dir, data_file), "wb") as f:
        f.write(struct.pack("ii", n, n))
        f.write(data.astype(np.float32).tobytes())

    xml = open(SRC, encoding="utf-8").read()
    if "<hfield" in xml:
        raise SystemExit("source xml already has hfield")

    hfield_asset = (
        f'<hfield name="rough" size="{SIZE_M/2} {SIZE_M/2} {hmax-hmin:.4f} {BASE_Z}" '
        f'nrow="{n}" ncol="{n}" file="{data_file}" content_type="bin"/>\n'
    )
    if "<asset>" in xml:
        xml = xml.replace("<asset>", "<asset>\n" + hfield_asset, 1)
    else:
        raise SystemExit("no <asset> tag found")

    # replace the floor plane geom with a heightfield geom
    pat = re.compile(r'<geom name="floor"[^>]*/>')
    replacement = (
        f'<geom name="floor" type="hfield" hfield="rough" '
        # MuJoCo 3.x collision surface = geom_pos.z + size[2]*data;
        # base_z (size[3]) is parser-positive but not added to collision.
        f'pos="0 0 {hmin:.4f}" material="groundplane"/>'
    )
    xml2, nsub = pat.subn(replacement, xml)
    assert nsub == 1, f"floor geom replacements: {nsub}"

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(xml2)
    print(
        "saved", OUT, "n=", n, "amp", amp, "hmin", round(hmin, 3),
        "hmax", round(hmax, 3),
    )
    return h


if __name__ == "__main__":
    build()
