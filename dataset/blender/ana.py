import sys
import numpy as np
import trimesh
from pathlib import Path
from plyfile import PlyData
from sklearn.neighbors import KDTree

target = Path(sys.argv[1])

if target.is_file():
    ply_files = [target]
elif target.is_dir():
    ply_files = list(target.glob("*.ply"))
else:
    raise FileNotFoundError(f"Introuvable : {target}")

if not ply_files:
    raise FileNotFoundError(f"Aucun .ply trouvé dans {target}")

print(f"{len(ply_files)} fichier(s) trouvé(s)")

_OCC_CHUNK          = 4096
_N_NEIGHBORS        = 64
_PROXIMITY_RADIUS   = 2.0
_PROXIMITY_COS_THRESH = 0.3

all_log_height          = []
all_log_area            = []
all_occluded            = []
all_cos_angles          = []
all_grazing_angle       = []
all_elevation_angle     = []
all_obstacle_proximity  = []
all_slope_discontinuity = []

for path in ply_files:
    print(f"  → {path.name}", end="  ")
    ply   = PlyData.read(path)
    verts = np.stack([
        ply["vertex"]["x"],
        ply["vertex"]["y"],
        ply["vertex"]["z"],
    ], axis=1).astype(np.float32)

    face_data = ply["face"]["vertex_indices"]

    # ── Triangulation : quads → 2 triangles ─────────────────────────────────
    tris = []
    n_skipped = 0
    for f in face_data:
        f = list(f)
        if len(f) == 3:
            tris.append(f)
        elif len(f) == 4:
            tris.append([f[0], f[1], f[2]])
            tris.append([f[0], f[2], f[3]])
        else:
            n_skipped += 1

    if not tris:
        print("  SKIP (aucun triangle)")
        continue

    faces = np.array(tris, dtype=np.int32)
    print(f"{len(faces)} triangles  ({n_skipped} faces ignorées)")

    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]

    centroid = ((v0 + v1 + v2) / 3.0).astype(np.float32)   # (M, 3)

    # ── Normales unitaires ───────────────────────────────────────────────────
    cross  = np.cross(v1 - v0, v2 - v0).astype(np.float32)
    norm2  = np.linalg.norm(cross, axis=1)
    safe_n = np.where(norm2 > 1e-12, norm2, 1.0)
    normal = (cross / safe_n[:, None]).astype(np.float32)   # (M, 3)

    # ── Source synthétique ───────────────────────────────────────────────────
    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    src = np.array(
        [(bbox_min[0] + bbox_max[0]) * 0.5,
         (bbox_min[1] + bbox_max[1]) * 0.5,
         bbox_max[2] + 100.0],
        dtype=np.float32,
    )

    # ── Vecteurs source → centroïde ──────────────────────────────────────────
    vec        = src[None, :] - centroid                    # (M, 3)
    dist_raw   = np.linalg.norm(vec, axis=1)
    dist       = np.where(dist_raw > 1e-6, dist_raw, 1e-6).astype(np.float32)
    dir_to_src = (vec / dist[:, None]).astype(np.float32)   # (M, 3)

    # ── Features existantes ──────────────────────────────────────────────────
    z_min      = verts[:, 2].min()
    height_agl = centroid[:, 2] - z_min
    all_log_height.append(np.log10(height_agl + 1.0))

    area = 0.5 * np.linalg.norm(cross, axis=1)
    all_log_area.append(np.log10(area + 1.0))

    # ── occluded ─────────────────────────────────────────────────────────────
    mesh    = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    vec_d   = vec.astype(np.float64)
    dist_d  = dist.astype(np.float64)
    dirs_d  = vec_d / dist_d[:, None]
    eps     = 1e-3
    origins = centroid.astype(np.float64) + dirs_d * eps

    occ = np.zeros(len(faces), dtype=np.uint8)
    with np.errstate(divide="ignore", invalid="ignore"):
        for start in range(0, len(faces), _OCC_CHUNK):
            end  = min(start + _OCC_CHUNK, len(faces))
            hits, ray_idx, _ = mesh.ray.intersects_location(
                ray_origins    = origins[start:end],
                ray_directions = dirs_d[start:end],
                multiple_hits  = False,
            )
            if len(ray_idx) == 0:
                continue
            hit_dist = np.linalg.norm(
                hits - origins[start:end][ray_idx], axis=1
            )
            mask = hit_dist < (dist_d[start:end][ray_idx] - 2.0 * eps)
            occ[start + ray_idx[mask]] = 1
    all_occluded.append(occ.astype(np.float32))

    # ── [10] cos_angles ──────────────────────────────────────────────────────
    cos_angles = np.einsum("ij,ij->i", normal, dir_to_src).clip(-1.0, 1.0)
    all_cos_angles.append(cos_angles)

    # ── [11] grazing_angle ───────────────────────────────────────────────────
    facing_source = (cos_angles > 0).astype(np.float32)
    grazing_angle = (1.0 - np.abs(cos_angles)) * facing_source
    all_grazing_angle.append(grazing_angle)

    # ── [12] elevation_angle ─────────────────────────────────────────────────
    # Angle vertical depuis la source vers le centroïde, normalisé par π/2
    dz            = src[2] - centroid[:, 2]
    horiz_dist    = np.sqrt(vec[:, 0] ** 2 + vec[:, 1] ** 2)
    elevation_angle = (
        np.arctan2(np.abs(dz), np.maximum(horiz_dist, 1e-6)) / (np.pi / 2.0)
    ).astype(np.float32)
    all_elevation_angle.append(elevation_angle)

    # ── [13] obstacle_proximity ──────────────────────────────────────────────
    # Pour chaque centroïde, compte les voisins dans _PROXIMITY_RADIUS
    # dont cos_angle > seuil ET plus proches de la source que le point courant
    tree      = KDTree(centroid)
    idxs_list = tree.query_radius(centroid, r=_PROXIMITY_RADIUS)

    obstacle_proximity = np.zeros(len(faces), dtype=np.float32)
    for i, nbr_idxs in enumerate(idxs_list):
        if len(nbr_idxs) == 0:
            continue
        nbr_cos  = cos_angles[nbr_idxs]
        nbr_dist = dist[nbr_idxs]
        mask     = (nbr_cos > _PROXIMITY_COS_THRESH) & (nbr_dist < dist[i])
        obstacle_proximity[i] = mask.sum() / max(len(nbr_idxs), 1)
    all_obstacle_proximity.append(obstacle_proximity)

    # ── [14] slope_discontinuity ─────────────────────────────────────────────
    k           = min(_N_NEIGHBORS + 1, len(faces))
    _, knn_idxs = tree.query(centroid, k=k)
    knn_idxs    = knn_idxs[:, 1:]                          # retire soi-même

    nbr_normals         = normal[knn_idxs]                 # (M, k, 3)
    cos_with_nbrs       = np.einsum(
        "ij,ikj->ik", normal, nbr_normals
    ).clip(-1.0, 1.0)                                      # (M, k)
    slope_discontinuity = cos_with_nbrs.var(axis=1).astype(np.float32)
    all_slope_discontinuity.append(slope_discontinuity)

# ── Concaténation ────────────────────────────────────────────────────────────
log_height          = np.concatenate(all_log_height)
log_area            = np.concatenate(all_log_area)
occluded            = np.concatenate(all_occluded)
cos_angles_all      = np.concatenate(all_cos_angles)
grazing_angle_all   = np.concatenate(all_grazing_angle)
elevation_angle_all = np.concatenate(all_elevation_angle)
obstacle_prox_all   = np.concatenate(all_obstacle_proximity)
slope_disc_all      = np.concatenate(all_slope_discontinuity)

# ── Affichage ─────────────────────────────────────────────────────────────────
def _print_stat(name: str, arr: np.ndarray) -> tuple[float, float]:
    m, s = arr.mean(), arr.std()
    print(f"\n{name}")
    print(f"  mean={m:.4f}  std={s:.4f}")
    print(f"  min={arr.min():.4f}  max={arr.max():.4f}")
    print(f"  p5={np.percentile(arr, 5):.4f}  p95={np.percentile(arr, 95):.4f}")
    return float(m), float(s)

print("\n" + "="*60)
print("STATS COMPLÈTES")
print("="*60)

lh_m,  lh_s  = _print_stat("log_height",          log_height)
la_m,  la_s  = _print_stat("log_area",             log_area)
oc_m,  oc_s  = _print_stat("occluded",             occluded)
ca_m,  ca_s  = _print_stat("cos_angles",           cos_angles_all)
ga_m,  ga_s  = _print_stat("grazing_angle",        grazing_angle_all)
ea_m,  ea_s  = _print_stat("elevation_angle",      elevation_angle_all)
op_m,  op_s  = _print_stat("obstacle_proximity",   obstacle_prox_all)
sd_m,  sd_s  = _print_stat("slope_discontinuity",  slope_disc_all)

print(f"""
{'='*60}
Copy-paste dans NODE_STATS :

    "log_height":           ({lh_m:.4f}, {lh_s:.4f}),
    "log_area":             ({la_m:.4f}, {la_s:.4f}),
    "occluded":             ({oc_m:.4f}, {oc_s:.4f}),
    "cos_angles":           ({ca_m:.4f}, {ca_s:.4f}),
    "grazing_angle":        ({ga_m:.4f}, {ga_s:.4f}),
    "elevation_angle":      ({ea_m:.4f}, {ea_s:.4f}),
    "obstacle_proximity":   ({op_m:.4f}, {op_s:.4f}),
    "slope_discontinuity":  ({sd_m:.4f}, {sd_s:.4f}),
{'='*60}
""")
