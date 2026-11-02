"""Spatial image converters (2026-11-01): depth, correspondence, multi-view, counting and spatial sources turned into
the board's BLINK and CV-Bench question forms (see vision_mix for the item fields).

  nyu_depth  depth           NYU Depth v2 train frames (the repo's 2 tars, ~70 video scenes; NYU_PER_SCENE frames
                             spread over each). BLINK Relative_Depth: two circled points A and B, the gold "closer"
                             (or, 1 in 4, "farther") by DEPTH_GAP in both planar and ray depth, each point with
                             valid depth and no depth edge within two ring radii. The h5 files are read by h5_chunked
                             (no h5py in the venv; equal to h5py's arrays on sample frames).
  hypersim   depth           Same form from Hypersim's rendered planar depth. Official train scenes only
                             (data/raw/extracted/hypersim/metadata_images_split_scene_v1.csv, from apple/ml-hypersim;
                             Omni3D, the source of CV-Bench Distance, tests on the official test scenes), at most
                             HYPERSIM_PER_CAM frames per camera. Only frames whose tonemap.jpg and depth npz are both
                             present and readable are used, so a re-run picks up the rest of the download.
  hpatches   correspondence  BLINK Visual_Correspondence: REF on a textured point of image 1, four circled points on
                             image k, one the image of REF under the ground-truth H_1_k, the others 12% of the short
                             side apart. BLINK's Visual_Correspondence images are HPatches frames, so every sequence
                             with a frame like a board image (60 of 116) is left out.
  spair71k   correspondence  BLINK Semantic_Correspondence from trn pairs (human keypoints): REF a source keypoint,
                             the candidates its target keypoint and three other keypoints of the target object.
                             Images like a board image are left out (BLINK uses SPair test images; 3 trn images match).
  co3d_tv    multi-image     BLINK Multi-view_Reasoning from the CO3D v2 tv train sequences (21; see co3d_tv for the
                             pose checks and the left/right rule). Licence CC BY-NC 4.0.
  sat        spatial         SAT train split (static is a subset of it): its own questions and options, subtask the
                             question type, with the "other" type split by template (count, depth, relation, ...).
                             Equal shares per subtask, then per gold position where order is kept. Yes/no become
                             ["Yes", "No"] and drawn letters keep order; other options are shuffled by the trainer.
  fsc147     counting        FSC-147 train split, human point counts. "How many <class> are in the image?" with four
                             ascending options: consecutive numbers up to FSC_EXACT objects, else ("Approximately
                             how many") a geometric ladder with ratio 1.35-1.6, so no distractor is near the gold.
                             Gold position balanced. Ambiguous class names are renamed or skipped.
  fsc147_crop counting       FSC-147 train images cropped to a window holding 1..FSC_CROP_MAX of the human points,
                             counts in equal shares (11-02: the fsc147 pool is 88% counts over 11, and CountBench's
                             5-7 items fell from 79.8% to 57.6% after training on it). No point lies within FSC_EDGE
                             exemplar sides of the window's edge, so no object is cut. Up to FSC_CROPS_PER_IMAGE
                             windows per image, with different counts. Upscaled to FSC_CROP_SIDE on the short side;
                             four consecutive options from 0 up.

Marks follow BLINK: a red ring, its letter (or REF) in white on a black box above it, sized to the image.
Skipped: nuscenes_qa. Its questions are about the whole 360-degree scene from 3D boxes (six cameras and LiDAR;
objects can be out of view or hidden in every camera), statuses such as moving or parked need time, and it is a
1,534-row mini set.
"""
from __future__ import annotations

import contextlib
import csv
import gzip
import io
import itertools
import json
import math
import re
import struct
import tarfile
import zipfile
import zlib
from collections import defaultdict

import numpy as np

from qwen_decision.v3.vision_mix import BOARD, ROOT, item, norm, shares

RAW = ROOT / "data/raw"
EXTRACTED = RAW / "extracted"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
DEPTH_GAP = 1.15      # far / near depth ratio for a relative-depth item
BOARD_SIM = 0.9       # 32x32 grey correlation at which an image counts as a board image
NYU_PER_SCENE = 40
FSC_EXACT = 12
FSC_CROP_MAX = 12     # fsc147_crop counts 1..FSC_CROP_MAX
FSC_CROP_SIDE = 448   # short side a crop is upscaled to (the 384-high images give crops of ~100-380 px)
FSC_EDGE = 0.6        # no point within this many exemplar sides of a crop's edge
FSC_CROPS_PER_IMAGE = 3  # with different counts

DEPTH_Q = "Two points are circled on the image, labeled by A and B beside each circle. Which point is {} the camera?"
VC_Q = ("A point is circled on the first image, labeled with REF. We change the camera position or lighting and shoot "
        "the second image. You are given multiple red-circled points on the second image, choices of \"A, B, C, D\" "
        "are drawn beside each circle. Which point on the second image corresponds to the point in the first image? "
        "Select from the following options.\n(A) Point A\n(B) Point B\n(C) Point C\n(D) Point D")
SC_Q = ("Humans can find corresponding points for different objects in the same category. For instance, if there are "
        "images of two different cats, then the left ear tip of one cat corresponds to the left ear tip of the other "
        "cat, and the right front paw of one cat corresponds to the right front paw of the other cat.\nGiven the "
        "following two images, a reference point is annotated on the first image, labeled with REF. You are given "
        "multiple red-circled points on the second image, choices of \"A, B, C, D\" are drawn beside each circle. "
        "Select between the choices on the second image and find the corresponding point for the reference point. "
        "Which point is corresponding to the reference point?")
POINTS = ["Point A", "Point B", "Point C", "Point D"]


# --- shared helpers ---------------------------------------------------------------------------------------------

def marker_scale(w: int, h: int) -> tuple[int, int, int]:
    """Ring radius, ring width and label font size, scaled like BLINK's marks (r 6 at 375 px, 11 at 1024 px)."""
    s = min(w, h)
    r = max(6, round(0.011 * s))
    return r, max(2, round(r / 3.5)), max(12, round(0.021 * s))


def draw_points(im, points, labels):
    """BLINK-style marks: a red ring on each point, its label in white on a black box just above (or below)."""
    from decision_index.vision.convert import font
    from PIL import ImageDraw

    im = im.convert("RGB")
    d = ImageDraw.Draw(im)
    w, h = im.size
    r, lw, fs = marker_scale(w, h)
    f = font(fs, FONT)
    for (x, y), lab in zip(points, labels):
        d.ellipse([x - r, y - r, x + r, y + r], outline=(255, 0, 0), width=lw)
        x0, y0, x1, y1 = d.textbbox((0, 0), lab, font=f)
        pad = max(2, fs // 8)
        bw, bh = x1 - x0 + 2 * pad, y1 - y0 + 2 * pad
        bx = min(max(0, x - bw / 2), w - bw)
        by = y - r - r // 2 - bh
        if by < 0:
            by = y + r + r // 2
        d.rectangle([bx, by, bx + bw, by + bh], fill=(0, 0, 0))
        d.text((bx + pad - x0, by + pad - y0), lab, fill=(255, 255, 255), font=f)
    return im


def thumb(im) -> np.ndarray:
    from PIL import Image

    g = np.asarray(im.convert("L").resize((32, 32), Image.BILINEAR), dtype=np.float32).ravel()
    g -= g.mean()
    return g / (np.linalg.norm(g) + 1e-6)


def board_thumbs(pattern: str) -> np.ndarray:
    """Thumbs of the board's BLINK images whose file name matches pattern (glob)."""
    from PIL import Image

    paths = sorted((BOARD / "full_blink/img").glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no board images match {pattern}")
    return np.stack([thumb(Image.open(p)) for p in paths])


def far_apart(cands, k: int, keep: list, min_d: float, rng) -> list | None:
    """k points from cands (shuffled), each at least min_d from the others and from `keep`."""
    cands = list(cands)
    rng.shuffle(cands)
    out = []
    for p in cands:
        if all(math.dist(p, q) >= min_d for q in keep + out):
            out.append(p)
            if len(out) == k:
                return out
    return None


def box_sum(a: np.ndarray, k: int) -> np.ndarray:
    """Sum over the k x k window centred on each pixel (zero outside the image), same shape as a; k odd."""
    p = k // 2
    c = np.pad(np.pad(a.astype(np.float64), p), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]


def depth_pair(z: np.ndarray, focal: tuple[float, float, float, float], rng) -> tuple | None:
    """Two points (x, y), near first: no invalid depth or depth edge near either, apart, far/near >= DEPTH_GAP in
    planar and in ray depth (z times the ray length per unit z), so "closer to the camera" holds either way."""
    h, w = z.shape
    r = marker_scale(w, h)[0]
    z = np.where(np.isfinite(z) & (z > 0), z, 0).astype(np.float32)
    jump = np.zeros_like(z)  # relative depth step to the next pixel; a large one is an occlusion edge
    jump[:, :-1] = np.abs(np.diff(z, axis=1)) / np.maximum(z[:, :-1], 1e-3)
    jump[:-1] = np.maximum(jump[:-1], np.abs(np.diff(z, axis=0)) / np.maximum(z[:-1], 1e-3))
    k = 4 * r + 1  # the ring and a ring's width around it, in case depth and colour are slightly misaligned
    ok = box_sum((z <= 0) | (jump >= 0.03), k) < 0.5
    m = round(0.08 * min(w, h)) + r
    ok[:m], ok[-m:], ok[:, :m], ok[:, -m:] = False, False, False, False
    ys, xs = np.nonzero(ok)
    if len(ys) < 200:
        return None
    fx, fy, cx, cy = focal
    for _ in range(200):
        i, j = rng.randrange(len(ys)), rng.randrange(len(ys))
        a, b = (int(xs[i]), int(ys[i])), (int(xs[j]), int(ys[j]))
        if math.dist(a, b) < 0.12 * min(w, h):
            continue
        za, zb = float(z[a[1], a[0]]), float(z[b[1], b[0]])
        ra, rb = (math.sqrt(1 + ((p[0] - cx) / fx) ** 2 + ((p[1] - cy) / fy) ** 2) for p in (a, b))
        if za > zb:
            a, b, za, zb, ra, rb = b, a, zb, za, rb, ra
        if zb >= DEPTH_GAP * za and zb * rb >= DEPTH_GAP * za * ra:
            return a, b
    return None


def depth_item(source: str, sid: str, saver, im, near, far, rng, subtask: str) -> dict | None:
    """BLINK Relative_Depth item; which point is A, and the closer/farther wording, are random."""
    farther = rng.random() < 0.25
    pts = [near, far] if rng.random() < 0.5 else [far, near]
    target = far if farther else near
    word = "farther" if farther else "closer"
    img = saver.pil(draw_points(im, pts, ["A", "B"]))
    return item(source, sid, [img], DEPTH_Q.format("farther from" if farther else "closer to"),
                [f"A is {word}", f"B is {word}"], pts.index(target), subtask + ("_farther" if farther else ""),
                "program", ordered=True)


def h5_chunked(raw: bytes, shapes: dict) -> dict:
    """{name: array} for gzip-chunked HDF5 datasets, read from the leaves of their v1 chunk B-trees.

    shapes maps a dataset name to (shape, chunk shape, dtype); datasets are told apart by rank, so each shape must
    have a distinct rank. Raises when a dataset's chunks are not all found.
    """
    found = {k: {} for k in shapes}
    for m in re.finditer(b"TREE\x01\x00", raw):  # chunk-type leaf nodes
        o, used = m.start(), struct.unpack_from("<H", raw, m.start() + 6)[0]
        for name, (shape, chunk, _) in shapes.items():
            nd = len(shape) + 1
            step, got = 16 + 8 * nd, []
            if o + 24 + used * step > len(raw):
                continue
            for e in range(used):
                k = o + 24 + e * step
                size, mask = struct.unpack_from("<II", raw, k)
                offs = struct.unpack_from(f"<{nd}Q", raw, k + 8)
                addr = struct.unpack_from("<Q", raw, k + 8 + 8 * nd)[0]
                if mask or offs[-1] or addr + size > len(raw) or any(
                        a % c or a >= s for a, c, s in zip(offs, chunk, shape)):
                    got = None
                    break
                got.append((offs[:-1], addr, size))
            for offs, addr, size in got or ():
                found[name][offs] = (addr, size)
    out = {}
    for name, (shape, chunk, dtype) in shapes.items():
        need = math.prod(-(-s // c) for s, c in zip(shape, chunk))
        if len(found[name]) != need:
            raise ValueError(f"{name}: {len(found[name])} of {need} chunks")
        arr = np.empty(shape, dtype)
        for offs, (addr, size) in found[name].items():
            block = np.frombuffer(zlib.decompress(raw[addr:addr + size]), dtype).reshape(chunk)
            sl = tuple(slice(a, min(a + c, s)) for a, c, s in zip(offs, chunk, shape))
            arr[sl] = block[tuple(slice(0, x.stop - x.start) for x in sl)]
        out[name] = arr
    return out


# --- depth ------------------------------------------------------------------------------------------------------

NYU = RAW / "sayakpaul__nyu_depth_v2@50579a0b5914/data"
NYU_H5 = {"depth": ((480, 640), (60, 80), "<f4"), "rgb": ((3, 480, 640), (1, 60, 80), "u1")}
NYU_FOCAL = (518.86, 519.47, 325.58, 253.74)  # Kinect colour camera (NYU toolbox)


def nyu_depth(saver, cap, rng):
    from PIL import Image

    with contextlib.ExitStack() as stack:
        tars = {p: stack.enter_context(tarfile.open(p)) for p in sorted(NYU.glob("train-*.tar"))}
        scenes = defaultdict(list)
        for p, t in tars.items():
            for m in t.getmembers():
                if m.isfile() and m.name.endswith(".h5"):
                    scenes[m.name.split("/")[1]].append((p, m))
        picked = []
        for name in sorted(scenes):  # NYU_PER_SCENE frames spread evenly over each scene's video
            frames = sorted(scenes[name], key=lambda x: x[1].name)
            step = max(1, len(frames) / NYU_PER_SCENE)
            start = rng.random() * step
            picked += [frames[int(start + i * step)] for i in range(min(NYU_PER_SCENE, len(frames)))]
        rng.shuffle(picked)
        for p, m in picked:
            a = h5_chunked(tars[p].extractfile(m).read(), NYU_H5)
            pair = depth_pair(a["depth"], NYU_FOCAL, rng)
            if pair is None:
                continue
            im = Image.fromarray(np.ascontiguousarray(a["rgb"].transpose(1, 2, 0)))
            yield depth_item("nyu_depth", m.name.removeprefix("train/").removesuffix(".h5"), saver, im, *pair, rng,
                             "relative_depth")


HYPERSIM = RAW / "ritianyu__Hypersim@1b10f85c2774"
HYPERSIM_SPLIT = EXTRACTED / "hypersim/metadata_images_split_scene_v1.csv"
HYPERSIM_PER_CAM = 12
HYPERSIM_FOCAL = (886.81, 886.81, 512.0, 384.0)  # 1024 x 768 at 60 degrees horizontal field of view


def hypersim(saver, cap, rng):
    from PIL import Image

    frames = defaultdict(list)
    with HYPERSIM_SPLIT.open() as f:
        for r in csv.DictReader(f):
            if r["split_partition_name"] == "train" and r["included_in_public_release"] == "True":
                frames[(r["scene_name"], r["camera_name"])].append(int(r["frame_id"]))
    picked = []
    for (scene, cam), ids in sorted(frames.items()):
        base = HYPERSIM / scene / "images"
        if not base.is_dir():
            continue
        have = []
        for i in sorted(ids):
            jpg = base / f"scene_{cam}_final_preview/frame.{i:04d}.tonemap.jpg"
            npz = base / f"scene_{cam}_geometry_hdf5/frame.{i:04d}.depth_meters_plane.npz"
            if jpg.is_file() and npz.is_file():
                have.append((f"{scene}/{cam}/{i:04d}", jpg, npz))
        picked += rng.sample(have, min(HYPERSIM_PER_CAM, len(have)))
    rng.shuffle(picked)
    for sid, jpg, npz in picked:
        try:
            with np.load(npz) as z:
                depth = z["data"].astype(np.float32)
            im = Image.open(jpg)
            im.load()
        except (OSError, ValueError, KeyError, zipfile.BadZipFile):
            continue  # still being written
        if im.size != depth.shape[::-1]:
            continue
        pair = depth_pair(depth, HYPERSIM_FOCAL, rng)
        if pair is not None:
            yield depth_item("hypersim", sid, saver, im, *pair, rng, "relative_depth")


# --- correspondence ---------------------------------------------------------------------------------------------

HPATCHES = RAW / "vbalnt__hpatches@594f211e80b7/hpatches-sequences-release.zip"
HPATCHES_PER_PAIR = 4


def textured(im, n: int, margin: int, rng) -> list[tuple[int, int]]:
    """Up to n random points among the top 30% of local gradient energy, at least margin from the border."""
    g = np.asarray(im.convert("L"), dtype=np.float32)
    e = np.zeros_like(g)
    e[1:-1, 1:-1] = np.abs(g[2:, 1:-1] - g[:-2, 1:-1]) + np.abs(g[1:-1, 2:] - g[1:-1, :-2])
    box = box_sum(e, 9)
    box[:margin], box[-margin:], box[:, :margin], box[:, -margin:] = 0, 0, 0, 0
    ys, xs = np.nonzero(box > np.quantile(box[box > 0], 0.7)) if (box > 0).any() else ((), ())
    idx = rng.sample(range(len(ys)), min(n, len(ys)))
    return [(int(xs[i]), int(ys[i])) for i in idx]


def hpatches(saver, cap, rng):
    from decision_index.vision.convert import Letters
    from PIL import Image

    board = board_thumbs("val_Visual_Correspondence_*")
    z = zipfile.ZipFile(HPATCHES)
    seqs = defaultdict(dict)
    for n in z.namelist():
        parts = n.split("/")
        if len(parts) == 3 and parts[2]:
            seqs[parts[1]][parts[2]] = n
    letters = Letters(rng)
    pairs = []
    for s, files in sorted(seqs.items()):
        ims = {k: Image.open(io.BytesIO(z.read(files[f"{k}.ppm"]))).convert("RGB") for k in range(1, 7)}
        if (np.stack([thumb(im) for im in ims.values()]) @ board.T).max() > BOARD_SIM:
            continue  # a board sequence
        for k in range(2, 7):
            H = np.array([[float(v) for v in line.split()] for line in z.read(files[f"H_1_{k}"]).decode().split("\n")
                          if line.strip()])
            pairs.append((s, k, ims[1], ims[k], H))
    rng.shuffle(pairs)
    for s, k, im1, im2, H in pairs:
        w2, h2 = im2.size
        r2 = marker_scale(w2, h2)[0]
        m1, m2 = 6 * marker_scale(*im1.size)[0], 6 * r2
        made = 0
        for ref in textured(im1, 40, m1, rng):
            x, y, w = H @ np.array([ref[0], ref[1], 1.0])
            match = (x / w, y / w)
            if w <= 0 or not (m2 <= match[0] < w2 - m2 and m2 <= match[1] < h2 - m2):
                continue
            others = far_apart(textured(im2, 60, m2, rng), 3, [match], max(0.12 * min(w2, h2), 6 * r2), rng)
            if others is None:
                continue
            pos = letters.pick(4)
            pts = others[:pos] + [match] + others[pos:]
            imgs = [saver.pil(draw_points(im1, [ref], ["REF"])), saver.pil(draw_points(im2, pts, list("ABCD")))]
            yield item("hpatches", f"{s}/1-{k}/{ref[0]},{ref[1]}", imgs, VC_Q, POINTS, pos,
                       "illumination" if s.startswith("i_") else "viewpoint", "program", ordered=True)
            made += 1
            if made == HPATCHES_PER_PAIR:
                break


SPAIR = RAW / "url/spair71k/SPair-71k.tar.gz"


def spair71k(saver, cap, rng):
    from decision_index.vision.convert import Letters
    from PIL import Image

    pairs, images, kps = [], {}, {}
    with tarfile.open(SPAIR) as t:  # one streaming pass; the archive is 230 MB
        for m in t:
            if not m.isfile():
                continue
            p = m.name.split("/")
            if p[1] == "PairAnnotation" and p[2] == "trn":
                pairs.append(json.loads(t.extractfile(m).read()))
            elif p[1] == "JPEGImages":
                images[f"{p[2]}/{p[3]}"] = t.extractfile(m).read()
            elif p[1] == "ImageAnnotation":
                a = json.loads(t.extractfile(m).read())
                kps[f"{p[2]}/{a['filename']}"] = [v for v in a["kps"].values() if v]
    used = {f"{pr['category']}/{n}" for pr in pairs for n in (pr["src_imname"], pr["trg_imname"])}
    board = board_thumbs("val_*_Correspondence_*")
    bad = {k for k in used if (thumb(Image.open(io.BytesIO(images[k]))) @ board.T).max() > BOARD_SIM}
    pairs = [pr for pr in pairs if not {f"{pr['category']}/{pr['src_imname']}",
                                         f"{pr['category']}/{pr['trg_imname']}"} & bad]
    rng.shuffle(pairs)
    letters = Letters(rng)
    for pr in pairs:
        src, trg = f"{pr['category']}/{pr['src_imname']}", f"{pr['category']}/{pr['trg_imname']}"
        im1, im2 = (Image.open(io.BytesIO(images[k])).convert("RGB") for k in (src, trg))
        w2, h2 = im2.size
        r2 = marker_scale(w2, h2)[0]
        inside = [tuple(p) for p in kps[trg] if r2 + 2 <= p[0] < w2 - r2 - 2 and r2 + 2 <= p[1] < h2 - r2 - 2]
        order = list(range(len(pr["src_kps"])))
        rng.shuffle(order)
        for i in order:
            ref, match = tuple(pr["src_kps"][i]), tuple(pr["trg_kps"][i])
            if match not in inside or not all(8 <= c < d - 8 for c, d in zip(ref, im1.size)):
                continue
            others = far_apart([p for p in inside if p != match], 3, [match], max(4 * r2, 0.05 * min(w2, h2)), rng)
            if others is None:
                continue
            pos = letters.pick(4)
            pts = others[:pos] + [match] + others[pos:]
            imgs = [saver.pil(draw_points(im1, [ref], ["REF"])), saver.pil(draw_points(im2, pts, list("ABCD")))]
            yield item("spair71k", f"{pr['pair_id']}/{pr['kps_ids'][i]}", imgs, SC_Q, POINTS, pos, pr["category"],
                       "human", ordered=True)
            break


# --- multi-view -------------------------------------------------------------------------------------------------

CO3D = RAW / "mutou0308__co3dv2@9ba8906d619f/tv.zip"
CO3D_FRAMES = 40      # frames per sequence whose pose is checked
CO3D_PER_SEQ = 60
MV_Q = ("The images are frames from a video. The video is shooting a static scene. The camera is either moving "
        "clockwise (left) or counter-clockwise (right) around the object. The first image is from the beginning of "
        "the video and the second image is from the end. Is the camera moving left or right when shooting the "
        "video? Select from the following options.\n(A) left\n(B) right")


def co3d_points(z, seq: str, rng) -> np.ndarray | None:
    try:
        raw = z.read(f"tv/{seq}/pointcloud.ply")
    except KeyError:
        return None
    h = raw.index(b"end_header\n") + len(b"end_header\n")
    head = raw[:h].decode()
    n = int(re.search(r"element vertex (\d+)", head).group(1))
    if "property float x\nproperty float y\nproperty float z\nproperty uchar red" not in head:
        return None
    pts = np.frombuffer(raw[h:h + 15 * n], dtype=np.dtype([("p", "<f4", 3), ("c", "u1", 3)]))["p"]
    return pts[rng.sample(range(n), min(n, 5000))].astype(np.float64)


def co3d_pose_ok(z, f: dict, pts: np.ndarray) -> bool:
    """The frame's pose, read as PyTorch3D (x_cam = x_world R + T; +x left, +y up; NDC isotropic), puts the
    sequence's point cloud on the object mask, and clearly better than the mirrored reading does."""
    from PIL import Image

    mask = np.asarray(Image.open(io.BytesIO(z.read(f["mask"]["path"]))).convert("L")) > 128
    h, w = mask.shape
    v = f["viewpoint"]
    cam = pts @ np.array(v["R"]) + np.array(v["T"])
    front = cam[:, 2] > 1e-6
    cam = cam[front]
    s = min(w, h) / 2
    xn = v["focal_length"][0] * cam[:, 0] / cam[:, 2] + v["principal_point"][0]
    yn = v["focal_length"][1] * cam[:, 1] / cam[:, 2] + v["principal_point"][1]
    hits = []
    for sx in (-1, 1):
        u, y = w / 2 + sx * xn * s, h / 2 - yn * s
        ins = (u >= 0) & (u < w) & (y >= 0) & (y < h)
        hits.append(mask[y[ins].astype(int), u[ins].astype(int)].mean() if ins.sum() > 100 else 0.0)
    return front.mean() > 0.9 and hits[0] >= 0.8 and hits[0] >= hits[1] + 0.1


def co3d_tv(saver, cap, rng):
    """BLINK Multi-view_Reasoning from CO3D v2 tv train sequences (frame types train_*): two posed frames of one
    sequence, "left" when the camera centre moves toward the first and the second camera's left (both cosines at
    least 0.5) and the azimuth around the object (axis meeting point, about the mean camera up) turns clockwise
    seen from above by 20-120 degrees; "right" for the mirror case. Frame order is random, so gold is balanced."""
    z = zipfile.ZipFile(CO3D)
    by = defaultdict(list)
    for f in json.loads(gzip.decompress(z.read("tv/frame_annotations.jgz"))):
        if f["meta"]["frame_type"].startswith("train_"):
            by[f["sequence_name"]].append(f)
    made = []
    for seq in sorted(by):
        pts = co3d_points(z, seq, rng)
        if pts is None:
            continue
        frames = rng.sample(by[seq], min(CO3D_FRAMES, len(by[seq])))
        good = [f for f in frames if co3d_pose_ok(z, f, pts)]
        if len(good) < 4:
            continue
        R = np.array([f["viewpoint"]["R"] for f in good])
        C = -np.einsum("nj,nij->ni", np.array([f["viewpoint"]["T"] for f in good]), R)  # -T R^T
        left, up, fwd = R[:, :, 0], R[:, :, 1], R[:, :, 2]
        P = np.eye(3)[None] - fwd[:, :, None] * fwd[:, None, :]
        center = np.linalg.solve(P.sum(0), np.einsum("nij,nj->i", P, C))  # nearest point to all optical axes
        u = up.mean(0) / np.linalg.norm(up.mean(0))
        e1 = np.cross(u, fwd[0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(u, e1)
        az = np.degrees(np.arctan2((C - center) @ e2, (C - center) @ e1))
        pairs = [(i, j) for i in range(len(good)) for j in range(len(good)) if i != j]
        rng.shuffle(pairs)
        n = 0
        for i, j in pairs:
            d = C[j] - C[i]
            turn = (az[j] - az[i] + 180) % 360 - 180  # counter-clockwise seen from above is positive
            c1, c2 = d @ left[i] / np.linalg.norm(d), d @ left[j] / np.linalg.norm(d)
            if not 20 <= abs(turn) <= 120 or min(abs(c1), abs(c2)) < 0.5 or not (c1 > 0) == (c2 > 0) == (turn < 0):
                continue
            imgs = [saver.bytes(z.read(good[k]["image"]["path"])) for k in (i, j)]
            made.append(item("co3d_tv", f"{seq}/{good[i]['frame_number']}-{good[j]['frame_number']}", imgs, MV_Q,
                             ["left", "right"], 0 if c1 > 0 else 1, "camera_orbit", "program", ordered=True))
            n += 1
            if n == CO3D_PER_SEQ:
                break
    rng.shuffle(made)
    yield from made


# --- spatial and counting ---------------------------------------------------------------------------------------

SAT = RAW / "array__SAT@bda5dde942d7/SAT_train.parquet"
SAT_OTHER = [("count", r"^How many "), ("depth_point", r"^Which point is closer"),
             ("depth_object", r"^Which object is closer to the camera"),
             ("distance", r"^(Estimate the real world|Which object is closer to )"),
             ("camera_move", r"Did the camera move"), ("height", r"higher height"),
             ("front_behind", r"\b(in front|behind|further away)\b"), ("relation", r"\b(left|right|above|below)\b")]
SIDES = {"left", "right", "above", "below"}


def sat_subtask(r: dict) -> str:
    if r["question_type"] != "other":
        return r["question_type"]
    if {norm(a) for a in r["answers"]} <= SIDES:
        return "relation"
    return next((name for name, pat in SAT_OTHER if re.search(pat, r["question"])), "other")


def sat_options(r: dict) -> tuple[list[str], int, bool] | None:
    """Deduplicated options, gold index and ordered flag (yes/no as ["Yes", "No"], drawn letters sorted)."""
    opts = []
    for a in r["answers"]:
        a = re.sub(r"\s+", " ", a).strip()
        if a and norm(a) not in {norm(o) for o in opts}:
            opts.append(a)
    gold = norm(re.sub(r"\s+", " ", r["correct_answer"]))
    keys = [norm(o) for o in opts]
    if gold not in keys or len(opts) < 2:
        return None
    if sorted(keys) == ["no", "yes"]:
        return ["Yes", "No"], ["yes", "no"].index(gold), True
    if all(re.fullmatch(r"[A-J]", o) for o in opts):
        opts = sorted(opts)
        return opts, [norm(o) for o in opts].index(gold), True
    return opts, keys.index(gold), False


def sat(saver, cap, rng):
    import pyarrow.parquet as pq

    f = pq.ParquetFile(SAT)
    rows = f.read(columns=["question", "answers", "question_type", "correct_answer"]).to_pylist()
    groups = defaultdict(lambda: defaultdict(list))  # subtask -> gold index if ordered, else -1 -> rows
    for i, r in enumerate(rows):
        q = re.sub(r"\s+", " ", r["question"]).strip()
        o = sat_options(r)
        if o and q and len(q) + sum(len(x) for x in o[0]) <= 2500:
            groups[sat_subtask(r)][o[1] if o[2] else -1].append(i)
    take = shares({k: sum(map(len, g.values())) for k, g in groups.items()}, cap)
    chosen = {}
    for k, g in groups.items():  # equal shares per subtask, then per gold position where order is kept
        for gk, n in shares({gk: len(v) for gk, v in g.items()}, take[k]).items():
            chosen.update((i, k) for i in rng.sample(g[gk], n))
    start = 0
    for batch in f.iter_batches(batch_size=512, columns=["image_bytes"]):
        col = batch.column(0)
        for j in range(len(col)):
            i = start + j
            if i in chosen:
                r = rows[i]
                opts, gold, ordered = sat_options(r)
                imgs = [saver.bytes(b) for b in col[j].as_py()]
                yield item("sat", f"train:{i}", imgs, re.sub(r"\s+", " ", r["question"]).strip(), opts, gold,
                           chosen[i], "program", ordered=ordered)
        start += len(col)


FSC = RAW / "isentropic__FSC147@3e420cb6537e"
FSC_NAMES = {"cupcake tray": "cupcakes", "go game": "go stones", "stairs": "steps", "watermelon": "watermelons",
             "cereals": "cereal boxes", "ice cream": "ice cream tubs", "jeans": "pairs of jeans", "spoon": "spoons",
             "chopstick": "chopsticks", "lipstick": "lipsticks", "naan bread": "pieces of naan bread",
             "m&m pieces": "M&M pieces", "goldfish snack": "goldfish crackers"}
FSC_SKIP = {"mini blinds", "supermarket shelf", "polka dot tiles"}  # what is counted is unclear from the name


def count_options(n: int, pos: int, rng) -> list[int] | None:
    """Four ascending counts with n at index pos: consecutive up to FSC_EXACT, else a geometric ladder."""
    if n <= FSC_EXACT:
        opts = [n - pos + j for j in range(4)]
    else:
        f = rng.uniform(1.35, 1.6)
        opts = [round(n * f ** (j - pos)) for j in range(4)]
        opts[pos] = n
    ok = opts[0] >= 1 and all(a < b for a, b in itertools.pairwise(opts))
    return opts if ok else None


def fsc147(saver, cap, rng):
    from decision_index.vision.convert import Letters

    split = json.loads((FSC / "Train_Test_Val_FSC_147.json").read_text())["train"]
    ann = json.loads((FSC / "annotation_FSC147_384.json").read_text())
    cls = dict(line.rstrip("\n").split("\t") for line in (FSC / "ImageClasses_FSC147.txt").open() if line.strip())
    names = sorted(split)
    rng.shuffle(names)
    letters, seen = Letters(rng), set()
    for name in names:
        c = cls[name]
        if c in FSC_SKIP:
            continue
        img = saver.file(FSC / "images_384_VarV2" / name)
        if img in seen:  # a few images are in the set twice under different names
            continue
        seen.add(img)
        n = len(ann[name]["points"])
        pos = letters.pick(4)
        opts = count_options(n, pos, rng)
        if opts is None:
            letters.counts["ABCD"[pos]] -= 1
            continue
        noun = FSC_NAMES.get(c, c)
        q = f"How many {noun} are in the image?" if n <= FSC_EXACT else f"Approximately how many {noun} are in the image?"
        yield item("fsc147", name.removesuffix(".jpg"), [img], q,
                   [str(o) for o in opts], pos, "exact" if n <= FSC_EXACT else "approximate", "human", ordered=True)


def fsc_windows(points: list, side: float, W: int, H: int, rng, tries: int = 4000) -> dict[int, tuple]:
    """Windows (x0, y0, x1, y1) by how many points they hold, each with no point near its edge."""
    g = np.random.default_rng(rng.randrange(2**32))
    m = FSC_EDGE * side
    lo = min(max(2.5 * side, 64), W, H)
    w = np.exp(g.uniform(np.log(lo), np.log(W), tries))  # log-uniform sides, so small windows are tried as often
    h = np.clip(w * np.exp(g.uniform(np.log(0.6), np.log(1.6), tries)), lo, H)
    x0, y0 = g.uniform(0, 1, tries) * (W - w), g.uniform(0, 1, tries) * (H - h)
    x1, y1 = x0 + w, y0 + h
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    px, py = p[:, :1], p[:, 1:]  # points x windows
    inside = (px >= x0 + m) & (px <= x1 - m) & (py >= y0 + m) & (py <= y1 - m)
    near = (px > x0 - m) & (px < x1 + m) & (py > y0 - m) & (py < y1 + m)
    n = inside.sum(0)
    ok = ((near & ~inside).sum(0) == 0) & (n >= 1) & (n <= FSC_CROP_MAX)
    out = {}
    for i in np.nonzero(ok)[0]:
        out.setdefault(int(n[i]), (x0[i], y0[i], x1[i], y1[i]))
    return out


def fsc147_crop(saver, cap, rng):
    from PIL import Image

    split = json.loads((FSC / "Train_Test_Val_FSC_147.json").read_text())["train"]
    ann = json.loads((FSC / "annotation_FSC147_384.json").read_text())
    cls = dict(line.rstrip("\n").split("\t") for line in (FSC / "ImageClasses_FSC147.txt").open() if line.strip())
    names = sorted(split)
    rng.shuffle(names)
    made, seen = defaultdict(int), set()  # count -> items so far, so the counts stay in equal shares
    for name in names:
        c = cls[name]
        if c in FSC_SKIP:
            continue
        raw = (FSC / "images_384_VarV2" / name).read_bytes()
        if hash(raw) in seen:  # a few images are in the set twice under different names
            continue
        seen.add(hash(raw))
        a = ann[name]
        side = float(np.median([max(b[2][0] - b[0][0], b[2][1] - b[0][1]) for b in a["box_examples_coordinates"]]))
        if side <= 0:
            continue
        with Image.open(io.BytesIO(raw)) as im:
            im = im.convert("RGB")
        wins = fsc_windows(a["points"], side, *im.size, rng)
        for n in sorted(wins, key=lambda k: (made[k], rng.random()))[:FSC_CROPS_PER_IMAGE]:
            made[n] += 1
            x0, y0, x1, y1 = wins[n]
            crop = im.crop((round(x0), round(y0), round(x1), round(y1)))
            s = FSC_CROP_SIDE / min(crop.size)
            if s > 1:
                crop = crop.resize((round(crop.width * s), round(crop.height * s)), Image.BICUBIC)
            pos = rng.randrange(min(4, n + 1))
            bucket = "1-4" if n <= 4 else "5-8" if n <= 8 else f"9-{FSC_CROP_MAX}"
            yield item("fsc147_crop", f"{name.removesuffix('.jpg')}:{n}", [saver.pil(crop)],
                       f"How many {FSC_NAMES.get(c, c)} are in the image?", [str(n - pos + j) for j in range(4)],
                       pos, bucket, "human", ordered=True)


CONVERTERS = {
    "nyu_depth": ("depth", nyu_depth),
    "hypersim": ("depth", hypersim),
    "hpatches": ("correspondence", hpatches),
    "spair71k": ("correspondence", spair71k),
    "co3d_tv": ("multi-image", co3d_tv),
    "sat": ("spatial", sat),
    "fsc147": ("counting", fsc147),
    "fsc147_crop": ("counting", fsc147_crop),
}
