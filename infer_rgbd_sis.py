# infer_rgbd_sis.py

import argparse
import importlib
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.transforms import ResizeShortestEdge


PALETTE = np.array([
    [255, 0, 0],      # red
    [0, 255, 0],      # green
    [0, 0, 255],      # blue
    [255, 255, 0],    # yellow
    [255, 0, 255],    # magenta
    [0, 255, 255],    # cyan
    [255, 128, 0],
    [128, 0, 255],
    [255, 128, 128],
    [128, 255, 128],
], dtype=np.uint8)


def imread_unicode(path, flag=cv2.IMREAD_UNCHANGED):
    path = str(path)
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, flag)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def imwrite_unicode(path, img):
    path = str(path)
    ext = Path(path).suffix
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"Cannot encode image: {path}")
    buf.tofile(path)


def find_file_by_stem(folder, stem):
    folder = Path(folder)
    exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

    for ext in exts:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p

    raise FileNotFoundError(f"Cannot find file with stem={stem} in {folder}")


def collect_stems(rgb_dir, ids=None):
    if ids is not None and len(ids) > 0:
        return ids

    rgb_dir = Path(rgb_dir)
    exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

    stems = []
    for p in sorted(rgb_dir.iterdir()):
        if p.suffix.lower() in exts:
            stems.append(p.stem)

    return stems


def read_rgb_bgr(path):
    img = imread_unicode(path, cv2.IMREAD_COLOR)
    return img


def read_depth_float(path):
    depth = imread_unicode(path, cv2.IMREAD_UNCHANGED)

    if depth.ndim == 3:
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)

    depth = depth.astype(np.float32)
    depth = depth - depth.min()

    if depth.max() > 0:
        depth = depth / depth.max()

    return depth


def depth_to_rgb_vis(depth_float):
    depth = depth_float.copy()
    depth = depth - depth.min()

    if depth.max() > 0:
        depth = depth / depth.max()

    depth = (depth * 255).astype(np.uint8)
    depth_rgb = cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)
    return depth_rgb


def build_cfg(config_file, weights, score_thresh, project_root=None):
    if project_root is None:
        project_root = str(Path(__file__).resolve().parent)

    sys.path.insert(0, project_root)

    from calibnet import add_calibnet_config

    cfg = get_cfg()
    add_calibnet_config(cfg)

    cfg.merge_from_file(config_file)

    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    if hasattr(cfg.MODEL, "ROI_HEADS"):
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh

    if hasattr(cfg.MODEL, "RETINANET"):
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = score_thresh

    if hasattr(cfg.MODEL, "FCOS"):
        cfg.MODEL.FCOS.SCORE_THRESH_TEST = score_thresh

    if hasattr(cfg.MODEL, "CALIBNET"):
        try:
            cfg.MODEL.CALIBNET.SCORE_THRESH_TEST = score_thresh
        except Exception:
            pass

        try:
            cfg.MODEL.CALIBNET.INFERENCE_TH_TEST = score_thresh
        except Exception:
            pass

    cfg.freeze()
    return cfg


def get_resize_transform(cfg, image_bgr):
    min_size = cfg.INPUT.MIN_SIZE_TEST

    if isinstance(min_size, int):
        min_size = [min_size]
    else:
        min_size = list(min_size)

    aug = ResizeShortestEdge(
        short_edge_length=min_size,
        max_size=cfg.INPUT.MAX_SIZE_TEST,
        sample_style="choice"
    )

    return aug.get_transform(image_bgr)


@torch.no_grad()
def inference_one(
    model,
    cfg,
    rgb_path,
    depth_path,
    score_thresh=0.3,
    mask_thresh=0.5,
    min_area_ratio=0.001,
    max_instances=10,
    depth_as_3ch=False,
    debug=False
):
    rgb_bgr = read_rgb_bgr(rgb_path)
    depth_float = read_depth_float(depth_path)

    ori_h, ori_w = rgb_bgr.shape[:2]

    transform = get_resize_transform(cfg, rgb_bgr)

    rgb_resized = transform.apply_image(rgb_bgr)

    depth_resized = cv2.resize(
        depth_float,
        (rgb_resized.shape[1], rgb_resized.shape[0]),
        interpolation=cv2.INTER_LINEAR
    )

    image_tensor = torch.as_tensor(
        rgb_resized.astype("float32").transpose(2, 0, 1)
    )

    if depth_as_3ch:
        depth_3ch = np.stack(
            [depth_resized, depth_resized, depth_resized],
            axis=2
        )
        depth_tensor = torch.as_tensor(
            depth_3ch.astype("float32").transpose(2, 0, 1)
        )
    else:
        depth_tensor = torch.as_tensor(
            depth_resized.astype("float32")[None, :, :]
        )

    inputs = {
        "image": image_tensor,
        "depth": depth_tensor,
        "height": ori_h,
        "width": ori_w,
    }

    outputs = model([inputs])[0]

    if "instances" not in outputs:
        raise RuntimeError(
            f"Model output has no key 'instances'. Output keys: {outputs.keys()}"
        )

    instances = outputs["instances"].to("cpu")

    if len(instances) == 0:
        if debug:
            print("No instances predicted.")
        return rgb_bgr, [], np.array([], dtype=np.float32)

    if not instances.has("pred_masks"):
        raise RuntimeError("Model output instances have no pred_masks.")

    pred_masks = instances.pred_masks.numpy()

    if pred_masks.dtype != np.bool_:
        pred_masks = pred_masks > mask_thresh

    if instances.has("scores"):
        scores = instances.scores.numpy()
    else:
        scores = np.ones(len(pred_masks), dtype=np.float32)

    if debug:
        print("before filter num:", len(scores))
        if len(scores) > 0:
            print("score min/max:", float(scores.min()), float(scores.max()))
            print("top scores:", scores[:10])

    keep = scores >= score_thresh
    pred_masks = pred_masks[keep]
    scores = scores[keep]

    if len(pred_masks) > 0:
        h, w = pred_masks.shape[-2:]
        areas = pred_masks.reshape(len(pred_masks), -1).sum(axis=1)
        area_thresh = h * w * min_area_ratio

        area_keep = areas >= area_thresh
        pred_masks = pred_masks[area_keep]
        scores = scores[area_keep]
        areas = areas[area_keep]

    if len(pred_masks) > 0:
        order = np.argsort(-scores)
        order = order[:max_instances]

        pred_masks = pred_masks[order]
        scores = scores[order]

    if debug:
        print("after filter num:", len(scores))
        if len(scores) > 0:
            print("kept scores:", scores)
            kept_areas = pred_masks.reshape(len(pred_masks), -1).sum(axis=1)
            print("kept areas:", kept_areas)

    return rgb_bgr, pred_masks, scores


def masks_to_label(masks, h, w):
    label = np.zeros((h, w), dtype=np.uint16)

    for idx, mask in enumerate(masks, start=1):
        label[mask.astype(bool)] = idx

    return label


def label_to_color(label, colors=None):
    h, w = label.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    ids = [v for v in np.unique(label) if v != 0]

    for i, inst_id in enumerate(ids):
        if colors is not None and i < len(colors):
            color = np.array(colors[i], dtype=np.uint8)
        else:
            color = PALETTE[i % len(PALETTE)]

        color_mask[label == inst_id] = color

    return color_mask


def read_gt_color_masks(gt_path, target_hw=None):
    gt_bgr = imread_unicode(gt_path, cv2.IMREAD_COLOR)
    gt_rgb = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB)

    if target_hw is not None:
        h, w = target_hw
        gt_rgb = cv2.resize(
            gt_rgb,
            (w, h),
            interpolation=cv2.INTER_NEAREST
        )

    colors = np.unique(gt_rgb.reshape(-1, 3), axis=0)

    gt_masks = []
    gt_colors = []

    for color in colors:
        if np.all(color == 0):
            continue

        mask = np.all(gt_rgb == color, axis=-1)

        if mask.sum() < 10:
            continue

        gt_masks.append(mask)
        gt_colors.append(color.astype(np.uint8))

    return gt_masks, gt_colors, gt_rgb


def mask_iou(mask1, mask2):
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)

    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 0.0

    return float(inter) / float(union)


def assign_pred_colors_by_gt(
    pred_masks,
    gt_masks,
    gt_colors,
    iou_thresh=0.01,
    gt_cover_thresh=0.20
):
    """
    预测 mask 和 GT mask 做匹配，然后预测结果使用对应 GT 的颜色。

    同时使用:
    1. IoU = inter / union
    2. GT coverage = inter / gt_area

    GT coverage 对小目标更友好，比如杯子这类小实例。
    """
    num_pred = len(pred_masks)
    num_gt = len(gt_masks)

    pred_colors = [
        PALETTE[i % len(PALETTE)].copy()
        for i in range(num_pred)
    ]

    matched_gt_indices = [-1 for _ in range(num_pred)]
    matched_ious = [0.0 for _ in range(num_pred)]
    matched_gt_covers = [0.0 for _ in range(num_pred)]

    if num_pred == 0 or num_gt == 0:
        return pred_colors, matched_gt_indices, matched_ious, matched_gt_covers

    match_items = []

    for i in range(num_pred):
        pred = pred_masks[i].astype(bool)
        pred_area = pred.sum()

        if pred_area == 0:
            continue

        for j in range(num_gt):
            gt = gt_masks[j].astype(bool)
            gt_area = gt.sum()

            if gt_area == 0:
                continue

            inter = np.logical_and(pred, gt).sum()
            union = np.logical_or(pred, gt).sum()

            iou = float(inter) / float(union) if union > 0 else 0.0
            gt_cover = float(inter) / float(gt_area) if gt_area > 0 else 0.0

            match_score = max(iou, gt_cover)
            match_items.append((match_score, iou, gt_cover, i, j))

    match_items.sort(key=lambda x: x[0], reverse=True)

    used_pred = set()
    used_gt = set()

    for match_score, iou, gt_cover, pred_idx, gt_idx in match_items:
        if pred_idx in used_pred or gt_idx in used_gt:
            continue

        if iou < iou_thresh and gt_cover < gt_cover_thresh:
            continue

        pred_colors[pred_idx] = gt_colors[gt_idx]
        matched_gt_indices[pred_idx] = gt_idx
        matched_ious[pred_idx] = iou
        matched_gt_covers[pred_idx] = gt_cover

        used_pred.add(pred_idx)
        used_gt.add(gt_idx)

    return pred_colors, matched_gt_indices, matched_ious, matched_gt_covers


def overlay_instances(
    rgb_bgr,
    masks,
    colors=None,
    alpha=0.55,
    contour_thickness=2
):
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    out = rgb.copy().astype(np.float32)

    if colors is None:
        colors = [
            PALETTE[i % len(PALETTE)]
            for i in range(len(masks))
        ]

    for i, mask in enumerate(masks):
        m = mask.astype(bool)

        if m.sum() == 0:
            continue

        color = np.array(colors[i], dtype=np.float32)

        out[m] = out[m] * (1 - alpha) + color * alpha

        m_uint8 = m.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            m_uint8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        out_uint8 = np.clip(out, 0, 255).astype(np.uint8)

        cv2.drawContours(
            out_uint8,
            contours,
            -1,
            color=(255, 255, 255),
            thickness=contour_thickness
        )

        out = out_uint8.astype(np.float32)

    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def put_title(img_rgb, title, height=36):
    h, w = img_rgb.shape[:2]

    canvas = np.zeros((h + height, w, 3), dtype=np.uint8)
    canvas[height:, :, :] = img_rgb

    cv2.putText(
        canvas,
        title,
        (8, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return canvas


def make_panel(rgb_bgr, gt_rgb, depth_rgb, pred_bgr):
    rgb_rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    pred_rgb = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2RGB)

    h, w = rgb_rgb.shape[:2]

    gt_rgb = cv2.resize(gt_rgb, (w, h), interpolation=cv2.INTER_NEAREST)
    depth_rgb = cv2.resize(depth_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    pred_rgb = cv2.resize(pred_rgb, (w, h), interpolation=cv2.INTER_LINEAR)

    imgs = [
        put_title(rgb_rgb, "RGB"),
        put_title(gt_rgb, "GT"),
        put_title(depth_rgb, "Depth"),
        put_title(pred_rgb, "Prediction"),
    ]

    panel_rgb = np.concatenate(imgs, axis=1)
    panel_bgr = cv2.cvtColor(panel_rgb, cv2.COLOR_RGB2BGR)

    return panel_bgr


def save_match_info(
    path,
    scores,
    matched_gt_indices,
    matched_ious,
    matched_gt_covers,
    pred_colors
):
    lines = []
    lines.append("pred_id score matched_gt_id matched_iou matched_gt_cover color_rgb\n")

    for i in range(len(scores)):
        color = pred_colors[i]
        color_str = f"{int(color[0])},{int(color[1])},{int(color[2])}"

        matched_id = matched_gt_indices[i] + 1 if matched_gt_indices[i] >= 0 else -1

        lines.append(
            f"{i + 1} "
            f"{float(scores[i]):.6f} "
            f"{matched_id} "
            f"{matched_ious[i]:.6f} "
            f"{matched_gt_covers[i]:.6f} "
            f"{color_str}\n"
        )

    with open(path, "w") as f:
        f.writelines(lines)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True, help="CalibNet config yaml")
    parser.add_argument("--weights", required=True, help="trained .pth weight")
    parser.add_argument("--rgb_dir", required=True)
    parser.add_argument("--depth_dir", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument(
        "--gt_dir",
        default=None,
        help="colored instance GT folder. If provided, prediction colors will match GT colors by IoU and GT coverage."
    )

    parser.add_argument("--ids", nargs="*", default=None)

    parser.add_argument("--score_thresh", type=float, default=0.20)
    parser.add_argument("--mask_thresh", type=float, default=0.5)

    parser.add_argument(
        "--match_iou_thresh",
        type=float,
        default=0.01,
        help="IoU threshold for assigning GT color to prediction."
    )

    parser.add_argument(
        "--gt_cover_thresh",
        type=float,
        default=0.20,
        help="GT coverage threshold for assigning GT color to prediction."
    )

    parser.add_argument(
        "--min_area_ratio",
        type=float,
        default=0.001,
        help="Remove masks whose area is smaller than H*W*min_area_ratio."
    )

    parser.add_argument(
        "--max_instances",
        type=int,
        default=10,
        help="Keep top-K instances by score."
    )

    parser.add_argument(
        "--depth_as_3ch",
        action="store_true",
        help="Use this if the depth backbone expects 3-channel depth input."
    )

    parser.add_argument(
        "--project_root",
        default=None,
        help="CalibNet project root. Default is the folder of this script."
    )

    parser.add_argument(
        "--project_import",
        default=None,
        help="Optional extra import, e.g. calibnet"
    )

    parser.add_argument("--save_panel", action="store_true")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    project_root = args.project_root
    if project_root is None:
        project_root = str(Path(__file__).resolve().parent)

    sys.path.insert(0, project_root)

    if args.project_import is not None:
        importlib.import_module(args.project_import)

    cfg = build_cfg(
        config_file=args.config,
        weights=args.weights,
        score_thresh=args.score_thresh,
        project_root=project_root
    )

    model = build_model(cfg)
    model.eval()

    DetectionCheckpointer(model).load(args.weights)

    out_dir = Path(args.out_dir)

    mask_dir = out_dir / "masks"
    color_mask_dir = out_dir / "color_masks"
    vis_dir = out_dir / "vis"
    score_dir = out_dir / "scores"
    match_dir = out_dir / "matches"
    panel_dir = out_dir / "panels"

    mask_dir.mkdir(parents=True, exist_ok=True)
    color_mask_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(parents=True, exist_ok=True)
    match_dir.mkdir(parents=True, exist_ok=True)

    if args.save_panel:
        panel_dir.mkdir(parents=True, exist_ok=True)

    stems = collect_stems(args.rgb_dir, args.ids)

    print(f"Total images: {len(stems)}")
    print(f"Output dir: {out_dir}")

    for idx, stem in enumerate(stems):
        print(f"[{idx + 1}/{len(stems)}] infer {stem}")

        rgb_path = find_file_by_stem(args.rgb_dir, stem)
        depth_path = find_file_by_stem(args.depth_dir, stem)

        rgb_bgr, pred_masks, scores = inference_one(
            model=model,
            cfg=cfg,
            rgb_path=rgb_path,
            depth_path=depth_path,
            score_thresh=args.score_thresh,
            mask_thresh=args.mask_thresh,
            min_area_ratio=args.min_area_ratio,
            max_instances=args.max_instances,
            depth_as_3ch=args.depth_as_3ch,
            debug=args.debug
        )

        h, w = rgb_bgr.shape[:2]

        gt_rgb = None

        pred_colors = [
            PALETTE[i % len(PALETTE)]
            for i in range(len(pred_masks))
        ]

        matched_gt_indices = [-1 for _ in range(len(pred_masks))]
        matched_ious = [0.0 for _ in range(len(pred_masks))]
        matched_gt_covers = [0.0 for _ in range(len(pred_masks))]

        if args.gt_dir is not None:
            gt_path = find_file_by_stem(args.gt_dir, stem)

            gt_masks, gt_colors, gt_rgb = read_gt_color_masks(
                gt_path,
                target_hw=(h, w)
            )

            (
                pred_colors,
                matched_gt_indices,
                matched_ious,
                matched_gt_covers
            ) = assign_pred_colors_by_gt(
                pred_masks=pred_masks,
                gt_masks=gt_masks,
                gt_colors=gt_colors,
                iou_thresh=args.match_iou_thresh,
                gt_cover_thresh=args.gt_cover_thresh
            )

        label = masks_to_label(pred_masks, h, w)

        color_label_rgb = label_to_color(
            label,
            colors=pred_colors
        )

        vis_bgr = overlay_instances(
            rgb_bgr=rgb_bgr,
            masks=pred_masks,
            colors=pred_colors,
            alpha=0.55,
            contour_thickness=2
        )

        imwrite_unicode(mask_dir / f"{stem}.png", label)

        color_label_bgr = cv2.cvtColor(color_label_rgb, cv2.COLOR_RGB2BGR)
        imwrite_unicode(color_mask_dir / f"{stem}.png", color_label_bgr)

        imwrite_unicode(vis_dir / f"{stem}.png", vis_bgr)

        np.savetxt(
            str(score_dir / f"{stem}.txt"),
            scores,
            fmt="%.6f"
        )

        save_match_info(
            match_dir / f"{stem}.txt",
            scores=scores,
            matched_gt_indices=matched_gt_indices,
            matched_ious=matched_ious,
            matched_gt_covers=matched_gt_covers,
            pred_colors=pred_colors
        )

        if args.save_panel:
            depth_float = read_depth_float(depth_path)
            depth_rgb = depth_to_rgb_vis(depth_float)
            depth_rgb = cv2.resize(
                depth_rgb,
                (w, h),
                interpolation=cv2.INTER_LINEAR
            )

            if gt_rgb is None:
                gt_rgb = np.zeros((h, w, 3), dtype=np.uint8)

            panel_bgr = make_panel(
                rgb_bgr=rgb_bgr,
                gt_rgb=gt_rgb,
                depth_rgb=depth_rgb,
                pred_bgr=vis_bgr
            )

            imwrite_unicode(panel_dir / f"{stem}.png", panel_bgr)

    print("Done.")


if __name__ == "__main__":
    main()