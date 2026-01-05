#!/usr/bin/env python3
import json
import argparse
import os
from typing import Dict, List, Any, Tuple


def parse_pred_id(pred_id: str) -> Tuple[str, str, str]:
    """
    解析预测 JSON 中的 id 字段：
    形如: cluster1|S.oshi-1_98|AMP-binding.1|2-348
    返回:
        (cluster_label, seq_id, amp_name)
    """
    parts = pred_id.split("|")
    if len(parts) < 3:
        raise ValueError(f"Unexpected prediction id format: {pred_id}")
    cluster_label = parts[0]             # "cluster1"
    seq_id = parts[1]                    # "S.oshi-1_98"
    amp_name = parts[2]                  # "AMP-binding.1"
    return cluster_label, seq_id, amp_name


def build_prediction_index(pred_json_path: str) -> Dict[Tuple[str, str, str], List[dict]]:
    """
    读取 substrate_predictions_top3_all.weight.json，
    构建一个索引: (cluster_label, seq_id, amp_name) -> predictions 列表
    """
    with open(pred_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    index: Dict[Tuple[str, str, str], List[dict]] = {}

    # data 可以是 list 或 dict，根据你实际文件结构调整
    if isinstance(data, dict) and "results" in data:
        items = data["results"]
    else:
        items = data   # 假设就是一个 list

    for item in items:
        pred_id = item["id"]
        preds = item.get("predictions", [])
        key = parse_pred_id(pred_id)  # (cluster_label, seq_id, amp_name)
        index[key] = preds

    return index


def merge_predictions(
    nrps_json_path: str,
    pred_json_path: str,
    out_path: str
) -> None:
    # 1) 读入两个 JSON
    with open(nrps_json_path, "r", encoding="utf-8") as f:
        nrps_data = json.load(f)

    pred_index = build_prediction_index(pred_json_path)

    cluster_list = nrps_data.get("clusters", [])

    n_attached = 0
    n_missing = 0

    for cluster in cluster_list:
        cid = cluster.get("cluster_id")
        cluster_label = f"cluster{cid}"  # 与 pred JSON 中的 "cluster1" 对齐

        for orf in cluster.get("orfs", []):
            seq_id = orf.get("seq_id")

            for module in orf.get("modules", []):
                for domain in module.get("domains", []):
                    name = domain.get("name", "")

                    # 只对 A-domain 做预测合并：name 像 "AMP-binding.1"
                    if not name.startswith("AMP-binding."):
                        continue

                    amp_name = name  # e.g. "AMP-binding.1"

                    key = (cluster_label, seq_id, amp_name)
                    preds = pred_index.get(key)

                    if preds is not None:
                        domain["predictions"] = preds
                        n_attached += 1
                    else:
                        n_missing += 1
                        # 可以按需打开调试:
                        # print(f"[WARN] No predictions for {key}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nrps_data, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Predictions attached to {n_attached} A-domains "
          f"(missing {n_missing}).")
    print(f"[INFO] Merged NRPS JSON written to: {out_path}")


def main():
    p = argparse.ArgumentParser(
        description="Merge substrate_predictions_topK_MODEL.weight.json into NRPS_modules.json."
    )
    p.add_argument(
        "--nrps-json",
        required=True,
        help="Path to NRPS_modules.json (from detect_nrps_modules)."
    )
    p.add_argument(
        "--pred-json",
        required=True,
        help="Path to substrate_predictions_topK_MODEL.weight.json."
    )
    p.add_argument(
        "--out",
        required=True,
        help="Output merged JSON file path."
    )
    args = p.parse_args()

    if not os.path.exists(args.nrps_json):
        raise SystemExit(f"[ERROR] NRPS JSON not found: {args.nrps_json}")
    if not os.path.exists(args.pred_json):
        raise SystemExit(f"[ERROR] Prediction JSON not found: {args.pred_json}")

    merge_predictions(args.nrps_json, args.pred_json, args.out)


if __name__ == "__main__":
    main()