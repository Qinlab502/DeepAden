#!/usr/bin/env python3
import csv
import sys
import json
import os
import argparse
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Literal, Optional

from Bio import SeqIO  # pip install biopython


# ------- 1. Domain types and shorthand -------

CONDENSATION_DOMAINS = {
    "Condensation",
    "Condensation_DCL",
    "Condensation_LCL",
    "Condensation_Starter",
    "Condensation_Dual",
    "Cglyc",
}

AMP_DOMAINS = {
    "AMP-binding",
}

TE_DOMAINS = {
    "Thioesterase",
}

PCP_DOMAINS = {
    "PCP",
    "PP-binding",
}

EPIM_DOMAINS = {
    "Epimerization",
}

OTHER_SHORTHAND = {
    "TIGR01720": "TIGR01720",
    "NRPS-COM_Nterm": "nCOM",
    "NRPS-COM_Cterm": "cCOM",
}

DomainType = Literal["C", "A", "PCP", "E"]


def classify_domain(domain_name: str) -> List[DomainType]:
    """Return C/A/PCP/E tags for a domain, used for module core matching."""
    dn = domain_name.strip()
    cats: List[DomainType] = []
    if dn in CONDENSATION_DOMAINS:
        cats.append("C")
    if dn in AMP_DOMAINS:
        cats.append("A")
    if dn in PCP_DOMAINS:
        cats.append("PCP")
    if dn in EPIM_DOMAINS:
        cats.append("E")
    return cats


def shorthand_for_full_chain(domain_name: str) -> str:
    """Map domain name to chain shorthand (C/A/PCP/E/TE/COM/etc.)."""
    dn = domain_name.strip()
    if dn in CONDENSATION_DOMAINS:
        return "C"
    if dn in AMP_DOMAINS:
        return "A"
    if dn in PCP_DOMAINS:
        return "PCP"
    if dn in EPIM_DOMAINS:
        return "E"
    if dn in TE_DOMAINS:
        return "TE"
    if dn in OTHER_SHORTHAND:
        return OTHER_SHORTHAND[dn]
    return dn


# ------- 2. Linearize ORF domain chain -------

@dataclass
class SeqChains:
    seq_id: str
    full_chain: List[str]          # all domain shorthands
    core_types: List[DomainType]   # only C/A/PCP/E
    core_idx: List[int]            # indices in full_chain for core_types


def build_linear_chains_for_seq(rows: List[dict]) -> SeqChains:
    if not rows:
        raise ValueError("build_linear_chains_for_seq() got empty rows")

    seq_id = rows[0]["Seq_ID"]
    try:
        sorted_rows = sorted(rows, key=lambda r: int(r["Order"]))
    except KeyError:
        raise ValueError("Input CSV must contain an 'Order' column (int-like).")

    full_chain: List[str] = []
    core_types: List[DomainType] = []
    core_idx: List[int] = []

    for i, r in enumerate(sorted_rows):
        dn = r["Domain"].strip()
        full_chain.append(shorthand_for_full_chain(dn))
        cats = classify_domain(dn)
        for c in cats:
            core_types.append(c)
            core_idx.append(i)

    return SeqChains(seq_id=seq_id, full_chain=full_chain,
                     core_types=core_types, core_idx=core_idx)


# ------- 3. Module patterns and matching -------

@dataclass
class ModulePattern:
    pattern: List[DomainType]
    allow_overlap: bool


MODULE_PATTERNS: List[ModulePattern] = [
    ModulePattern(pattern=["C", "A", "PCP", "E"], allow_overlap=False),
    ModulePattern(pattern=["C", "A", "PCP"], allow_overlap=False),
    ModulePattern(pattern=["A", "PCP"], allow_overlap=False),
]


@dataclass
class ModuleDetectResult:
    module_count: int
    a_count: int
    is_nrps_orf_relaxed: bool
    core_chain_str: str
    segments: List[Tuple[int, int]]  # [start,end] on core_types


def find_modules_in_chain(core_types: List[DomainType],
                          core_idx: List[int],
                          full_chain: List[str]) -> ModuleDetectResult:
    n = len(core_types)
    used_core_pos = [False] * n
    matched_segments: List[Tuple[int, int]] = []

    i = 0
    while i < n:
        matched = False
        for mp in MODULE_PATTERNS:
            m = len(mp.pattern)
            if i + m > n:
                continue
            window = core_types[i:i + m]
            if window == mp.pattern:
                if mp.pattern == ["A", "PCP"]:
                    full_i = core_idx[i]
                    full_j = core_idx[i + 1]
                    if full_j != full_i + 1:
                        continue
                if not mp.allow_overlap:
                    if any(used_core_pos[k] for k in range(i, i + m)):
                        continue
                matched_segments.append((i, i + m - 1))
                if not mp.allow_overlap:
                    for k in range(i, i + m):
                        used_core_pos[k] = True
                i += m
                matched = True
                break
        if not matched:
            i += 1

    module_count = len(matched_segments)
    a_count = sum(1 for t in core_types if t == "A")
    is_nrps_orf_relaxed = module_count > 0

    if matched_segments:
        used_full_idx: List[int] = []
        for start, end in matched_segments:
            for k in range(start, end + 1):
                used_full_idx.append(core_idx[k])
        used_full_idx = sorted(set(used_full_idx))
        core_chain_tokens = [full_chain[j] for j in used_full_idx]
        core_chain_str = "-".join(core_chain_tokens)
    else:
        core_chain_str = "-"

    return ModuleDetectResult(
        module_count=module_count,
        a_count=a_count,
        is_nrps_orf_relaxed=is_nrps_orf_relaxed,
        core_chain_str=core_chain_str,
        segments=matched_segments,
    )


# ------- 3b. JSON structures -------

@dataclass
class DomainToken:
    name: str
    short: str
    order: int
    index: int          # index in full_chain
    aa_seq: str


@dataclass
class ModuleBlock:
    orf_id: str
    module_index: int
    domains: List[DomainToken]
    is_orphan: bool
    warnings: List[str]


def build_domain_tokens_for_seq(rows: List[dict]) -> List[DomainToken]:
    sorted_rows = sorted(rows, key=lambda r: int(r["Order"]))
    tokens: List[DomainToken] = []
    for i, r in enumerate(sorted_rows):
        name = r["Domain"].strip()
        short = shorthand_for_full_chain(name)
        if "domain_order" in r and r["domain_order"].strip() != "":
            order = int(r["domain_order"])
        else:
            order = int(r["Order"])
        aa_seq = r.get("domain_sequence", "").strip()
        tokens.append(DomainToken(
            name=name,
            short=short,
            order=order,
            index=i,
            aa_seq=aa_seq,
        ))
    return tokens


# ------- 3c. Assign domains to modules -------

def split_modules_for_orf(
    orf_id: str,
    rows_for_orf: List[dict],
    chains: SeqChains,
    detect_res: ModuleDetectResult,
) -> List[ModuleBlock]:
    """
    Use C/A/PCP/E cores to define modules, then assign helpers:
      - TIGR01720: nearest module containing E, else nearest previous module
      - cCOM: previous module
      - nCOM: next module
      - TE: last non-orphan module
      - OTHER: nearest module
    """
    tokens = build_domain_tokens_for_seq(rows_for_orf)
    full_n = len(tokens)
    modules: List[ModuleBlock] = []

    if not detect_res.segments:
        return modules

    # 1) base modules from core segments
    base_modules: List[ModuleBlock] = []
    for idx, (cs, ce) in enumerate(detect_res.segments, start=1):
        full_start = chains.core_idx[cs]
        full_end = chains.core_idx[ce]
        domain_slice = [tk for tk in tokens if full_start <= tk.index <= full_end]
        base_modules.append(ModuleBlock(
            orf_id=orf_id,
            module_index=idx,
            domains=domain_slice,
            is_orphan=False,
            warnings=[],
        ))

    module_ranges: List[Tuple[int, int]] = []
    for m in base_modules:
        if not m.domains:
            module_ranges.append((0, -1))
        else:
            min_idx = min(d.index for d in m.domains)
            max_idx = max(d.index for d in m.domains)
            module_ranges.append((min_idx, max_idx))

    # 2) orphan modules: C/A after last core
    last_core_full_idx = chains.core_idx[detect_res.segments[-1][1]]
    orphan_tokens: List[DomainToken] = [
        tk for tk in tokens
        if tk.index > last_core_full_idx and tk.short in ("C", "A")
    ]
    next_mod_idx = len(base_modules) + 1
    orphan_modules: List[ModuleBlock] = []
    for tk in orphan_tokens:
        orphan_modules.append(ModuleBlock(
            orf_id=orf_id,
            module_index=next_mod_idx,
            domains=[tk],
            is_orphan=True,
            warnings=["Incomplete module - possible assembly issue"],
        ))
        next_mod_idx += 1

    all_modules: List[ModuleBlock] = base_modules + orphan_modules

    # 3) E-containing info
    module_has_E = []
    for m in all_modules:
        module_has_E.append(any(d.short == "E" for d in m.domains))

    def find_nearest_module_by_index(dom_idx: int) -> Optional[int]:
        best_m = None
        best_dist = None
        for midx, (start, end) in enumerate(module_ranges):
            if start > end:
                continue
            if dom_idx < start:
                dist = start - dom_idx
            elif dom_idx > end:
                dist = dom_idx - end
            else:
                return midx
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_m = midx
        return best_m

    def find_nearest_E_module(dom_idx: int) -> Optional[int]:
        best_m = None
        best_dist = None
        for midx, (start, end) in enumerate(module_ranges):
            if start > end:
                continue
            if not module_has_E[midx]:
                continue
            if dom_idx < start:
                dist = start - dom_idx
            elif dom_idx > end:
                dist = dom_idx - end
            else:
                return midx
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_m = midx
        return best_m

    assigned = [False] * full_n
    for m in all_modules:
        for d in m.domains:
            assigned[d.index] = True

    for tk in tokens:
        if assigned[tk.index]:
            continue

        short = tk.short
        idx = tk.index

        target_mid: Optional[int] = None

        if short == "TIGR01720":
            e_mod = find_nearest_E_module(idx)
            if e_mod is not None:
                target_mid = e_mod
            else:
                nearest = find_nearest_module_by_index(idx)
                if nearest is not None:
                    if module_ranges[nearest][0] <= idx:
                        left_candidates = [
                            (midx, rng) for midx, rng in enumerate(module_ranges)
                            if rng[0] <= idx and rng[1] <= idx
                        ]
                        if left_candidates:
                            best_m = None
                            best_dist = None
                            for midx, (s, e) in left_candidates:
                                dist = idx - e
                                if best_dist is None or dist < best_dist:
                                    best_dist = dist
                                    best_m = midx
                            target_mid = best_m
                        else:
                            target_mid = nearest
                    else:
                        left_candidates = [
                            (midx, rng) for midx, rng in enumerate(module_ranges)
                            if rng[1] < idx
                        ]
                        if left_candidates:
                            best_m = None
                            best_dist = None
                            for midx, (s, e) in left_candidates:
                                dist = idx - e
                                if best_dist is None or dist < best_dist:
                                    best_dist = dist
                                    best_m = midx
                            target_mid = best_m
                        else:
                            target_mid = nearest

        elif short == "cCOM":
            nearest = find_nearest_module_by_index(idx)
            if nearest is not None:
                if module_ranges[nearest][0] <= idx:
                    cand = nearest - 1
                else:
                    cand = nearest
                if 0 <= cand < len(all_modules):
                    target_mid = cand

        elif short == "nCOM":
            nearest = find_nearest_module_by_index(idx)
            if nearest is not None:
                if module_ranges[nearest][0] > idx:
                    cand = nearest
                else:
                    cand = nearest + 1
                if 0 <= cand < len(all_modules):
                    target_mid = cand

        elif short == "TE":
            if base_modules:
                target_mid = len(base_modules) - 1

        else:
            nearest = find_nearest_module_by_index(idx)
            if nearest is not None:
                target_mid = nearest

        if target_mid is not None and 0 <= target_mid < len(all_modules):
            all_modules[target_mid].domains.append(tk)
            assigned[idx] = True

    all_modules.sort(key=lambda m: m.module_index)
    for m in all_modules:
        m.domains.sort(key=lambda d: d.index)

    return all_modules


# ------- 4. Cluster-related helpers -------

def parse_seq_id(seq_id: str) -> Tuple[str, Optional[int]]:
    if "_" not in seq_id:
        return seq_id, None
    genome, idx_str = seq_id.rsplit("_", 1)
    try:
        idx = int(idx_str)
    except ValueError:
        idx = None
    return genome, idx


def is_strict_nrps_orf(core_chain: str, module_count: int, a_count: int) -> bool:
    """Strict NRPS ORF: at least 1 module, 1 A, and 'A-PCP' in core chain."""
    if module_count < 1 or a_count < 1:
        return False
    if "A-PCP" not in core_chain:
        return False
    return True


def build_clusters_from_summary(summary_rows: List[dict]) -> Tuple[Dict[str, int], Dict[int, bool]]:
    """Cluster strict NRPS ORFs into BGCs based on index gaps."""
    GAP_MAX = 4
    genome_to_orfs: Dict[str, List[Tuple[int, str, int, int]]] = defaultdict(list)

    for row in summary_rows:
        seq_id = row["Seq_ID"]
        genome, idx = parse_seq_id(seq_id)
        if idx is None:
            continue
        module_count = int(row["Module_count"])
        a_count = int(row["A_domain_count"])
        core_chain = row["Core_domain_chain"]
        strict = is_strict_nrps_orf(core_chain, module_count, a_count)

        if strict:
            genome_to_orfs[genome].append((idx, seq_id, module_count, a_count))

    orf_to_cluster_id: Dict[str, int] = {}
    cluster_is_nrps: Dict[int, bool] = {}
    cur_cluster_id = 0

    for genome, orf_list in genome_to_orfs.items():
        if not orf_list:
            continue
        orf_list.sort(key=lambda x: x[0])
        current_cluster: List[Tuple[int, str, int, int]] = [orf_list[0]]

        for prev, cur in zip(orf_list, orf_list[1:]):
            prev_idx = prev[0]
            cur_idx = cur[0]
            if cur_idx - prev_idx <= GAP_MAX:
                current_cluster.append(cur)
            else:
                if current_cluster:
                    this_id = cur_cluster_id
                    cur_cluster_id += 1
                    orf_count = len(current_cluster)
                    total_mod = sum(x[2] for x in current_cluster)
                    total_a = sum(x[3] for x in current_cluster)
                    is_cluster = (
                        total_mod >= 3 and
                        orf_count > 0 and
                        (total_a / orf_count) >= 1.0
                    )
                    cluster_is_nrps[this_id] = is_cluster
                    for _, sid, _, _ in current_cluster:
                        orf_to_cluster_id[sid] = this_id
                current_cluster = [cur]

        if current_cluster:
            this_id = cur_cluster_id
            cur_cluster_id += 1
            orf_count = len(current_cluster)
            total_mod = sum(x[2] for x in current_cluster)
            total_a = sum(x[3] for x in current_cluster)
            is_cluster = (
                total_mod >= 3 and
                orf_count > 0 and
                (total_a / orf_count) >= 1.0
            )
            cluster_is_nrps[this_id] = is_cluster
            for _, sid, _, _ in current_cluster:
                orf_to_cluster_id[sid] = this_id

    return orf_to_cluster_id, cluster_is_nrps


def extend_clusters_with_te_from_annotated(
    all_rows: List[dict],
    orf_to_cluster_id: Dict[str, int],
    cluster_is_nrps: Dict[int, bool],
    gap_max: int = 4,
) -> None:
    """
    Extend NRPS clusters by including TE-only ORFs, assigning each to
    the nearest strict NRPS ORF within the same genome (if within gap_max).
    """
    orf_has_te: Dict[str, bool] = {}
    orf_is_nrps: Dict[str, bool] = {}
    orf_idx_by_genome: Dict[str, int] = {}
    genome_to_orfs: Dict[str, List[str]] = defaultdict(list)

    for row in all_rows:
        seq_id = row["Seq_ID"]
        genome, idx = parse_seq_id(seq_id)
        if idx is None:
            continue

        genome_to_orfs[genome].append(seq_id)
        if seq_id not in orf_idx_by_genome:
            orf_idx_by_genome[seq_id] = idx

        nrps_flag = row.get("NRPS_ORF", "").strip()
        if nrps_flag == "Yes":
            orf_is_nrps[seq_id] = True
        elif nrps_flag == "No":
            orf_is_nrps[seq_id] = False

        dn = row["Domain"].strip()
        if dn == "Thioesterase":
            orf_has_te[seq_id] = True

    for genome, seq_ids in genome_to_orfs.items():
        uniq_orfs = sorted(
            {sid for sid in seq_ids if sid in orf_idx_by_genome},
            key=lambda s: orf_idx_by_genome[s],
        )

        backbone: List[Tuple[int, str, int]] = []
        for sid in uniq_orfs:
            if not orf_is_nrps.get(sid, False):
                continue
            if sid not in orf_to_cluster_id:
                continue
            cid = orf_to_cluster_id[sid]
            if not cluster_is_nrps.get(cid, False):
                continue
            backbone.append((orf_idx_by_genome[sid], sid, cid))

        if not backbone:
            continue

        te_candidates: List[Tuple[int, str]] = []
        for sid in uniq_orfs:
            if orf_is_nrps.get(sid, False):
                continue
            if not orf_has_te.get(sid, False):
                continue
            if sid in orf_to_cluster_id:
                continue
            te_candidates.append((orf_idx_by_genome[sid], sid))

        if not te_candidates:
            continue

        backbone.sort(key=lambda x: x[0])

        def find_nearest_backbone(te_idx: int) -> Tuple[int, int]:
            best_cid = None
            best_dist = None
            for idx, sid, cid in backbone:
                dist = abs(idx - te_idx)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_cid = cid
            if best_cid is None or best_dist is None:
                return -1, 10**9
            return best_cid, best_dist

        for te_idx, te_sid in te_candidates:
            cid, dist = find_nearest_backbone(te_idx)
            if dist <= gap_max:
                orf_to_cluster_id[te_sid] = cid
                print(
                    f"[INFO] TE-only ORF {te_sid} (idx={te_idx}) "
                    f"assigned to cluster {cid} (distance={dist})"
                )


# ------- 5. Main pipeline: detect modules & build JSON + extracted_cds.fasta -------

def detect_nrps_modules(csv_in: str, faa_in: str, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)

    all_rows: List[dict] = []
    by_seq: Dict[str, List[dict]] = defaultdict(list)

    with open(csv_in, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = {"Seq_ID", "Domain", "Order", "domain_sequence"}
        if not required_cols.issubset(reader.fieldnames or []):
            raise ValueError(
                f"Input CSV must contain columns: {', '.join(required_cols)}; "
                f"found: {reader.fieldnames}"
            )

        for row in reader:
            all_rows.append(row)
            by_seq[row["Seq_ID"]].append(row)

    if not all_rows:
        raise ValueError("Input CSV contains no data rows.")

    if "." in os.path.basename(csv_in):
        base, ext = os.path.basename(csv_in).rsplit(".", 1)
        annotated_csv = os.path.join(outdir, f"{base}_annotated.{ext}")
        summary_csv = os.path.join(outdir, f"{base}_annotated_summary.{ext}")
    else:
        base = os.path.basename(csv_in)
        annotated_csv = os.path.join(outdir, base + "_annotated.csv")
        summary_csv = os.path.join(outdir, base + "_annotated_summary.csv")

    seq_to_chains: Dict[str, SeqChains] = {}
    seq_to_res: Dict[str, ModuleDetectResult] = {}

    for seq_id, rows in by_seq.items():
        chains = build_linear_chains_for_seq(rows)
        res = find_modules_in_chain(
            chains.core_types,
            chains.core_idx,
            chains.full_chain,
        )
        seq_to_chains[seq_id] = chains
        seq_to_res[seq_id] = res

    summary_rows: List[dict] = []
    for seq_id in sorted(by_seq.keys()):
        res = seq_to_res[seq_id]
        if not res.is_nrps_orf_relaxed:
            continue
        chains = seq_to_chains[seq_id]
        full_str = "-".join(chains.full_chain)
        summary_rows.append({
            "Seq_ID": seq_id,
            "Full_domain_chain": full_str,
            "Core_domain_chain": res.core_chain_str,
            "Module_count": str(res.module_count),
            "A_domain_count": str(res.a_count),
        })

    orf_to_cluster_id, cluster_is_nrps = build_clusters_from_summary(summary_rows)

    # 1) Annotated CSV
    with open(annotated_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(all_rows[0].keys())
        extra_cols = [
            "NRPS_ORF",
            "Cluster_is_NRPS_cluster",
        ]
        for c in extra_cols:
            if c not in fieldnames:
                fieldnames.append(c)

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in all_rows:
            seq_id = row["Seq_ID"]
            res = seq_to_res[seq_id]

            row["NRPS_ORF"] = "Yes" if res.is_nrps_orf_relaxed else "No"

            cluster_flag = "No"
            if seq_id in orf_to_cluster_id:
                cid = orf_to_cluster_id[seq_id]
                if cluster_is_nrps.get(cid, False):
                    cluster_flag = "Yes"
            row["Cluster_is_NRPS_cluster"] = cluster_flag

            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"[INFO] Detailed annotated CSV written to: {annotated_csv}")

    all_rows_with_flags: List[dict] = []
    with open(annotated_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_rows_with_flags.append(row)

    extend_clusters_with_te_from_annotated(
        all_rows=all_rows_with_flags,
        orf_to_cluster_id=orf_to_cluster_id,
        cluster_is_nrps=cluster_is_nrps,
        gap_max=4,
    )

    # 2) Summary CSV (using updated orf_to_cluster_id)
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Cluster_ID",
            "Seq_ID",
            "Full_domain_chain",
            "Core_domain_chain",
            "Module_count",
            "A_domain_count",
            "Cluster_is_NRPS_cluster",
        ])

        cluster_to_orfs: Dict[int, List[str]] = defaultdict(list)
        seqid_to_summary: Dict[str, dict] = {row["Seq_ID"]: row for row in summary_rows}

        for seq_id, old_cid in orf_to_cluster_id.items():
            cluster_to_orfs[old_cid].append(seq_id)

        kept_old_cluster_ids = [
            cid for cid in sorted(cluster_to_orfs.keys())
            if cluster_is_nrps.get(cid, False)
        ]

        old_to_new_cluster_id: Dict[int, int] = {
            old_cid: (new_cid + 1) for new_cid, old_cid in enumerate(kept_old_cluster_ids)
        }

        for old_cid in kept_old_cluster_ids:
            new_cid = old_to_new_cluster_id[old_cid]
            orfs_in_cluster = sorted(cluster_to_orfs[old_cid])
            flag = "Yes"
            for seq_id in orfs_in_cluster:
                if seq_id in seqid_to_summary:
                    row = seqid_to_summary[seq_id]
                    full_chain = row["Full_domain_chain"]
                    core_chain = row["Core_domain_chain"]
                    module_count = row["Module_count"]
                    a_count = row["A_domain_count"]
                else:
                    chains = seq_to_chains.get(seq_id)
                    res = seq_to_res.get(seq_id)
                    if chains is not None:
                        full_chain = "-".join(chains.full_chain)
                    else:
                        full_chain = ""
                    if res is not None:
                        core_chain = res.core_chain_str
                        module_count = str(res.module_count)
                        a_count = str(res.a_count)
                    else:
                        core_chain = ""
                        module_count = "0"
                        a_count = "0"

                writer.writerow([
                    str(new_cid),
                    seq_id,
                    full_chain,
                    core_chain,
                    module_count,
                    a_count,
                    flag,
                ])

    print(f"[INFO] Summary CSV written to:           {summary_csv}")

    # 3) 构建 NRPS_modules.json，并把每个ORF的氨基酸序列写入 orf_seq 字段
    nrps_json_out = os.path.join(outdir, "NRPS_modules.json")
    adomains_fa_out = os.path.join(outdir, "extracted_cds.fasta")

    json_clusters = []

    # 读入ORF氨基酸序列（FASTA id 必须和 Seq_ID 一致）
    seq_dict = {rec.id: str(rec.seq) for rec in SeqIO.parse(faa_in, "fasta")}

    # extracted_cds.fasta 的内容在这里累积
    adomain_records: List[Tuple[str, str]] = []   # (header, seq)

    for old_cid in kept_old_cluster_ids:
        new_cid = old_to_new_cluster_id[old_cid]
        orfs_in_cluster = sorted(cluster_to_orfs[old_cid])
        cluster_entry = {
            "cluster_id": new_cid,
            "orfs": []
        }
        for seq_id in orfs_in_cluster:
            rows_for_orf = by_seq.get(seq_id, [])
            if not rows_for_orf:
                continue
            chains = seq_to_chains.get(seq_id)
            detect_res = seq_to_res.get(seq_id)
            if chains is None or detect_res is None:
                continue

            modules_for_orf = split_modules_for_orf(seq_id, rows_for_orf, chains, detect_res)
            amp_counter = 0

            orf_seq = seq_dict.get(seq_id)

            orf_entry = {
                "seq_id": seq_id,
                "orf_seq": orf_seq,
                "modules": []
            }

            if modules_for_orf:
                for m in modules_for_orf:
                    renamed_shorts: List[str] = []
                    renamed_domains: List[dict] = []

                    for d in m.domains:
                        if d.short == "A":
                            amp_counter += 1
                            new_name = f"AMP-binding.{amp_counter}"
                            new_short = f"A{amp_counter}"

                            # 写入 A-domain 到 extracted_cds.fasta
                            if d.aa_seq:
                                header = f">cluster{new_cid}|{seq_id}|AMP-binding.{amp_counter}"
                                adomain_records.append((header, d.aa_seq))

                        else:
                            new_name = d.name
                            new_short = d.short

                        renamed_shorts.append(new_short)
                        renamed_domains.append({
                            "name": new_name,
                            "short": new_short,
                            "domain_order": d.order,
                            "aa_seq": d.aa_seq,
                        })

                    mod_json = {
                        "module_index": m.module_index,
                        "is_orphan": m.is_orphan,
                        "warnings": m.warnings,
                        "domain_chain": renamed_shorts,
                        "domains": renamed_domains,
                    }
                    orf_entry["modules"].append(mod_json)

            else:
                # 没有被识别到模块，但为了兼容，仍然把所有domain作为一个伪模块输出
                tokens = build_domain_tokens_for_seq(rows_for_orf)
                renamed_domains: List[dict] = []
                renamed_shorts: List[str] = []

                for d in tokens:
                    if d.short == "A":
                        amp_counter += 1
                        new_name = f"AMP-binding.{amp_counter}"
                        new_short = f"A{amp_counter}"

                        # 写入 A-domain 到 extracted_cds.fasta
                        if d.aa_seq:
                            header = f">cluster{new_cid}|{seq_id}|A{amp_counter}"
                            adomain_records.append((header, d.aa_seq))
                    else:
                        new_name = d.name
                        new_short = d.short

                    renamed_shorts.append(new_short)
                    renamed_domains.append({
                        "name": new_name,
                        "short": new_short,
                        "domain_order": d.order,
                        "aa_seq": d.aa_seq,
                    })

                pseudo_mod = {
                    "module_index": 1,
                    "is_orphan": False,
                    "domain_chain": renamed_shorts,
                    "domains": renamed_domains,
                }
                orf_entry["modules"].append(pseudo_mod)

            cluster_entry["orfs"].append(orf_entry)
        json_clusters.append(cluster_entry)

    with open(nrps_json_out, "w", encoding="utf-8") as jf:
        json.dump({"clusters": json_clusters}, jf, ensure_ascii=False, indent=2)
    print(f"[INFO] NRPS module JSON written to:      {nrps_json_out}")

    # 4) 写出 extracted_cds.fasta
    if adomain_records:
        line_width = 60  # 每行氨基酸个数，按需修改，比如 60 / 70 / 80
        with open(adomains_fa_out, "w", encoding="utf-8") as af:
            for header, seq in adomain_records:
                af.write(header + "\n")
                # 将序列按 line_width 分行输出
                for i in range(0, len(seq), line_width):
                    af.write(seq[i:i + line_width] + "\n")
        print(f"[INFO] A-domain FASTA written to:       {adomains_fa_out}")
    else:
        print("[INFO] No A-domains found; adomains.fasta not created.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect NRPS modules from domain CSV and protein FASTA, and output NRPS_modules.json + extracted_cds.fasta.",
    )
    parser.add_argument(
        "-i", "--input-csv",
        required=True,
        help="Input domains CSV (columns: Seq_ID, Domain, Order, domain_sequence, [domain_order])",
    )
    parser.add_argument(
        "-f", "--faa",
        required=True,
        help="Protein FASTA file (proteins.faa).",
    )
    parser.add_argument(
        "-o", "--outdir",
        required=True,
        help="Output directory. Will be created if not existing.",
    )

    args = parser.parse_args()

    csv_in = args.input_csv
    faa_in = args.faa
    outdir = args.outdir

    if not os.path.exists(csv_in):
        sys.exit(f"[ERROR] Input CSV not found: {csv_in}")
    if not os.path.exists(faa_in):
        sys.exit(f"[ERROR] Input FASTA not found: {faa_in}")

    detect_nrps_modules(csv_in, faa_in, outdir)


if __name__ == "__main__":
    main()