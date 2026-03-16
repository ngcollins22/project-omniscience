import itertools
import heapq
import time
from typing import List, Tuple, Dict, Iterable

import numpy as np
import matplotlib.pyplot as plt

from geometry import dop_from_unit_los


def _seconds_between(a, b):
    # a and b may be Orekit AbsoluteDate or numeric seconds
    try:
        # AbsoluteDate.durationFrom(other) returns seconds (double)
        return float(a.durationFrom(b))
    except Exception:
        try:
            return float(a - b)
        except Exception:
            raise


def compute_site_timeseries_metrics(active_indices: Iterable[int],
                                    visible: np.ndarray,
                                    uvec: np.ndarray,
                                    times: List,
                                    ) -> Dict:
    """
    Compute per-site timeseries metrics for a given set of active satellite indices.

    Inputs:
      active_indices: iterable of indices into the N dimension of `visible`/`uvec`
      visible: bool[S, T, N]
      uvec: float[S, T, N, 3]
      times: list of length T (AbsoluteDate or numeric)

    Returns a dict with keys:
      'frac_ge3': array S (fraction of time each site has >=3 sats)
      'frac_ge4': array S
      'pdop_p95': array S
      'pdop_series': list of arrays (S elements each length T, NaN when <4)
      'vis_count_series': array (S, T) counts
      'revisit_stats': list of dicts per site with median/p95/worst (seconds)
    """
    active = np.array(list(active_indices), dtype=int)
    S, T, N = visible.shape

    vis_counts = np.sum(visible[:, :, active], axis=2)  # (S, T)

    frac_ge3 = np.mean(vis_counts >= 3, axis=1)
    frac_ge4 = np.mean(vis_counts >= 4, axis=1)

    pdop_series = [np.full((T,), np.nan, dtype=float) for _ in range(S)]
    # TWR-based DOP (no clock bias) available when m >= 3
    twr_pdop_series = [np.full((T,), np.nan, dtype=float) for _ in range(S)]

    # compute pdop per epoch using all visible active sats
    for s in range(S):
        for t in range(T):
            mask = visible[s, t, active]
            m = np.count_nonzero(mask)
            if m >= 4:
                unit_los = uvec[s, t, active, :][mask]
                p, h = dop_from_unit_los(unit_los)
                pdop_series[s][t] = p

            # TWR DOP fallback when m >= 3 (no clock bias)
            if m >= 3:
                unit_los3 = uvec[s, t, active, :][mask]
                # build G (m x 3)
                G = unit_los3.reshape((-1, 3))
                GTG = G.T @ G
                # check condition and invert
                try:
                    if np.linalg.cond(GTG) < 1e12:
                        Q = np.linalg.inv(GTG)
                        twr_pdop = float(np.sqrt(np.trace(Q)))
                        twr_pdop_series[s][t] = twr_pdop
                except np.linalg.LinAlgError:
                    pass

    pdop_p95 = np.array([
        (float(np.percentile(series[~np.isnan(series)], 95)) if np.any(~np.isnan(series)) else float('inf'))
        for series in pdop_series
    ], dtype=float)

    twr_pdop_p95 = np.array([
        (float(np.percentile(series[~np.isnan(series)], 95)) if np.any(~np.isnan(series)) else float('inf'))
        for series in twr_pdop_series
    ], dtype=float)

    # availability fractions for pdop/twr_pdop
    pdop_avail_frac = np.array([float(np.mean(~np.isnan(series))) for series in pdop_series], dtype=float)
    twr_pdop_avail_frac = np.array([float(np.mean(~np.isnan(series))) for series in twr_pdop_series], dtype=float)

    # revisit stats for visibility >=1 (communications)
    revisit_stats = []
    # compute timestep seconds array for converting index gaps to seconds
    if T >= 2:
        step_sec = _seconds_between(times[1], times[0])
    else:
        step_sec = 0.0

    for s in range(S):
        vis = vis_counts[s] > 0
        # find contiguous contact windows
        edges = np.diff(vis.astype(int))
        starts = np.where(edges == 1)[0] + 1
        ends = np.where(edges == -1)[0]
        if vis[0]:
            starts = np.insert(starts, 0, 0)
        if vis[-1]:
            ends = np.append(ends, T - 1)

        # compute gaps between windows (revisit times)
        gaps = []
        for i in range(len(ends) - 1):
            # time between end of window i and start of window i+1
            gap_steps = starts[i+1] - ends[i]
            gaps.append(gap_steps * step_sec)

        if len(gaps) == 0:
            stats = {'median': float('inf'), 'p95': float('inf'), 'worst': float('inf'), 'mean': float('inf')}
        else:
            stats = {'median': float(np.median(gaps)), 'p95': float(np.percentile(gaps, 95)), 'worst': float(max(gaps)), 'mean': float(np.mean(gaps))}

        revisit_stats.append(stats)

    return {
        'frac_ge3': frac_ge3,
        'frac_ge4': frac_ge4,
        'pdop_p95': pdop_p95,
        'pdop_avail_frac': pdop_avail_frac,
        'twr_pdop_p95': twr_pdop_p95,
        'twr_pdop_avail_frac': twr_pdop_avail_frac,
        'pdop_series': pdop_series,
        'twr_pdop_series': twr_pdop_series,
        'vis_count_series': vis_counts,
        'revisit_stats': revisit_stats,
    }


def _enumerate_launch_subsets(available_indices: List[int],
                              plane_of_index: Dict[int, int],
                              k: int) -> Iterable[Tuple[int, ...]]:
    """
    Enumerate subsets of size k drawn from available_indices such that the satellites
    belong to at most 2 distinct planes (plane_of_index maps index->plane).
    """
    # group available indices by plane
    plane_to_inds: Dict[int, List[int]] = {}
    for idx in available_indices:
        p = plane_of_index[idx]
        plane_to_inds.setdefault(p, []).append(idx)

    planes = list(plane_to_inds.keys())

    # single-plane picks
    for p in planes:
        inds = plane_to_inds[p]
        if len(inds) >= k:
            for comb in itertools.combinations(inds, k):
                yield comb

    # two-plane picks
    for p1, p2 in itertools.combinations(planes, 2):
        inds1 = plane_to_inds[p1]
        inds2 = plane_to_inds[p2]
        n1 = len(inds1)
        n2 = len(inds2)
        # choose i from plane1 and k-i from plane2
        min_i = max(0, k - n2)
        max_i = min(n1, k)
        for i in range(min_i, max_i + 1):
            j = k - i
            if j < 0 or j > n2:
                continue
            for comb1 in itertools.combinations(inds1, i) if i > 0 else [()]:
                for comb2 in itertools.combinations(inds2, j) if j > 0 else [()]:
                    yield tuple(list(comb1) + list(comb2))


def beam_search_deployment(sat_indices: List[int],
                           plane_of_index: Dict[int, int],
                           visible: np.ndarray,
                           uvec: np.ndarray,
                           times: List,
                           stages: List[int] = [4, 8, 8, 4],
                           beam_size: int = 200,
                           top_k_refine: int = 10,
                           score_primary='frac_ge3') -> List[Tuple[float, List[Tuple[int,...]], Dict]]:
    """
    Beam-search over deployment stages. Returns top candidates (score, list_of_stage_subsets, metrics_summary)

    The search is lexicographic: primary key is mean fraction >=3 across sites (higher better), then mean PDOP P95 (lower better).
    """
    S, T, N = visible.shape

    # Helper to compute summary metrics for a full deployment (union of stages)
    def evaluate_full(deployment_union: List[int]):
        metrics = compute_site_timeseries_metrics(deployment_union, visible, uvec, times)
        mean_frac_ge3 = float(np.mean(metrics['frac_ge3']))
        mean_frac_ge4 = float(np.mean(metrics['frac_ge4']))
        mean_pdop_p95 = float(np.mean(metrics['pdop_p95']))
        worst_outage = float(np.max(1.0 - metrics['frac_ge4']))
        # score tuple (primary to minimize)
        score = (-mean_frac_ge3, mean_pdop_p95, worst_outage)
        return score, metrics

    # initial beam: empty deployment
    beam = [ ((), [], None) ]  # tuple of (available_set_tuple, list_of_stage_subsets, last_eval_metrics)

    available_all = set(sat_indices)

    for stage_idx, stage_k in enumerate(stages):
        print(f"Stage {stage_idx+1}/{len(stages)}: adding {stage_k} sats — beam size {beam_size}")
        new_beam_candidates = []
        start_t = time.time()
        for avail_tuple, prev_subsets, _ in beam:
            used = set().union(*[set(x) for x in prev_subsets]) if prev_subsets else set()
            available = list(available_all - used)

            # enumerate feasible subsets for this launch
            for subset in _enumerate_launch_subsets(available, plane_of_index, stage_k):
                new_subsets = prev_subsets + [tuple(subset)]
                union = list(used.union(subset))
                score, metrics = evaluate_full(union)
                # push into heap keyed by score tuple
                heapq.heappush(new_beam_candidates, (score, new_subsets, metrics))
                # keep heap bounded to beam_size
                if len(new_beam_candidates) > beam_size * 5:
                    # prune to reduce memory (keep best beam_size*2 candidates)
                    new_beam_candidates = heapq.nsmallest(beam_size * 2, new_beam_candidates, key=lambda x: x[0])

        # select best beam_size candidates
        new_beam_candidates = heapq.nsmallest(beam_size, new_beam_candidates, key=lambda x: x[0])
        beam = [ (tuple(sorted(set().union(*[set(x) for x in b[1]]))) , b[1], b[2]) for b in new_beam_candidates ]
        dur = time.time() - start_t
        print(f"  Enumerated candidates; beam reduced to {len(beam)} items in {dur:.1f}s")

    # After final stage, evaluate and return sorted results
    results = []
    for _, subsets, metrics in beam:
        union = list(set().union(*[set(x) for x in subsets]))
        score, metrics = evaluate_full(union)
        results.append((score, subsets, metrics))

    # sort results
    results.sort(key=lambda x: x[0])

    # convert score tuple to a single float primary for compatibility (neg mean frac 3)
    final = []
    for score, subsets, metrics in results:
        prim = -score[0]
        final.append((prim, subsets, metrics))

    return final


def dump_summary_csv(out_path: str, subsets: List[Tuple[int,...]], metrics: Dict, sat_ids: List[int]):
    import csv
    # write a simple summary per site
    S = metrics['vis_count_series'].shape[0]
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        hdr = ['site_idx', 'frac_ge3', 'frac_ge4', 'pdop_p95', 'revisit_median', 'revisit_p95', 'revisit_worst']
        writer.writerow(hdr)
        for s in range(S):
            rs = metrics['revisit_stats'][s]
            writer.writerow([s, metrics['frac_ge3'][s], metrics['frac_ge4'][s], metrics['pdop_p95'][s], rs['median'], rs['p95'], rs['worst']])


def plot_site_timeseries(save_prefix: str, times: List, metrics: Dict, site_idx: int):
    # times to numeric seconds relative to first
    T = len(times)
    try:
        t0 = times[0]
        times_s = np.array([float(t.durationFrom(t0)) for t in times])
    except Exception:
        times_s = np.arange(T)

    vis_counts = metrics['vis_count_series'][site_idx]
    pdop = metrics['pdop_series'][site_idx]

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(times_s, vis_counts, label='visible_count')
    ax1.set_xlabel('seconds')
    ax1.set_ylabel('visible count')

    ax2 = ax1.twinx()
    ax2.plot(times_s, pdop, color='orange', label='pdop')
    ax2.set_ylabel('PDOP')

    plt.title(f'site_{site_idx}_timeseries')
    fig.tight_layout()
    png = f"{save_prefix}_site{site_idx}.png"
    fig.savefig(png)
    plt.close(fig)
