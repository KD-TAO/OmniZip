import torch
from typing import Dict, List, Tuple, Optional

def omnizip_audio_attn(
    audio_feature: torch.Tensor,
    video_feature: Optional[torch.Tensor],
    attn_logits: torch.Tensor,
    merging_ratio: float = 0.5,
    contextual_ratio: float = 0.03,
    g: int = 3,
) -> Tuple[torch.Tensor, Dict[int, List[int]]]:
    device = attn_logits.device
    def to_seq(x):
        if x is None:
            return None
        return x.mean(0) if x.dim() == 3 else x

    a = to_seq(audio_feature).to(device)
    v = to_seq(video_feature).to(device) if video_feature is not None else None
    N = a.size(0)
    if N == 0:
        return torch.zeros(0, dtype=torch.bool, device=device), {}

    keep_mask = torch.zeros(N, dtype=torch.bool, device=device)
    dominant_num = int(max(0, min(N, round((1.0 - merging_ratio) * N))))
    if dominant_num > 0:
        _, topk = torch.topk(attn_logits, dominant_num)
        keep_mask[topk] = True

    all_idx = torch.arange(N, device=device)
    remaining = all_idx[~keep_mask]
    contextual_num = int(max(0, round(contextual_ratio * N)))
    merge_plan: Dict[int, List[int]] = {}

    if remaining.numel() > 0 and contextual_num > 0 and g > 0:
        contextual_num = min(contextual_num, remaining.numel())
        step = max(1, remaining.numel() // contextual_num)
        init_pos = torch.arange(0, remaining.numel(), step, device=device)[:contextual_num]
        anchors = remaining[init_pos]
        keep_mask[anchors] = True

        rem_local_mask = torch.ones(remaining.numel(), dtype=torch.bool, device=device)
        rem_local_mask[init_pos] = False
        pool_global = remaining[torch.arange(remaining.numel(), device=device)[rem_local_mask]]

        if pool_global.numel() > 0 and g > 0:
            a_norm = a / (a.norm(dim=-1, keepdim=True) + 1e-6)
            sim_aa = a_norm[pool_global] @ a_norm[anchors].T
            assign = sim_aa.argmax(dim=1)
            
            if v is not None and v.numel() > 0:
                v_norm = v / (v.norm(dim=-1, keepdim=True) + 1e-6)
                sim_av = a_norm[pool_global] @ v_norm.T
                scores = sim_av.max(dim=1).values
            else:
                scores = sim_aa.max(dim=1).values
                
            C = anchors.numel()
            for c in range(C):
                mask_c = (assign == c)
                cand = pool_global[mask_c]
                if cand.numel() == 0:
                    merge_plan[int(anchors[c].item())] = []
                    continue
                scores_c = scores[mask_c]
                topg = min(g, cand.numel())
                _, sel = torch.topk(scores_c, topg, largest=True)
                chosen = cand[sel].tolist()
                merge_plan[int(anchors[c].item())] = chosen

    return keep_mask, merge_plan

def omnizip_istm(video_feature: torch.Tensor, num_tokens_per_frame: int, merging_ratio: float):
    num_frames = video_feature.shape[0] // num_tokens_per_frame
    mask = torch.zeros(video_feature.shape[0], dtype=torch.bool, device=video_feature.device)

    def dpcknn(tokens, keep_rate=0.5, k=5):
        N = tokens.shape[0]
        num_keep = int(N * keep_rate)
        if num_keep >= N:
            return torch.arange(N, device=tokens.device)
        k = min(k, N - 1)
        with torch.no_grad():
            normed = torch.nn.functional.normalize(tokens, dim=1)
            dist = 1 - torch.mm(normed, normed.T)
            dist.fill_diagonal_(float('inf'))

            # ρ: local density via KNN
            knn_dist, _ = torch.topk(dist, k, dim=1, largest=False)
            rho = torch.exp(-knn_dist.mean(dim=1))

            # δ: distance to nearest higher-density point
            higher_mask = rho.unsqueeze(0) > rho.unsqueeze(1)  # [i,j]: rho[j] > rho[i]
            delta_dist = dist.clone()
            delta_dist[~higher_mask] = float('inf')
            delta = delta_dist.min(dim=1).values

            # Highest density point: δ = max distance to any other point
            is_peak = delta.isinf()
            if is_peak.any():
                finite_dist = dist.clone()
                finite_dist.fill_diagonal_(0)
                delta[is_peak] = finite_dist[is_peak].max(dim=1).values

            score = delta * rho
            selected = torch.topk(score, num_keep).indices
        return selected

    keep_ratio = 1.0 - merging_ratio

    for t in range(num_frames):
        start_idx = t * num_tokens_per_frame
        end_idx = (t + 1) * num_tokens_per_frame
        tokens = video_feature[start_idx:end_idx]

        if t % 2 == 0:
            spatial_keep_rate = keep_ratio
            num_keep = int(num_tokens_per_frame * spatial_keep_rate)
            if num_keep < num_tokens_per_frame:
                keep_idx = dpcknn(tokens, keep_rate=spatial_keep_rate, k=5)
            else:
                keep_idx = torch.arange(num_tokens_per_frame, device=tokens.device)
            mask[start_idx + keep_idx] = True
        else:
            prev_tokens = video_feature[(t - 1) * num_tokens_per_frame : t * num_tokens_per_frame]
            prev_norm = torch.nn.functional.normalize(prev_tokens, p=2, dim=1)
            curr_norm = torch.nn.functional.normalize(tokens, p=2, dim=1)
            
            similarity = torch.nn.functional.cosine_similarity(curr_norm, prev_norm, dim=1)
            num_keep = int(num_tokens_per_frame * keep_ratio)
            
            if num_keep < num_tokens_per_frame:
                keep_idx = similarity.topk(num_keep, largest=False).indices
            else:
                keep_idx = torch.arange(num_tokens_per_frame, device=tokens.device)
            mask[start_idx + keep_idx] = True

    return mask

def omnizip(
    input_embeds: torch.Tensor,
    attn_logits: torch.Tensor,
    input_ids: torch.Tensor,
    audio_token_id: int,
    video_token_id: int,
    video_grid_thw: Tuple[int, int, int],
    audio_tokens_per_sec: int = 25,
    merging_ratio_audio: float = 0.5,
    merging_ratio_v: float = 0.5,
    contextual_ratio: float = 0.05,
    g: int = 3,
):
    device = input_embeds.device
    is_batched = input_embeds.dim() == 3
    if is_batched:
        B, L, D = input_embeds.shape
        flat_embeds = input_embeds.reshape(-1, D)
        flat_ids = input_ids.reshape(-1)
    else:
        L, D = input_embeds.shape
        flat_embeds = input_embeds
        flat_ids = input_ids

    video_token_mask = (flat_ids == video_token_id).to(device)
    audio_token_mask = (flat_ids == audio_token_id).to(device)

    video_indices = torch.nonzero(video_token_mask, as_tuple=True)[0]
    audio_indices = torch.nonzero(audio_token_mask, as_tuple=True)[0]

    video_feature = flat_embeds[video_indices]
    audio_feature = flat_embeds[audio_indices]

    T = video_grid_thw[0, 0].item()
    W = video_grid_thw[0, 1].item() // 2
    H = video_grid_thw[0, 2].item() // 2
    video_token_per_frame = H * W
    num_video_tokens = T * video_token_per_frame
    
    audio_mask, merge_plan = omnizip_audio_attn(
        audio_feature=audio_feature,
        video_feature=video_feature,
        attn_logits=attn_logits,
        merging_ratio=merging_ratio_audio,
        contextual_ratio=contextual_ratio,
        g=g,
    )

    if g > 0 and len(merge_plan) > 0:
        a_norm = audio_feature / (audio_feature.norm(dim=-1, keepdim=True) + 1e-6)
        v_norm = (
            video_feature / (video_feature.norm(dim=-1, keepdim=True) + 1e-6)
            if video_feature.numel() > 0 else None
        )
        for anchor_rel_idx, merge_rel_list in merge_plan.items():
            if not merge_rel_list:
                continue
            merge_rel = torch.tensor(merge_rel_list, device=device, dtype=torch.long)
            if v_norm is not None:
                scores = (a_norm[merge_rel] @ v_norm.T).max(dim=1).values
            else:
                scores = (a_norm[merge_rel] @ a_norm[anchor_rel_idx].unsqueeze(-1)).squeeze(-1)
            w = torch.softmax(scores, dim=0)
            anchor_vec = audio_feature[anchor_rel_idx]
            merged_vec = (audio_feature[merge_rel] * w.unsqueeze(-1)).sum(dim=0)
            new_anchor = (anchor_vec + merged_vec) / (1.0 + w.sum())
            anchor_global_idx = audio_indices[anchor_rel_idx]
            flat_embeds[anchor_global_idx] = new_anchor

    # Detect time chunks from input_ids interleaving pattern
    def detect_token_chunks(indices):
        """Split sorted token indices into consecutive runs (time chunks)."""
        if indices.numel() == 0:
            return []
        diffs = indices[1:] - indices[:-1]
        boundaries = (diffs > 1).nonzero(as_tuple=True)[0] + 1
        chunks = []
        prev = 0
        for b in boundaries.cpu().tolist():
            chunks.append((prev, b))
            prev = b
        chunks.append((prev, indices.numel()))
        return chunks

    video_chunks = detect_token_chunks(video_indices)
    audio_chunks = detect_token_chunks(audio_indices)

    num_video_chunks = len(video_chunks)
    num_audio_chunks = len(audio_chunks)
    num_guided = min(num_video_chunks, num_audio_chunks)

    # Audio retention for guided groups (where audio exists)
    audio_group_retention = []
    for i in range(num_guided):
        a_start, a_end = audio_chunks[i]
        group_mask = audio_mask[a_start:a_end]
        ratio = group_mask.float().mean().item() if group_mask.numel() > 0 else 0.0
        audio_group_retention.append(ratio)

    min_ratio, max_ratio = 0.35, 0.75
    base_vs_guided = [
        max(min_ratio, min(max_ratio, max_ratio + (min_ratio - max_ratio) * ret))
        for ret in audio_group_retention
    ]

    # Budget normalization: guided groups share the guided portion of the budget
    guided_video_tokens = sum(video_chunks[i][1] - video_chunks[i][0] for i in range(num_guided))
    unguided_video_tokens = sum(
        video_chunks[i][1] - video_chunks[i][0] for i in range(num_guided, num_video_chunks)
    )
    target_discard_tokens = merging_ratio_v * num_video_tokens
    unguided_discard = merging_ratio_v * unguided_video_tokens
    guided_target_discard = target_discard_tokens - unguided_discard

    adjusted_vs_guided = base_vs_guided.copy()
    if num_guided > 2 and guided_video_tokens > 0:
        guided_group_sizes = [video_chunks[i][1] - video_chunks[i][0] for i in range(num_guided)]
        base_guided_discard = sum(
            base_vs_guided[i] * guided_group_sizes[i] for i in range(num_guided)
        )

        if abs(base_guided_discard - guided_target_discard) > 1e-3 and base_guided_discard > 0:
            min_idx = base_vs_guided.index(min(base_vs_guided))
            max_idx = base_vs_guided.index(max(base_vs_guided))
            min_val, max_val = base_vs_guided[min_idx], base_vs_guided[max_idx]
            other_indices = [i for i in range(num_guided) if i != min_idx and i != max_idx]
            other_discard = sum(base_vs_guided[i] * guided_group_sizes[i] for i in other_indices)

            if other_indices and other_discard > 0:
                fixed_discard = min_val * guided_group_sizes[min_idx] + max_val * guided_group_sizes[max_idx]
                remain_target = guided_target_discard - fixed_discard
                scaling = remain_target / other_discard
                for i in other_indices:
                    v_new = base_vs_guided[i] * scaling
                    adjusted_vs_guided[i] = max(min_ratio, min(max_ratio, v_new))

                actual_discard = sum(
                    adjusted_vs_guided[i] * guided_group_sizes[i] for i in range(num_guided)
                )
                diff = guided_target_discard - actual_discard
                if abs(diff) > 1e-3:
                    total_other_size = sum(guided_group_sizes[i] for i in other_indices)
                    if total_other_size > 0:
                        adj_per_token = diff / total_other_size
                        for i in other_indices:
                            adjusted_vs_guided[i] = max(
                                min_ratio, min(max_ratio, adjusted_vs_guided[i] + adj_per_token)
                            )

    # Apply ISTC compression per chunk
    video_group_masks = []
    for i in range(num_video_chunks):
        v_start, v_end = video_chunks[i]
        group_feat = video_feature[v_start:v_end]

        num_frames_in_group = group_feat.shape[0] // video_token_per_frame
        if num_frames_in_group == 0:
            video_group_masks.append(torch.ones(group_feat.shape[0], dtype=torch.bool, device=device))
            continue

        if i < num_guided:
            ratio = adjusted_vs_guided[i]
        else:
            ratio = merging_ratio_v

        group_mask = omnizip_istm(
            group_feat,
            num_tokens_per_frame=video_token_per_frame,
            merging_ratio=ratio,
        )
        video_group_masks.append(group_mask)

    video_mask = torch.cat(video_group_masks, dim=0) if video_group_masks else torch.ones(0, dtype=torch.bool, device=device)
    global_mask = torch.ones(flat_embeds.size(0), dtype=torch.bool, device=device)

    assert video_mask.shape[0] == video_indices.shape[0]
    assert audio_mask.shape[0] == audio_indices.shape[0]

    global_mask[video_indices] = video_mask
    global_mask[audio_indices] = audio_mask

    if is_batched:
        input_embeds_out = flat_embeds.reshape(B, L, D)
    else:
        input_embeds_out = flat_embeds

    return input_embeds_out, global_mask