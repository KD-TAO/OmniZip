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

    if remaining.numel() > 0 and contextual_num > 0:
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

def omnizip_istm(video_feature, num_tokens_per_frame=196, merging_ratio=0.7):
    num_frames = video_feature.shape[0] // num_tokens_per_frame
    mask = torch.zeros(video_feature.shape[0], dtype=torch.bool, device=video_feature.device)

    def dpcknn(tokens, keep_rate=0.5, k=5):
        N = tokens.shape[0]
        num_keep = int(N * keep_rate)
        if num_keep >= N:
            return torch.arange(N, device=tokens.device)
        with torch.no_grad():
            normed = torch.nn.functional.normalize(tokens, dim=1)
            sim = torch.mm(normed, normed.T)
            sim.fill_diagonal_(-float('inf'))
            knn_vals, _ = torch.topk(sim, k, dim=1)
            knn_dist = knn_vals.mean(dim=1)
            selected = torch.topk(-knn_dist, num_keep, largest=True).indices
        return selected

    keep_ratio = 1 - merging_ratio

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
            mask[start_idx:end_idx][keep_idx] = True
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
            mask[start_idx:end_idx][keep_idx] = True

    return mask

def omnizip(
    input_embeds: torch.Tensor,
    attn_logits: torch.Tensor,
    input_ids: torch.Tensor,
    audio_token_id: int,
    video_token_id: int,
    num_input_frames: int,
    merging_ratio_audio: float = 0.5,
    merging_ratio_v: float = 0.5,
    contextual_ratio: float = 0.05,
    g: int = 3,
):
    device = input_embeds.device
    merging_ratio = merging_ratio_audio
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

    video_token_per_frame = video_feature.shape[0] // num_input_frames

    audio_mask, merge_plan = omnizip_audio_attn(
        audio_feature=audio_feature,
        video_feature=video_feature,
        attn_logits=attn_logits,
        merging_ratio=merging_ratio,
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

    if num_input_frames % 4 == 0:
        group_count = num_input_frames // 4
        num_video_tokens_per_group = max(1, video_feature.shape[0] // group_count)
        num_audio_tokens_per_group = max(1, audio_feature.shape[0] // group_count)

        audio_group_retention = []
        for i in range(group_count):
            start = i * num_audio_tokens_per_group
            end = (i + 1) * num_audio_tokens_per_group if i < group_count - 1 else audio_feature.shape[0]
            group_mask = audio_mask[start:end]
            ratio = group_mask.float().mean().item() if group_mask.numel() > 0 else 0.0
            audio_group_retention.append(ratio)

        n_groups = len(audio_group_retention)
        min_ratio, max_ratio = 0.35, 0.75

        base_vs = []
        for ret in audio_group_retention:
            mapped = max_ratio + (min_ratio - max_ratio) * ret
            base_vs.append(max(min_ratio, min(max_ratio, mapped)))

        target_total = merging_ratio_v * n_groups
        base_total = sum(base_vs)

        if abs(base_total - target_total) > 1e-3 and base_total > 0 and n_groups > 2:
            min_idx = base_vs.index(min(base_vs))
            max_idx = base_vs.index(max(base_vs))
            fixed_vs = base_vs.copy()
            min_val, max_val = fixed_vs[min_idx], fixed_vs[max_idx]
            other_indices = [i for i in range(n_groups) if i != min_idx and i != max_idx]
            other_sum = sum(fixed_vs[i] for i in other_indices)
            if other_indices and other_sum > 0:
                remain_target = target_total - min_val - max_val
                scaling = remain_target / other_sum
                adjusted_vs = []
                for i in range(n_groups):
                    if i == min_idx:
                        adjusted_vs.append(min_val)
                    elif i == max_idx:
                        adjusted_vs.append(max_val)
                    else:
                        v_new = fixed_vs[i] * scaling
                        adjusted_vs.append(max(min_ratio, min(max_ratio, v_new)))
                diff = target_total - sum(adjusted_vs)
                if abs(diff) > 1e-3:
                    adj = diff / len(other_indices) if other_indices else 0
                    for i in other_indices:
                        adjusted_vs[i] = max(
                            min_ratio, min(max_ratio, adjusted_vs[i] + adj)
                        )
            else:
                adjusted_vs = base_vs
        else:
            adjusted_vs = base_vs

        video_merging_ratios = adjusted_vs

        video_group_masks = []
        for i, video_merging_ratio in zip(range(group_count), video_merging_ratios):
            v_start = i * num_video_tokens_per_group
            v_end = (i + 1) * num_video_tokens_per_group if i < group_count - 1 else video_feature.shape[0]
            group_feat = video_feature[v_start:v_end]
            group_len = group_feat.size(0)

            if group_len % 4 == 0:
                group_mask = omnizip_istm(
                    group_feat, num_tokens_per_frame=video_token_per_frame, merging_ratio=video_merging_ratio
                )
            else:
                group_mask = torch.ones(group_len, dtype=torch.bool, device=group_feat.device)

            video_group_masks.append(group_mask)

        video_mask = torch.cat(video_group_masks, dim=0)
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


    num_video_tokens = video_feature.shape[0]
    num_audio_tokens = audio_feature.shape[0]

    VIDEO_GROUP_SIZE = video_token_per_frame * 4
    AUDIO_GROUP_SIZE = 50

    video_groups = []
    audio_groups = []
    v_ptr = a_ptr = 0

    while v_ptr + VIDEO_GROUP_SIZE <= num_video_tokens and a_ptr + AUDIO_GROUP_SIZE <= num_audio_tokens:
        video_groups.append((v_ptr, v_ptr + VIDEO_GROUP_SIZE))
        audio_groups.append((a_ptr, a_ptr + AUDIO_GROUP_SIZE))
        v_ptr += VIDEO_GROUP_SIZE
        a_ptr += AUDIO_GROUP_SIZE

    if v_ptr < num_video_tokens:
        if a_ptr < num_audio_tokens:
            video_groups.append((v_ptr, num_video_tokens))
            audio_groups.append((a_ptr, num_audio_tokens))
        else:
            video_groups.append((v_ptr, num_video_tokens))
            audio_groups.append((a_ptr, a_ptr))
    elif a_ptr < num_audio_tokens:
        video_groups.append((v_ptr, v_ptr))
        audio_groups.append((a_ptr, num_audio_tokens))

    assert len(video_groups) == len(audio_groups)
    group_num = len(video_groups)

    audio_group_retention = []
    for a_start, a_end in audio_groups:
        if a_start == a_end:
            ratio = 1.0
        else:
            group_mask = audio_mask[a_start:a_end]
            ratio = group_mask.float().mean().item() if group_mask.numel() > 0 else 0.0
        audio_group_retention.append(ratio)

    n_groups = len(audio_group_retention)
    min_ratio, max_ratio = 0.35, 0.75

    base_vs = []
    for ret in audio_group_retention:
        mapped = max_ratio + (min_ratio - max_ratio) * ret
        base_vs.append(max(min_ratio, min(max_ratio, mapped)))

    target_total = merging_ratio_v * n_groups
    base_total = sum(base_vs)

    if abs(base_total - target_total) > 1e-3 and base_total > 0 and n_groups > 2:
        min_idx = base_vs.index(min(base_vs))
        max_idx = base_vs.index(max(base_vs))
        fixed_vs = base_vs.copy()
        min_val, max_val = fixed_vs[min_idx], fixed_vs[max_idx]
        other_indices = [i for i in range(n_groups) if i != min_idx and i != max_idx]
        other_sum = sum(fixed_vs[i] for i in other_indices)
        if other_indices and other_sum > 0:
            remain_target = target_total - min_val - max_val
            scaling = remain_target / other_sum
            adjusted_vs = []
            for i in range(n_groups):
                if i == min_idx:
                    adjusted_vs.append(min_val)
                elif i == max_idx:
                    adjusted_vs.append(max_val)
                else:
                    v_new = fixed_vs[i] * scaling
                    adjusted_vs.append(max(min_ratio, min(max_ratio, v_new)))
            diff = target_total - sum(adjusted_vs)
            if abs(diff) > 1e-3:
                adj = diff / len(other_indices) if other_indices else 0
                for i in other_indices:
                    adjusted_vs[i] = max(
                        min_ratio, min(max_ratio, adjusted_vs[i] + adj)
                    )
        else:
            adjusted_vs = base_vs
    else:
        adjusted_vs = base_vs

    video_merging_ratios = adjusted_vs

    video_group_masks = []
    for idx, ((v_start, v_end), (a_start, a_end), video_merging_ratio) in enumerate(
        zip(video_groups, audio_groups, video_merging_ratios)
    ):
        group_feat = video_feature[v_start:v_end]
        group_len = group_feat.size(0)
        is_tail_video_group = (group_len != VIDEO_GROUP_SIZE)
        if group_len == 0:
            group_mask = torch.zeros(0, dtype=torch.bool, device=video_feature.device)
        elif is_tail_video_group:
            group_mask = torch.ones(group_len, dtype=torch.bool, device=video_feature.device)
        else:
            group_mask = omnizip_istm(
                group_feat, num_tokens_per_frame=video_token_per_frame, merging_ratio=video_merging_ratio
            )
        video_group_masks.append(group_mask)

    video_mask = torch.cat(video_group_masks, dim=0)
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






