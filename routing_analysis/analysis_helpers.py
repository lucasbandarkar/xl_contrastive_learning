import torch
import torch.nn.functional as F
import itertools

# --- JSD Main Function ---
def js_divergence(prob_p,prob_q):
    """
    Calculates Jensen-Shannon Divergence JSD(P || Q)
    between two sets of probabilities.

    Args:
        prob_p (torch.Tensor): Probabilities for distribution P. Shape: [..., D]
        prob_q (torch.Tensor): Probabilities for distribution Q. Shape: [..., D]

    Returns:
        torch.Tensor: JSD scores. Shape: [...] (one JSD score for each pair of distributions)
    """
    # 3. Calculate M = 0.5 * (P + Q)
    m_probs = 0.5 * (prob_p + prob_q)

    # 4. Handle potential log(0) issues for M.
    #    Clamp very small values to a minimum to avoid log(0) = -inf,
    #    which can happen if P and Q are extremely sparse or if there are numerical instabilities.
    m_probs = torch.clamp(m_probs, min=1e-10) # Ensure no zero values before taking log

    # 5. Calculate log_M
    log_m_probs = torch.log(m_probs)

    # 6. Calculate KLD(P || M) and KLD(Q || M)
    # Note for Lucas: 1st argument is logprob, second is probability
    kld_pm = F.kl_div(log_m_probs, prob_p, reduction='none').sum(dim=-1)
    kld_qm = F.kl_div(log_m_probs, prob_q, reduction='none').sum(dim=-1)

    # 7. Calculate JSD: 0.5 * (KLD(P || M) + KLD(Q || M))
    jsd = 0.5 * (kld_pm + kld_qm)
    return jsd

def calculate_entropy_normalized_jsd(p, q):
    """
    Calculate entropy-controlled JS divergence.
    
    Args:
        p, q: probability distributions
        method: 'residual', 'normalized', or 'conditional'
    """
    js = js_divergence(p, q)
    
    # Normalize by theoretical maximum JS at this entropy level
    p_entropy = calculate_entropy(p)
    q_entropy = calculate_entropy(q)
    avg_entropy = 0.5 * (p_entropy + q_entropy)
    
    # Theoretical max JS for distributions with this entropy
    # This is approximate - exact calculation is complex
    max_possible_js = torch.log(torch.tensor(p.size(-1), device=p.device, dtype=p.dtype)) - avg_entropy
    max_possible_js = torch.clamp(max_possible_js, min=1e-6)
    
    normalized_js = js / max_possible_js
    return normalized_js

def pairwise_mean_jsd(expert_importance_dict, from_english=True, entropy_normalized=False):
    # takes in a dictionary expert_importance_dict where the keys are the language
    # the values are a list of N tensors, each of shape (L, E)
    # where N is the number of sequences in the data
    # L is the number of layers in the model
    # E is the number of experts per layer
    eng_code = 'en' if len(expert_importance_dict.keys())==10 else 'eng'
    if from_english:
        mean_jsds_per_layer_from_eng = {} # better data struct ?
        for lang in expert_importance_dict.keys():
            if lang == eng_code:
                continue
            jsd_tensor = pairwise_jsd(expert_importance_dict[eng_code], expert_importance_dict[lang], entropy_normalized)
            mean_jsds_per_layer_from_eng[lang] = torch.mean(jsd_tensor, dim=0)
        return mean_jsds_per_layer_from_eng
    else:
        mean_jsds_per_layer_pairwise = {}
        for lang1, lang2 in itertools.combinations(expert_importance_dict.keys(), 2):
            jsd_tensor = pairwise_jsd(expert_importance_dict[lang1], expert_importance_dict[lang2], entropy_normalized)
            mean_jsds_per_layer_pairwise[(lang1, lang2)] = torch.mean(jsd_tensor, dim=0)
        return mean_jsds_per_layer_pairwise



def pairwise_jsd(expert_importance1, expert_importance2, entropy_normalized=False):
    # --- Calculate JSD for each corresponding pair of tensors ---
    # the input lists contain N tensors, each of shape (L, E)
    # where N is the number of sequences in the data
    # L is the number of layers in the model
    # E is the number of experts per layer

    # return: a tensor of size L
    jsd_scores_list = []

    for i in range(len(expert_importance1)):
        tensor_probs1 = expert_importance1[i] # Shape: (L, E) (e.g., [48, 128])
        tensor_probs2 = expert_importance2[i] # Shape: (L, E) (e.g., [48, 128])

        # calculate_jsd processes each of the L rows (distributions)
        # It returns a tensor of shape [48]
        if not entropy_normalized:
            jsd_per_row_tensor = js_divergence(tensor_probs1, tensor_probs2)
        else:
            jsd_per_row_tensor = calculate_entropy_normalized_jsd(tensor_probs1, tensor_probs2)
        jsd_scores_list.append(jsd_per_row_tensor)

    # --- Combine all JSD scores into a single tensor ---
    # Stack the N tensors (each of shape [48]) along a new dimension
    jsd_tensor = torch.stack(jsd_scores_list)

        # --- Verify the final shape ---
    print(f"\nShape of the final JSD tensor: {jsd_tensor.shape}")
    return jsd_tensor

def calculate_entropy(probs):
    """
    Calculate entropy from probabilities.
    
    Args:
        probs: torch.Tensor of shape (..., num_experts) - probabilities that sum to 1
    
    Returns:
        entropy: torch.Tensor of shape (...,)
    """
    eps = 1e-10
    entropy = -torch.sum(probs * torch.log(probs + eps), dim=-1)
    
    return entropy