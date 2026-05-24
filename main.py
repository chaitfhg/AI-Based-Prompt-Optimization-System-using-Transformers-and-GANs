"""
Transformer + GAN Prompt Optimizer
===================================
Neural architecture replacing rule-based scoring & mutation.

Architecture:
  - Transformer Encoder  → Discriminator (scores prompts 0-10 per criterion)
  - Transformer Decoder  → Generator     (mutates / synthesizes new prompts)
  - GAN training loop    → Generator improves against Discriminator feedback
  - Streamlit UI         → same UX as original

Install:
    pip install streamlit pandas matplotlib torch

Run:
    streamlit run app.py
"""

import streamlit as st
import random, re, json, copy, time, math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# ─────────────────────────────────────────────────────────
# CONSTANTS  (same taxonomy as original)
# ─────────────────────────────────────────────────────────

DOMAINS = {
    "general":          "General / cross-industry",
    "legal":            "Legal",
    "finance":          "Finance / banking",
    "healthcare":       "Healthcare / clinical",
    "software":         "Software engineering",
    "marketing":        "Marketing / copywriting",
    "hr":               "HR / people operations",
    "research":         "Academic / research",
    "customer_support": "Customer support",
    "sales":            "Sales",
    "product":          "Product management",
    "data_science":     "Data science / analytics",
}
FORMATS = {
    "flexible":     "Let optimizer decide",
    "bullet":       "Bullet points",
    "prose":        "Prose paragraphs",
    "structured":   "Structured (headers + sections)",
    "json":         "JSON / machine-readable",
    "table":        "Table format",
    "step_by_step": "Step-by-step numbered list",
    "executive":    "Executive summary (BLUF)",
}
TONES = {
    "professional":  "Professional",
    "formal":        "Formal / academic",
    "consultative":  "Consultative",
    "friendly":      "Friendly / approachable",
    "direct":        "Direct / assertive",
    "empathetic":    "Empathetic",
    "technical":     "Technical / precise",
}

# Vocabulary used to featurise text → token IDs ─────────────
VOCAB_TOKENS = [
    # clarity
    "specifically","clearly","concisely","accurately","precisely","step-by-step",
    "comprehensive","thorough","must","should","always","never","ensure","only",
    "exactly","explicitly","focus","prioritize",
    # structure
    "bullet","list","numbered","format","section","header","table","json","markdown",
    "summary","outline","heading","paragraph","first","then","finally","next",
    # role
    "you","are","act","as","role","expert","specialist","professional","analyst",
    "advisor","assistant","consultant","engineer","researcher",
    # audience
    "audience","user","reader","client","customer","beginner","team","executive",
    "manager","stakeholder","non-technical","technical","doctor","lawyer","developer",
    # constraints
    "not","avoid","never","without","exclude","limit","restrict","unless","except",
    "omit","refrain",
    # examples
    "example","e.g","such","like","instance","sample","demonstrate",
    # CoT
    "think","reason","explain","chain","break","analyze","consider","evaluate",
    "reflect","before","identify","determine",
    # action verbs
    "provide","generate","create","summarize","list","extract","produce","write",
    "compare","describe","draft","respond","evaluate","recommend",
    # domain keywords (aggregated)
    "legal","law","compliance","contract","financial","investment","portfolio",
    "medical","clinical","patient","diagnosis","code","software","function","api",
    "marketing","brand","campaign","employee","hiring","talent","research","study",
    "analysis","customer","support","sales","prospect","pipeline","product","feature",
    "roadmap","data","model","dataset","prediction",
    # tone words
    "formal","academic","rigorous","recommend","advise","suggest","friendly","simple",
    "helpful","direct","concise","brief","understand","feel","empathize","precise",
    "specification","implementation",
    # misc
    "output","result","deliverable","recommendation","action","conclusion",
    "decision","finding","insight","context","mission","objective","goal","task",
    "information","response","answer","question","guidance","instruction",
]

VOCAB = {tok: i + 4 for i, tok in enumerate(VOCAB_TOKENS)}
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3
VOCAB_SIZE = len(VOCAB) + 4

CRITERIA_NAMES = [
    "Specificity & clarity",
    "Domain alignment",
    "Tone match",
    "Output structure",
    "Completeness",
    "Conciseness",
    "Actionability",
    "Role definition",
    "Constraint clarity",
    "Audience awareness",
]
N_CRITERIA = len(CRITERIA_NAMES)

# ─────────────────────────────────────────────────────────
# TOKENIZER  (character n-gram bag → token IDs)
# ─────────────────────────────────────────────────────────

def tokenize(text: str, max_len: int = 128) -> List[int]:
    """Convert raw text to a list of integer token IDs."""
    words = re.findall(r"[a-z0-9\-]+", text.lower())
    ids = [BOS_ID]
    for w in words:
        ids.append(VOCAB.get(w, UNK_ID))
    ids.append(EOS_ID)
    if len(ids) > max_len:
        ids = ids[:max_len - 1] + [EOS_ID]
    return ids


def pad_batch(batch: List[List[int]], pad_id: int = PAD_ID) -> torch.Tensor:
    maxlen = max(len(s) for s in batch)
    padded = [s + [pad_id] * (maxlen - len(s)) for s in batch]
    return torch.tensor(padded, dtype=torch.long)


# ─────────────────────────────────────────────────────────
# POSITIONAL ENCODING
# ─────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 256):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ─────────────────────────────────────────────────────────
# DISCRIMINATOR  — Transformer Encoder → multi-criterion scorer
# ─────────────────────────────────────────────────────────

class PromptDiscriminator(nn.Module):
    """
    Input : token sequence  (B, T)
    Output: per-criterion scores  (B, N_CRITERIA)  in [0, 10]
    """
    def __init__(
        self,
        vocab_size:  int = VOCAB_SIZE,
        d_model:     int = 128,
        nhead:       int = 4,
        num_layers:  int = 3,
        dim_ff:      int = 256,
        n_criteria:  int = N_CRITERIA,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_enc = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.norm    = nn.LayerNorm(d_model)
        # per-criterion head
        self.head    = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, n_criteria),
        )

    def forward(self, token_ids: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embed(token_ids)                          # (B, T, D)
        x = self.pos_enc(x)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        # mean-pool over non-padding positions
        if key_padding_mask is not None:
            mask = (~key_padding_mask).float().unsqueeze(-1)
            x = (x * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:
            x = x.mean(1)
        x = self.norm(x)
        scores = self.head(x)                              # (B, N_CRITERIA)
        return torch.sigmoid(scores) * 10.0                # scale to [0, 10]


# ─────────────────────────────────────────────────────────
# GENERATOR  — Transformer Decoder (conditioned on latent + config)
# ─────────────────────────────────────────────────────────

class PromptGenerator(nn.Module):
    """
    Given a latent noise vector + config embedding,
    produce a soft token distribution  (B, T_out, VOCAB_SIZE).
    We decode greedily to get actual token IDs for the discriminator.
    """
    def __init__(
        self,
        vocab_size:   int = VOCAB_SIZE,
        latent_dim:   int = 64,
        config_dim:   int = 32,
        d_model:      int = 128,
        nhead:         int = 4,
        num_layers:   int = 3,
        dim_ff:        int = 256,
        max_out_len:  int = 80,
        dropout:      float = 0.1,
    ):
        super().__init__()
        self.max_out_len = max_out_len
        self.d_model     = d_model
        self.vocab_size  = vocab_size

        self.latent_proj = nn.Linear(latent_dim + config_dim, d_model)
        self.embed       = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_enc     = PositionalEncoding(d_model, dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.norm    = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, vocab_size)

    def _causal_mask(self, sz: int) -> torch.Tensor:
        return torch.triu(torch.ones(sz, sz, dtype=torch.bool), diagonal=1)

    def forward(
        self,
        z:          torch.Tensor,   # (B, latent_dim)
        cfg_embed:  torch.Tensor,   # (B, config_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
          logits     (B, T_out, vocab_size)   — for loss
          token_ids  (B, T_out)               — greedy decode, fed to discriminator
        """
        B = z.size(0)
        cond = torch.cat([z, cfg_embed], dim=-1)           # (B, latent+config)
        memory = self.latent_proj(cond).unsqueeze(1)       # (B, 1, D) — single-step memory

        # Teacher-forcing with BOS; generate auto-regressively
        tgt_ids = torch.full((B, 1), BOS_ID, dtype=torch.long, device=z.device)
        all_logits = []

        for _ in range(self.max_out_len - 1):
            tgt_emb = self.pos_enc(self.embed(tgt_ids))    # (B, t, D)
            tgt_mask = self._causal_mask(tgt_ids.size(1)).to(z.device)
            out = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            out = self.norm(out)
            logits = self.out_proj(out)                    # (B, t, V)
            all_logits.append(logits[:, -1:])              # last step
            next_tok = logits[:, -1].argmax(-1, keepdim=True)
            tgt_ids  = torch.cat([tgt_ids, next_tok], dim=1)
            if (next_tok == EOS_ID).all():
                break

        logits_seq = torch.cat(all_logits, dim=1)          # (B, T_out, V)
        return logits_seq, tgt_ids


# ─────────────────────────────────────────────────────────
# CONFIG EMBEDDING  — encodes domain/format/tone as a vector
# ─────────────────────────────────────────────────────────

DOMAIN_IDX  = {k: i for i, k in enumerate(DOMAINS)}
FORMAT_IDX  = {k: i for i, k in enumerate(FORMATS)}
TONE_IDX    = {k: i for i, k in enumerate(TONES)}

CONFIG_DIM = 32

class ConfigEmbedder(nn.Module):
    def __init__(self, config_dim: int = CONFIG_DIM):
        super().__init__()
        self.dom_emb = nn.Embedding(len(DOMAIN_IDX), config_dim // 3 + 1)
        self.fmt_emb = nn.Embedding(len(FORMAT_IDX), config_dim // 3 + 1)
        self.ton_emb = nn.Embedding(len(TONE_IDX),   config_dim // 3 + 1)
        in_dim = (config_dim // 3 + 1) * 3
        self.proj = nn.Linear(in_dim, config_dim)

    def forward(self, domain: str, fmt: str, tone: str) -> torch.Tensor:
        d = torch.tensor([DOMAIN_IDX.get(domain, 0)], dtype=torch.long)
        f = torch.tensor([FORMAT_IDX.get(fmt, 0)],    dtype=torch.long)
        t = torch.tensor([TONE_IDX.get(tone, 0)],     dtype=torch.long)
        emb = torch.cat([self.dom_emb(d), self.fmt_emb(f), self.ton_emb(t)], dim=-1)
        return self.proj(emb)   # (1, config_dim)


# ─────────────────────────────────────────────────────────
# RULE HEURISTICS  (kept as lightweight ground-truth signal
#                   for discriminator pre-training targets)
# ─────────────────────────────────────────────────────────

CLARITY_MARKERS   = ["specifically","clearly","concisely","accurately","precisely",
                     "step-by-step","comprehensive","thorough","must","should","always",
                     "never","ensure","only","exactly","explicitly","focus","prioritize"]
STRUCTURE_MARKERS = ["bullet","list","numbered","format","section","header","table",
                     "json","markdown","summary","outline","heading","paragraph",
                     "first","then","finally","next"]
ROLE_MARKERS      = ["you are","act as","your role","as a","expert","specialist",
                     "professional","analyst","advisor","assistant","consultant"]
AUDIENCE_MARKERS  = ["audience","user","reader","client","customer","professional",
                     "beginner","expert","team","executive","manager","stakeholder"]
CONSTRAINT_MARKERS= ["do not","don't","avoid","never","without","exclude","limit",
                     "restrict","only","unless","except","refrain","omit"]
COT_MARKERS       = ["think step","reason through","explain your reasoning",
                     "chain of thought","break down","analyze","consider","evaluate",
                     "reflect","before answering","first identify","then determine"]
ACTION_VERBS      = ["provide","generate","create","analyze","summarize","explain",
                     "list","identify","evaluate","respond","write","describe",
                     "compare","extract"]
DOMAIN_KW = {
    "legal":           ["legal","law","attorney","compliance","regulation","contract"],
    "finance":         ["financial","finance","investment","portfolio","revenue","budget"],
    "healthcare":      ["medical","clinical","patient","diagnosis","treatment","health"],
    "software":        ["code","software","function","api","debug","algorithm"],
    "marketing":       ["marketing","brand","campaign","audience","conversion","copy"],
    "hr":              ["employee","hr","hiring","performance","talent","onboarding"],
    "research":        ["research","study","analysis","methodology","hypothesis","data"],
    "customer_support":["customer","support","ticket","issue","resolution","escalate"],
    "sales":           ["sales","prospect","pipeline","deal","close","revenue"],
    "product":         ["product","feature","roadmap","user story","sprint","backlog"],
    "data_science":    ["data","model","dataset","training","feature","prediction"],
    "general":         [],
}
TONE_KW = {
    "professional":  ["professional","accurate","thorough","objective","formal","structured"],
    "formal":        ["formal","academic","scholarly","rigorous","cite","reference"],
    "consultative":  ["recommend","advise","suggest","consider","propose","evaluate"],
    "friendly":      ["friendly","simple","easy","helpful","approachable","clear"],
    "direct":        ["direct","concise","brief","straight","bottom line","key point"],
    "empathetic":    ["understand","feel","support","concern","perspective","empathize"],
    "technical":     ["technical","precise","specification","parameter","implementation"],
}

def _cnt(text, markers):
    lo = text.lower()
    return sum(1 for m in markers if m in lo)

def _wc(text):
    return len(text.split())

def rule_scores(text: str, domain: str, fmt: str, tone: str) -> List[float]:
    """Return 10 heuristic scores ∈ [0,10] used as soft labels."""
    lo = text.lower()
    wc = _wc(text)

    # specificity
    s0 = min(10.0, _cnt(text, CLARITY_MARKERS) * 0.8 +
             (2.0 if 40 <= wc <= 150 else 1.0 if wc <= 250 else 0.5) +
             min(2.0, _cnt(text, ACTION_VERBS) * 0.5))

    # domain
    kw = DOMAIN_KW.get(domain, [])
    s1 = 7.0 if not kw else min(10.0, _cnt(text, kw) * 2.0 + 1.0)

    # tone
    s2 = min(10.0, _cnt(text, TONE_KW.get(tone, [])) * 1.8 +
             (2.0 if TONES.get(tone, "").lower() in lo else 0.0) + 1.5)

    # structure
    has_bullet  = bool(re.search(r'^\s*[-*•]\s', text, re.M))
    has_num     = bool(re.search(r'^\s*\d+[\.\)]\s', text, re.M))
    has_headers = bool(re.search(r'^\s*\[.+\]|^#+\s|\bStep\s+\d+', text, re.M))
    s3 = min(10.0, _cnt(text, STRUCTURE_MARKERS) * 0.6 +
             3.0 * has_bullet + 3.0 * has_num + 3.0 * has_headers)

    # completeness (covers task/audience/domain/format/tone)
    s4 = min(10.0,
             (2.0 if DOMAINS.get(domain,"").lower().split(" / ")[0] in lo else 0.0) +
             (2.0 if FORMATS.get(fmt,"").lower().split()[0] in lo else 0.0) +
             (2.0 if TONES.get(tone,"").lower().split()[0] in lo else 0.0) +
             min(2.0, _cnt(text, COT_MARKERS) * 0.8) +
             min(2.0, _cnt(text, AUDIENCE_MARKERS) * 0.5))

    # conciseness
    if   40 <= wc <= 100:  s5 = 9.0
    elif 100 < wc <= 180:  s5 = 7.5
    elif 180 < wc <= 260:  s5 = 6.0
    elif 260 < wc <= 350:  s5 = 4.5
    else:                  s5 = 3.0
    fillers = ["in order to","please note","feel free to","do not hesitate","certainly","of course"]
    s5 = max(0.0, s5 - _cnt(text, fillers) * 0.8)

    # actionability
    s6 = min(10.0, _cnt(text, ACTION_VERBS) * 0.9 +
             min(2.0, _cnt(text, COT_MARKERS) * 0.7))

    # role
    s7 = min(10.0, _cnt(text, ROLE_MARKERS) * 1.5 +
             (2.5 if "you are" in lo or "act as" in lo else 0.0))

    # constraints
    s8 = min(10.0, _cnt(text, CONSTRAINT_MARKERS) * 1.2 +
             (2.0 if "never" in lo or "do not" in lo else 0.0))

    # audience
    s9 = min(10.0, _cnt(text, AUDIENCE_MARKERS) * 1.0 +
             (2.0 if "audience" in lo else 0.0))

    return [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9]


# ─────────────────────────────────────────────────────────
# MUTATION TEMPLATES  (seed corpus for generator training)
# ─────────────────────────────────────────────────────────

MUTATION_TEMPLATES = [
    lambda t, d, a, tn, f: (
        f"You are an expert in {DOMAINS.get(d, d)}.\n"
        f"Reason step by step before answering.\nTask: {t}\nAudience: {a}\n"
        f"Tone: {TONES.get(tn, tn)}.\nOutput format: {FORMATS.get(f, f)}.\n"
        f"Always explain reasoning before conclusions."
    ),
    lambda t, d, a, tn, f: (
        f"You are a senior {DOMAINS.get(d, d)} specialist.\nYour role is to: {t}\n"
        f"Do not include irrelevant information. Be precise and {TONES.get(tn,tn).lower()}.\n"
        f"Always tailor your response for: {a}.\nFormat: {FORMATS.get(f, f)}."
    ),
    lambda t, d, a, tn, f: (
        f"[Role] You are a {DOMAINS.get(d, d)} expert.\n[Task] {t}\n"
        f"[Audience] {a}\n[Tone] {TONES.get(tn, tn)}\n[Format] {FORMATS.get(f, f)}\n"
        f"[Rule] Lead with the most important insight. Be actionable and specific."
    ),
    lambda t, d, a, tn, f: (
        f"You are assisting {a}.\nThey need you to: {t}\n"
        f"You are an expert in {DOMAINS.get(d, d)}.\n"
        f"Respond in a {TONES.get(tn,tn).lower()} manner.\n"
        f"Structure your output as: {FORMATS.get(f, f)}.\nAvoid jargon unless the audience is technical."
    ),
    lambda t, d, a, tn, f: (
        f"Task: {t}\n\nRequirements:\n- Domain: {DOMAINS.get(d, d)}\n"
        f"- Audience: {a}\n- Tone: {TONES.get(tn, tn)}\n- Format: {FORMATS.get(f, f)}\n\n"
        f"Constraints:\n- Do not speculate.\n- Cite reasoning.\n- Avoid filler phrases.\n"
        f"- State assumptions explicitly."
    ),
    lambda t, d, a, tn, f: (
        f"You are a highly experienced {DOMAINS.get(d, d)} professional.\n"
        f"Objective: {t}\nWhen responding:\n"
        f"1. Identify the core need of {a}.\n"
        f"2. Provide a structured response in {FORMATS.get(f, f)} format.\n"
        f"3. Include specific, actionable details.\n"
        f"4. Summarize with a clear next step or recommendation."
    ),
    lambda t, d, a, tn, f: (
        f"Context: You operate in the {DOMAINS.get(d, d)} domain and assist {a}.\n"
        f"Mission: {t}\nGuiding principles:\n- Tone: {TONES.get(tn, tn)}\n"
        f"- Format: {FORMATS.get(f, f)}\n- Depth: Thorough but not verbose\n"
        f"- Accuracy: Never speculate; flag uncertainty\n"
        f"- Relevance: Every sentence must serve the mission"
    ),
    lambda t, d, a, tn, f: (
        f"You help {a} with {t}.\n"
        f"Be {TONES.get(tn,tn).lower()}, specific, and accurate.\n"
        f"Respond in {FORMATS.get(f, f)} format.\n"
        f"Use {DOMAINS.get(d, d)} domain expertise in every answer."
    ),
    lambda t, d, a, tn, f: (
        f"You are a {DOMAINS.get(d, d)} expert assistant.\n\n"
        f"Your task: {t}\nYour audience: {a}\n\n"
        f"Before responding, verify:\n- Is this accurate and relevant?\n"
        f"- Is the tone {TONES.get(tn, tn).lower()}?\n"
        f"- Is it formatted as {FORMATS.get(f, f)}?\n"
        f"- Would {a} find this genuinely useful?"
    ),
    lambda t, d, a, tn, f: (
        f"You are an AI assistant specializing in {DOMAINS.get(d, d)}.\n\n"
        f"Step 1 — Understand: {t}\n"
        f"Step 2 — Consider audience: {a}\n"
        f"Step 3 — Apply {DOMAINS.get(d, d)} knowledge\n"
        f"Step 4 — Structure as {FORMATS.get(f, f)}\n"
        f"Step 5 — Review for {TONES.get(tn, tn).lower()} tone\n\n"
        f"Always prioritize clarity, accuracy, and usefulness."
    ),
]

MICRO_MUTATIONS = [
    lambda p: p + "\nThink step by step before responding.",
    lambda p: p + "\nKeep your response concise — no unnecessary padding.",
    lambda p: p.rstrip() + "\nAlways respond in the format specified above.",
    lambda p: ("You are a highly experienced professional.\n" + p) if "you are" not in p.lower() else p,
    lambda p: p + "\nBefore finalizing, verify accuracy, relevance, and structure.",
    lambda p: p + "\nDo not speculate. State assumptions explicitly when uncertain.",
    lambda p: p + "\nWhere helpful, illustrate with a brief concrete example.",
]


# ─────────────────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────────────────

@dataclass
class EvalCriterion:
    name: str
    weight: float
    key: str

@dataclass
class EvalResult:
    scores: dict
    overall: float
    strengths: list
    weaknesses: list
    suggestions: list

@dataclass
class PromptCandidate:
    content: str
    generation: int
    eval_result: Optional[EvalResult] = None

    @property
    def score(self) -> float:
        return self.eval_result.overall if self.eval_result else 0.0

@dataclass
class GenerationRecord:
    generation: int
    candidates: list
    best: PromptCandidate
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))
    g_loss: float = 0.0
    d_loss: float = 0.0

@dataclass
class RunConfig:
    task: str
    audience: str
    domain: str
    output_format: str
    tone: str
    generations: int
    criteria: list
    seed_prompt: str = ""

DEFAULT_CRITERIA = [
    EvalCriterion("Specificity & clarity",   5.0, "specificity"),
    EvalCriterion("Domain alignment",        4.0, "domain"),
    EvalCriterion("Tone match",              4.0, "tone"),
    EvalCriterion("Output structure",        3.0, "structure"),
    EvalCriterion("Completeness",            4.0, "completeness"),
    EvalCriterion("Conciseness",             3.0, "conciseness"),
    EvalCriterion("Actionability",           4.0, "actionability"),
    EvalCriterion("Role definition",         3.0, "role"),
    EvalCriterion("Constraint clarity",      3.0, "constraints"),
    EvalCriterion("Audience awareness",      4.0, "audience"),
]

IMPROVEMENT_TIPS = {
    "specificity":   "Add imperative verbs (provide, analyze, generate) and reference the task more directly.",
    "domain":        "Include domain-specific terminology and reference the professional context explicitly.",
    "tone":          "Add tone-aligned vocabulary.",
    "structure":     "Specify an output format explicitly (e.g. 'Respond in bullet points').",
    "completeness":  "Cover all 5 dimensions: task, audience, domain, format, and tone.",
    "conciseness":   "Remove filler phrases and aim for 50–150 words.",
    "actionability": "Add outcome-focused instructions ('Provide a recommendation').",
    "role":          "Define the AI's role explicitly: 'You are a [domain] specialist'.",
    "constraints":   "Add guard rails: what to avoid, what NOT to include.",
    "audience":      "Reference the target audience explicitly in the prompt text.",
}


# ─────────────────────────────────────────────────────────
# GAN TRAINER
# ─────────────────────────────────────────────────────────

LATENT_DIM = 64

class GANTrainer:
    """
    Wraps Generator + Discriminator.
    - Pre-trains Discriminator on rule-score targets.
    - Trains Generator adversarially to produce high-scoring prompts.
    """

    def __init__(self, config: RunConfig, device: str = "cpu"):
        self.config   = config
        self.device   = device
        self.D = PromptDiscriminator().to(device)
        self.G = PromptGenerator(latent_dim=LATENT_DIM, config_dim=CONFIG_DIM).to(device)
        self.cfg_emb  = ConfigEmbedder().to(device)
        self.opt_D    = Adam(list(self.D.parameters()) + list(self.cfg_emb.parameters()), lr=2e-4, betas=(0.5, 0.999))
        self.opt_G    = Adam(self.G.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.d_loss_history: List[float] = []
        self.g_loss_history: List[float] = []

    def _cfg_tensor(self) -> torch.Tensor:
        return self.cfg_emb(self.config.domain, self.config.output_format, self.config.tone)

    def _real_batch(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tokenise real texts → token_ids, padding mask, rule score targets."""
        ids = [tokenize(t) for t in texts]
        token_tensor = pad_batch(ids).to(self.device)
        pad_mask = (token_tensor == PAD_ID)
        targets = torch.tensor(
            [rule_scores(t, self.config.domain, self.config.output_format, self.config.tone)
             for t in texts],
            dtype=torch.float32,
        ).to(self.device)
        return token_tensor, pad_mask, targets

    def _z(self, B: int) -> torch.Tensor:
        return torch.randn(B, LATENT_DIM, device=self.device)

    def pretrain_D(self, real_texts: List[str], steps: int = 8):
        """Pretrain discriminator on rule-heuristic targets."""
        self.D.train()
        mse = nn.MSELoss()
        for _ in range(steps):
            tok, mask, targets = self._real_batch(real_texts)
            self.opt_D.zero_grad()
            preds = self.D(tok, mask)
            loss  = mse(preds, targets)
            loss.backward()
            self.opt_D.step()

    def train_step(self, real_texts: List[str]) -> Tuple[float, float]:
        """One adversarial step. Returns (D_loss, G_loss)."""
        B = len(real_texts)
        cfg = self._cfg_tensor().expand(B, -1)

        # ── Train Discriminator ────────────────────────────
        self.D.train(); self.G.eval()
        self.opt_D.zero_grad()

        tok, mask, targets = self._real_batch(real_texts)
        real_preds = self.D(tok, mask)
        d_real_loss = F.mse_loss(real_preds, targets)

        with torch.no_grad():
            _, fake_ids = self.G(self._z(B), cfg)
        fake_mask  = (fake_ids == PAD_ID)
        fake_preds = self.D(fake_ids.detach(), fake_mask)
        fake_targets = torch.zeros_like(fake_preds)          # fake = score 0
        d_fake_loss  = F.mse_loss(fake_preds, fake_targets)

        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        nn.utils.clip_grad_norm_(self.D.parameters(), 1.0)
        self.opt_D.step()

        # ── Train Generator ────────────────────────────────
        self.G.train(); self.D.eval()
        self.opt_G.zero_grad()

        logits, fake_ids2 = self.G(self._z(B), cfg)
        fake_mask2  = (fake_ids2 == PAD_ID)
        g_preds     = self.D(fake_ids2, fake_mask2)
        # Generator wants discriminator to output high scores
        g_target    = torch.full_like(g_preds, 8.0)
        g_loss = F.mse_loss(g_preds, g_target)

        # diversity regularisation — penalise token collapse
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * (probs + 1e-8).log()).sum(-1).mean()
        g_loss = g_loss - 0.05 * entropy

        g_loss.backward()
        nn.utils.clip_grad_norm_(self.G.parameters(), 1.0)
        self.opt_G.step()

        return d_loss.item(), g_loss.item()

    @torch.no_grad()
    def generate_prompts(self, n: int, task: str, audience: str) -> List[str]:
        """Generate n prompt texts by decoding token IDs → words."""
        self.G.eval()
        cfg = self._cfg_tensor().expand(n, -1)
        _, token_ids = self.G(self._z(n), cfg)
        rev_vocab = {v: k for k, v in VOCAB.items()}
        prompts = []
        for row in token_ids:
            words = []
            for tid in row.tolist():
                if tid == EOS_ID:
                    break
                if tid in (PAD_ID, BOS_ID):
                    continue
                words.append(rev_vocab.get(tid, "<unk>"))
            # stitch into a readable string; wrap with task/audience for usability
            gen_phrase = " ".join(words) if words else "provide a clear and accurate response"
            prompt = (
                f"You are an expert in {DOMAINS.get(self.config.domain, 'general')}.\n"
                f"Task: {task}\nAudience: {audience}\n"
                f"Instructions (generated): {gen_phrase}.\n"
                f"Tone: {TONES.get(self.config.tone, 'professional')}. "
                f"Format: {FORMATS.get(self.config.output_format, 'flexible')}.\n"
                f"Do not speculate. Be concise and actionable."
            )
            prompts.append(prompt)
        return prompts


# ─────────────────────────────────────────────────────────
# NEURAL SCORER (wraps D for evaluation)
# ─────────────────────────────────────────────────────────

class NeuralScorer:
    def __init__(self, D: PromptDiscriminator, criteria: List[EvalCriterion]):
        self.D = D
        self.criteria = criteria

    @torch.no_grad()
    def evaluate(self, candidate: PromptCandidate, config: RunConfig) -> PromptCandidate:
        self.D.eval()
        ids   = tokenize(candidate.content)
        tok   = pad_batch([ids])
        mask  = (tok == PAD_ID)
        raw   = self.D(tok, mask).squeeze(0).tolist()  # 10 scores

        scores = {}
        weighted_sum, total_w = 0.0, 0.0
        for i, c in enumerate(self.criteria):
            s = raw[i] if i < len(raw) else 5.0
            scores[c.name] = round(s, 2)
            weighted_sum += s * c.weight
            total_w      += c.weight

        overall = weighted_sum / total_w if total_w else 0.0
        sorted_s = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        strengths  = [k for k, v in sorted_s[:3] if v >= 6.5][:2] or ["Baseline structure present"]
        weaknesses = [k for k, v in sorted_s[-3:] if v < 5.5][:2] or ["Minor refinements possible"]
        suggestions = []
        for cname, cscore in sorted_s:
            if cscore < 5.0:
                key = next((c.key for c in self.criteria if c.name == cname), "")
                suggestions.append(IMPROVEMENT_TIPS.get(key, f"Improve {cname.lower()}."))
            if len(suggestions) >= 3:
                break

        candidate.eval_result = EvalResult(
            scores=scores, overall=round(overall, 3),
            strengths=strengths, weaknesses=weaknesses, suggestions=suggestions,
        )
        return candidate


# ─────────────────────────────────────────────────────────
# OPTIMIZATION LOOP
# ─────────────────────────────────────────────────────────

POPULATION_SIZE = 5
ELITE_SIZE      = 2
D_PRETRAIN_STEPS = 10
GAN_STEPS_PER_GEN = 3

class GANOptimizationLoop:

    def __init__(self, config: RunConfig):
        self.config  = config
        self.trainer = GANTrainer(config)
        self.scorer  = NeuralScorer(self.trainer.D, config.criteria)

    def _template_prompts(self, n: int) -> List[str]:
        fns = random.sample(MUTATION_TEMPLATES, min(n, len(MUTATION_TEMPLATES)))
        t, d, a, tn, f = (self.config.task, self.config.domain, self.config.audience,
                          self.config.tone, self.config.output_format)
        return [fn(t, d, a, tn, f) for fn in fns]

    def _micro_mutate(self, prompt: str) -> str:
        for m in random.sample(MICRO_MUTATIONS, 2):
            prompt = m(prompt)
        return prompt

    def run(self, progress_callback=None) -> List[GenerationRecord]:
        config = self.config
        total_steps = config.generations * POPULATION_SIZE
        step = 0

        # Seed population from templates (+ optional user seed)
        seed_texts = self._template_prompts(POPULATION_SIZE)
        if config.seed_prompt.strip():
            seed_texts[0] = config.seed_prompt.strip()

        # Pre-train discriminator on heuristic labels
        self.trainer.pretrain_D(seed_texts, steps=D_PRETRAIN_STEPS)

        population = [PromptCandidate(p, 0) for p in seed_texts]
        records: List[GenerationRecord] = []

        for gen in range(1, config.generations + 1):
            real_texts = [c.content for c in population]

            # GAN adversarial training for this generation
            d_loss_avg, g_loss_avg = 0.0, 0.0
            for _ in range(GAN_STEPS_PER_GEN):
                dl, gl = self.trainer.train_step(real_texts)
                d_loss_avg += dl; g_loss_avg += gl
            d_loss_avg /= GAN_STEPS_PER_GEN
            g_loss_avg /= GAN_STEPS_PER_GEN

            # Score population with neural discriminator
            evaluated = []
            for c in population:
                c.generation = gen
                self.scorer.evaluate(c, config)
                evaluated.append(c)
                step += 1
                if progress_callback:
                    progress_callback(step / total_steps, gen, len(evaluated))

            evaluated.sort(key=lambda c: c.score, reverse=True)
            records.append(GenerationRecord(
                generation=gen, candidates=evaluated, best=evaluated[0],
                g_loss=round(g_loss_avg, 4), d_loss=round(d_loss_avg, 4),
            ))

            if gen < config.generations:
                # Elite + generated + micro-mutated
                elite = evaluated[:ELITE_SIZE]
                n_gen = POPULATION_SIZE - ELITE_SIZE - 1
                gen_texts = self.trainer.generate_prompts(max(1, n_gen), config.task, config.audience)
                mutated   = self._micro_mutate(elite[0].content)

                population = (
                    [PromptCandidate(e.content, gen) for e in elite] +
                    [PromptCandidate(p, gen) for p in gen_texts] +
                    [PromptCandidate(mutated, gen)]
                )[:POPULATION_SIZE]

        return records


# ─────────────────────────────────────────────────────────
# EXPORT HELPERS
# ─────────────────────────────────────────────────────────

def records_to_dataframe(records):
    rows = []
    for r in records:
        for c in r.candidates:
            row = {
                "Generation": r.generation,
                "Score":      round(c.score, 3),
                "G_Loss":     r.g_loss,
                "D_Loss":     r.d_loss,
                "Strengths":  "; ".join(c.eval_result.strengths)   if c.eval_result else "",
                "Weaknesses": "; ".join(c.eval_result.weaknesses)  if c.eval_result else "",
                "Prompt":     c.content,
            }
            if c.eval_result:
                for crit, sc in c.eval_result.scores.items():
                    row[f"[Score] {crit}"] = round(sc, 2)
            rows.append(row)
    return pd.DataFrame(rows)

def records_to_json(records, config: RunConfig) -> str:
    out = {
        "run_timestamp": datetime.now().isoformat(),
        "model": "Transformer-GAN",
        "config": {
            "task": config.task, "audience": config.audience,
            "domain": config.domain, "output_format": config.output_format,
            "tone": config.tone, "generations": config.generations,
        },
        "generations": [],
    }
    for r in records:
        gd = {"generation": r.generation, "timestamp": r.timestamp,
              "g_loss": r.g_loss, "d_loss": r.d_loss, "candidates": []}
        for c in r.candidates:
            gd["candidates"].append({
                "prompt":           c.content,
                "score":            round(c.score, 3),
                "criterion_scores": c.eval_result.scores if c.eval_result else {},
            })
        out["generations"].append(gd)
    return json.dumps(out, indent=2)


# ─────────────────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────────────────

def plot_score_history(records):
    gens        = [r.generation for r in records]
    best_scores = [r.best.score for r in records]
    avg_scores  = [sum(c.score for c in r.candidates) / len(r.candidates) for r in records]
    worst_scores= [min(c.score for c in r.candidates) for r in records]

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.fill_between(gens, worst_scores, best_scores, alpha=0.12, color="#7F77DD")
    ax.plot(gens, best_scores,  "o-",  color="#534AB7", lw=2,   ms=7, label="Best")
    ax.plot(gens, avg_scores,   "s--", color="#1D9E75", lw=1.5, ms=5, label="Average")
    ax.plot(gens, worst_scores, "^:",  color="#D85A30", lw=1.2, ms=5, label="Worst")
    ax.set_xlabel("Generation"); ax.set_ylabel("Score (0–10)")
    ax.set_title("Score evolution across generations", fontweight="normal")
    ax.set_ylim(0, 10.5); ax.set_xticks(gens)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(); ax.grid(True, alpha=0.2, linestyle="--")
    fig.tight_layout(); return fig

def plot_gan_losses(records):
    gens   = [r.generation for r in records]
    g_loss = [r.g_loss for r in records]
    d_loss = [r.d_loss for r in records]
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(gens, g_loss, "o-", color="#534AB7", lw=2, ms=6, label="Generator loss")
    ax.plot(gens, d_loss, "s--",color="#D85A30", lw=2, ms=6, label="Discriminator loss")
    ax.set_xlabel("Generation"); ax.set_ylabel("MSE Loss")
    ax.set_title("GAN training losses per generation", fontweight="normal")
    ax.set_xticks(gens)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(); ax.grid(True, alpha=0.2, linestyle="--")
    fig.tight_layout(); return fig

def plot_criteria_bars(eval_result: EvalResult):
    if not eval_result or not eval_result.scores:
        return None
    labels = list(eval_result.scores.keys())
    values = [eval_result.scores[l] for l in labels]
    colors = ["#534AB7" if v >= 7 else "#1D9E75" if v >= 5 else "#D85A30" for v in values]
    fig, ax = plt.subplots(figsize=(8, max(3, len(labels) * 0.5)))
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1], height=0.6)
    ax.set_xlim(0, 10); ax.set_xlabel("Score (0–10)")
    ax.set_title("Criterion scores — best prompt", fontweight="normal")
    ax.axvline(x=7, color="#AAAAAA", linestyle="--", lw=0.8, alpha=0.7)
    for bar, val in zip(bars, values[::-1]):
        ax.text(val + 0.15, bar.get_y() + bar.get_height() / 2, f"{val:.1f}", va="center", fontsize=9)
    fig.tight_layout(); return fig


# ─────────────────────────────────────────────────────────
# STREAMLIT APP
# ─────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Prompt Optimizer — Transformer GAN",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "criteria" not in st.session_state: st.session_state.criteria = copy.deepcopy(DEFAULT_CRITERIA)
if "records"  not in st.session_state: st.session_state.records  = []
if "config"   not in st.session_state: st.session_state.config   = None
if "running"  not in st.session_state: st.session_state.running  = False

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.title("🤖 Prompt Optimizer")
    st.caption("Transformer + GAN · Fully offline · PyTorch")
    st.divider()

    st.subheader("Task configuration")
    task = st.text_area(
        "Task / goal",
        placeholder="e.g. Extract and summarize action items from meeting transcripts",
        height=90,
    )
    audience = st.text_area(
        "Target audience",
        placeholder="e.g. C-suite executives, non-technical, prefer concise bullets",
        height=70,
    )
    seed = st.text_area(
        "Seed prompt (optional)",
        placeholder="Paste an existing prompt to evolve from, or leave blank",
        height=70,
    )
    st.divider()

    st.subheader("Optimization settings")
    domain = st.selectbox("Domain / industry", list(DOMAINS.keys()), format_func=lambda k: DOMAINS[k])
    fmt    = st.selectbox("Output format",     list(FORMATS.keys()), format_func=lambda k: FORMATS[k])
    tone   = st.selectbox("Tone",              list(TONES.keys()),   format_func=lambda k: TONES[k])
    gens   = st.slider("Generations", min_value=3, max_value=15, value=6)
    st.divider()

    st.subheader("Evaluation criteria")
    st.caption("Adjust weight 0–5. Discriminator uses weighted average as final score.")
    updated_criteria = []
    for c in st.session_state.criteria:
        w = st.slider(c.name, 0.0, 5.0, float(c.weight), 0.5, key=f"w_{c.key}")
        updated_criteria.append(EvalCriterion(c.name, w, c.key))
    st.session_state.criteria = updated_criteria

    st.divider()
    run_btn   = st.button("🚀 Run optimizer", type="primary", use_container_width=True, disabled=st.session_state.running)
    reset_btn = st.button("↺ Reset", use_container_width=True)

    if reset_btn:
        st.session_state.records  = []
        st.session_state.config   = None
        st.session_state.criteria = copy.deepcopy(DEFAULT_CRITERIA)
        st.rerun()

# ── Tabs ──────────────────────────────────────────────────
tab_results, tab_gan, tab_candidates, tab_log, tab_about = st.tabs([
    "📈 Results", "⚡ GAN Training", "🔬 All candidates", "📋 Run log", "ℹ️ Architecture"
])

# ── Run ───────────────────────────────────────────────────
if run_btn:
    if not task.strip():
        st.error("Please enter a task description.")
    elif not audience.strip():
        st.error("Please enter a target audience.")
    else:
        active_criteria = [c for c in st.session_state.criteria if c.weight > 0]
        config = RunConfig(
            task=task.strip(), audience=audience.strip(),
            domain=domain, output_format=fmt, tone=tone,
            generations=gens, criteria=active_criteria, seed_prompt=seed.strip(),
        )
        st.session_state.config  = config
        st.session_state.running = True

        with tab_results:
            status   = st.empty()
            prog_bar = st.progress(0)

            def cb(pct, gen, ev):
                prog_bar.progress(min(1.0, pct))
                status.info(f"Generation {gen}/{gens} — scored {ev}/{POPULATION_SIZE} candidates  |  GAN training active")

            t0      = time.time()
            loop    = GANOptimizationLoop(config)
            records = loop.run(progress_callback=cb)
            elapsed = round(time.time() - t0, 1)

            st.session_state.records = records
            st.session_state.running = False
            prog_bar.progress(1.0)
            status.success(f"✅ Done — {gens} generations in {elapsed}s")

# ── Results tab ───────────────────────────────────────────
with tab_results:
    records = st.session_state.records
    config  = st.session_state.config

    if not records:
        st.info("Configure your task in the sidebar and click **Run optimizer** to begin.")
    else:
        best_overall = max((r.best for r in records), key=lambda c: c.score)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best score",        f"{best_overall.score:.2f} / 10")
        c2.metric("Generations",       len(records))
        c3.metric("Total candidates",  sum(len(r.candidates) for r in records))
        delta = records[-1].best.score - records[0].best.score
        c4.metric("Score improvement", f"{delta:+.2f}", delta=f"{delta:+.2f}")

        st.divider()
        st.subheader("Score evolution")
        st.pyplot(plot_score_history(records))
        st.divider()

        st.subheader("🏆 Best prompt")
        st.code(best_overall.content, language="text")

        left, right = st.columns(2)
        with left:
            if best_overall.eval_result:
                er = best_overall.eval_result
                st.markdown("**Strengths**")
                for s in er.strengths:    st.success(s)
                st.markdown("**Weaknesses**")
                for w in er.weaknesses:   st.warning(w)
                st.markdown("**Suggestions**")
                for sg in er.suggestions: st.info(sg)
        with right:
            fig = plot_criteria_bars(best_overall.eval_result)
            if fig: st.pyplot(fig)

        st.divider()
        st.subheader("Export")
        e1, e2 = st.columns(2)
        with e1:
            csv = records_to_dataframe(records).to_csv(index=False).encode("utf-8")
            st.download_button("📥 CSV",  csv, "results.csv", "text/csv", use_container_width=True)
        with e2:
            js = records_to_json(records, config).encode("utf-8")
            st.download_button("📥 JSON", js, "results.json", "application/json", use_container_width=True)
        st.text_area("Copy best prompt", value=best_overall.content, height=160)

# ── GAN tab ───────────────────────────────────────────────
with tab_gan:
    records = st.session_state.records
    if not records:
        st.info("Run the optimizer first to see GAN training metrics.")
    else:
        st.subheader("GAN training losses")
        st.caption("Generator loss ↓ = generator improves at fooling the discriminator. Discriminator loss ↓ = discriminator better at distinguishing real vs generated prompts.")
        st.pyplot(plot_gan_losses(records))

        st.divider()
        st.subheader("Per-generation summary")
        gan_rows = [
            {"Generation": r.generation, "Best score": round(r.best.score, 3),
             "G loss": r.g_loss, "D loss": r.d_loss,
             "Avg score": round(sum(c.score for c in r.candidates) / len(r.candidates), 3)}
            for r in records
        ]
        st.dataframe(pd.DataFrame(gan_rows), use_container_width=True, hide_index=True)

# ── All candidates tab ────────────────────────────────────
with tab_candidates:
    records = st.session_state.records
    if not records:
        st.info("Run the optimizer first.")
    else:
        gen_options = [f"Generation {r.generation}" for r in records]
        sel = st.selectbox("Select generation", gen_options, index=len(gen_options) - 1)
        record = next(r for r in records if r.generation == int(sel.split()[-1]))

        for i, c in enumerate(record.candidates):
            with st.expander(f"#{i+1}  Score: {c.score:.2f}/10  {'🏆 Best' if i == 0 else ''}", expanded=(i == 0)):
                st.code(c.content, language="text")
                if c.eval_result:
                    er = c.eval_result
                    cols = st.columns(3)
                    with cols[0]:
                        st.markdown("**Strengths**")
                        for s in er.strengths: st.success(s)
                    with cols[1]:
                        st.markdown("**Weaknesses**")
                        for w in er.weaknesses: st.warning(w)
                    with cols[2]:
                        st.markdown("**Criterion scores**")
                        for crit, sc in er.scores.items():
                            color = "🟢" if sc >= 7 else "🟡" if sc >= 5 else "🔴"
                            st.write(f"{color} `{crit}` — **{sc:.1f}**")

# ── Run log ───────────────────────────────────────────────
with tab_log:
    records = st.session_state.records
    config  = st.session_state.config
    if not records:
        st.info("Run the optimizer first.")
    else:
        if config:
            st.markdown(f"**Task:** {config.task}")
            st.markdown(f"**Domain:** {DOMAINS.get(config.domain)} · **Tone:** {TONES.get(config.tone)} · **Format:** {FORMATS.get(config.output_format)}")
        log_rows = []
        for r in records:
            for c in r.candidates:
                log_rows.append({
                    "Gen":        r.generation,
                    "Score":      round(c.score, 3),
                    "G Loss":     r.g_loss,
                    "D Loss":     r.d_loss,
                    "Strengths":  "; ".join(c.eval_result.strengths)  if c.eval_result else "",
                    "Weaknesses": "; ".join(c.eval_result.weaknesses) if c.eval_result else "",
                    "Prompt":     c.content[:100] + ("…" if len(c.content) > 100 else ""),
                })
        st.dataframe(pd.DataFrame(log_rows), use_container_width=True, hide_index=True)

# ── Architecture tab ──────────────────────────────────────
with tab_about:
    st.markdown(f"""
## Transformer + GAN Prompt Optimizer

Replaces the original rule-based scorer and template mutator with a full neural architecture.

### Setup

```bash
pip install streamlit pandas matplotlib torch
streamlit run app.py
```

### Architecture overview

```
┌────────────────────────────────────────────────────────────┐
│                     DISCRIMINATOR (D)                      │
│  TokenIDs → Embedding → PositionalEncoding                 │
│           → TransformerEncoder (3 layers, 4 heads)         │
│           → Mean Pool → LayerNorm                          │
│           → Linear head → 10 criterion scores ∈ [0,10]    │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│                      GENERATOR (G)                         │
│  z (Gaussian noise, dim=64)                                │
│  + ConfigEmbedding(domain, format, tone, dim=32)           │
│  → Linear projection → d_model=128 (memory)               │
│  → TransformerDecoder (3 layers, 4 heads, autoregressive)  │
│  → Linear → Vocab logits → Greedy decode → Token IDs      │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│                   ADVERSARIAL TRAINING                     │
│  D pretrain:  real prompts → MSE vs rule-heuristic targets │
│  Per generation:                                           │
│    D step: real (high score) vs fake (zero score) → MSE   │
│    G step: generate → feed to D → want score=8.0 → MSE    │
│            + entropy regularisation (diversity)            │
└────────────────────────────────────────────────────────────┘
```

### Key design decisions

| Component | Choice | Reason |
|---|---|---|
| D architecture | Transformer Encoder | Captures long-range prompt structure |
| G architecture | Transformer Decoder | Autoregressive token generation |
| GAN objective | MSE (LSGAN) | More stable than binary cross-entropy |
| Vocabulary | 200-token domain vocab | Covers all prompt-relevant terms |
| Config conditioning | Learned embedding (domain+format+tone) | G adapts to each run config |
| Diversity loss | Token entropy regularisation | Prevents token collapse in G |
| Bootstrapping | Rule heuristics as D pre-training targets | Cold-start the GAN stably |

### Discriminator dimensions
`vocab={VOCAB_SIZE}` · `d_model=128` · `heads=4` · `layers=3` · `ff_dim=256` · `criteria={N_CRITERIA}`

### Generator dimensions
`latent_dim=64` · `config_dim=32` · `d_model=128` · `heads=4` · `layers=3` · `max_out_len=80`
    """)