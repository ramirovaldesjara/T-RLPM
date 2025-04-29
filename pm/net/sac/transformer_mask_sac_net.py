import torch
import torch.nn as nn
from functools import partial
from torch.nn import functional as F
from pm.registry import NET


@NET.register_module(force=True)
class TransformerActorMaskSAC(nn.Module):
    def __init__(self,
                *args,
                embed_dim: int = 128,
                depth: int = 2,
                num_heads: int = 4,
                norm_layer: nn.LayerNorm = partial(nn.LayerNorm, eps=1e-6),
                cls_embed: bool = True,
                **kwargs):
        super(TransformerActorMaskSAC, self).__init__()
        self.embed_dim = embed_dim
        self.norm_layer = norm_layer
        self.cls_embed = cls_embed


        if cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.action_proj = nn.Linear(1, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.decoder_pred = nn.Linear(embed_dim, 2)  # for mean and log_std

        self.initialize_weights()

    def initialize_weights(self):
        if self.cls_embed:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 1e-6)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x):
        B, L, C = x.shape
        if self.cls_embed:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed.expand(B, x.size(1), -1)
        x = self.encoder(x)
        return x

    def forward_decoder(self, x):
        return self.decoder_pred(x)

    def forward(self, x):
        latent = self.forward_encoder(x)
        if self.cls_embed:
            cash_logits = latent[:, 0:1]  # CLS used as cash
            x = latent[:, 1:]
        else:
            x = latent
            cash_logits = torch.zeros_like(x[:, :1])
        logits = self.forward_decoder(torch.cat([cash_logits, x], dim=1))
        scores = logits[:, :, 0]
        indices = torch.argsort(scores, dim=-1)
        soft_scores = scores * torch.log(indices + 1)
        weights = F.softmax(soft_scores, dim=-1)
        return weights.squeeze(-1)

    def get_action(self, x):
        latent = self.forward_encoder(x)
        pred = self.forward_decoder(latent)
        mu, log_std = pred.chunk(2, dim=-1)
        std = log_std.clamp(-16, 2).exp()
        dist = torch.distributions.Normal(mu, std)
        sampled = dist.rsample().squeeze(-1)
        indices = torch.argsort(sampled, dim=-1)
        soft_logits = sampled * torch.log(indices + 1)
        weights = F.softmax(soft_logits, dim=-1)
        return weights

    def get_action_logprob(self, x):
        latent = self.forward_encoder(x)
        pred = self.forward_decoder(latent)
        mu, log_std = pred.chunk(2, dim=-1)
        std = log_std.clamp(-16, 2).exp()
        dist = torch.distributions.Normal(mu, std)
        sample = dist.rsample().squeeze(-1)
        #indices todos son 0s
        indices = torch.argsort(sample, dim=-1)
        soft_logits = sample * torch.log(indices + 1)
        weights = F.softmax(soft_logits, dim=-1)
        # reshape for log_prob computation
        logprob = dist.log_prob(sample.unsqueeze(-1)).squeeze(-1)  # [B, N]
        logprob = logprob - (-weights.pow(2) + 1.000001).log()
        return weights, logprob.sum(dim=1)


@NET.register_module(force=True)
class TranformerCriticMaskSAC(nn.Module):
    def __init__(self,
                *args,
                embed_dim: int = 128,
                depth: int = 2,
                num_heads: int = 4,
                norm_layer: nn.LayerNorm = partial(nn.LayerNorm, eps=1e-6),
                cls_embed: bool = True,
                **kwargs):
        super(TranformerCriticMaskSAC, self).__init__()
        self.embed_dim = embed_dim
        self.norm_layer = norm_layer
        self.cls_embed = cls_embed
        self.action_proj = nn.Linear(1, embed_dim)


        if cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.decoder_pred = nn.Linear(embed_dim, 2)

        self.initialize_weights()

    def initialize_weights(self):
        if self.cls_embed:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 1e-6)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x, action):
        if len(action.shape) == 2:
            action = action.unsqueeze(-1)
        action_embed = self.action_proj(action)

        B, L, C = x.shape

        if self.cls_embed:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        x = x + action_embed

        x = x + self.pos_embed.expand(B, x.size(1), -1)
        x = self.encoder(x)
        return x

    def forward_decoder(self, x):
        return self.decoder_pred(x)

    def forward(self, x, action):
        latent = self.forward_encoder(x, action)
        latent = latent.mean(dim=1)
        return self.forward_decoder(latent).mean(dim=1)

    def get_q_min(self, x, action):
        latent = self.forward_encoder(x, action)
        latent = latent.mean(dim=1)
        return self.forward_decoder(latent).min(dim=1)[0]

    def get_q1_q2(self, x, action):
        latent = self.forward_encoder(x, action)
        latent = latent.mean(dim=1)
        q1, q2 = self.forward_decoder(latent).chunk(2, dim=1)
        return q1.squeeze(-1), q2.squeeze(-1)
