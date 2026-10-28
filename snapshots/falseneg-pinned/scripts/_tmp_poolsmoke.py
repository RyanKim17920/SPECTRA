import torch
from waivphaet.models.encoder import build_encoder
torch.manual_seed(0)
m = build_encoder(backbone="owkin/phikon-v2", pooling="clsmean", lora_rank=32, lora_alpha=64).eval()
m2 = build_encoder(backbone="owkin/phikon-v2", pooling="clsmean", lora_rank=32,
                   lora_alpha=64, pool_head="gem", infer_pool_head=True).eval()
m2.load_state_dict(m.state_dict(), strict=False)
m2.pool_head.load_state_dict(torch.load("runs/poolcmp-gem-381014/step_0001500/pool_head.pt", map_location="cpu"))
x = torch.randint(0,255,(4,224,224,3),dtype=torch.uint8)
with torch.inference_mode():
    a, b = m.embed(x), m2.embed(x)
res = {
 "shape_default": tuple(a.shape), "shape_gem": tuple(b.shape),
 "cls_half_identical": bool(torch.allclose(a[:,:1024], b[:,:1024])),
 "mean_half_norm_default": round(a[:,1024:].norm(dim=1).mean().item(),4),
 "mean_half_norm_gem": round(b[:,1024:].norm(dim=1).mean().item(),4),
 "cls_half_norm": round(a[:,:1024].norm(dim=1).mean().item(),4),
 "mean_half_min_default": round(a[:,1024:].min().item(),4),
 "mean_half_min_gem": round(b[:,1024:].min().item(),4),
 "gem_p": round(float(m2.pool_head.p.detach()),6),
}
for k,v in res.items(): print(f"RESULT {k} = {v}")
