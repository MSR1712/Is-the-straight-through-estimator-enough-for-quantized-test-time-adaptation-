"""qtta_lib.py: ResNet-20, fake quantization, and Gaussian-smoothed surrogate gradients (QTTA)."""
import copy, glob, math, os, types
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision, torchvision.transforms as T

MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)

# global backward config; a QLayer can override sigma via layer.sigma
QCFG = types.SimpleNamespace(mode="ste", sigma=0.5)   # mode: "ste" | "gauss"; sigma = sigma/Delta
# clip-ratio search grids (start, stop, n) for calibration; set to (1,1,1) for plain min-max
CALIB = types.SimpleNamespace(act=(0.3, 1.0, 36), w=(0.4, 1.0, 31))


# ---------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False); self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False);     self.bn2 = nn.BatchNorm2d(cout)
        self.pad = (cout - cin) if (stride != 1 or cin != cout) else 0
        self.stride = stride
    def shortcut(self, x):
        if self.pad == 0: return x
        x = x[:, :, ::self.stride, ::self.stride]
        return F.pad(x, (0, 0, 0, 0, self.pad // 2, self.pad // 2))
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))

class ResNetCifar(nn.Module):
    def __init__(self, n=3, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False); self.bn1 = nn.BatchNorm2d(16)
        layers, cin = [], 16
        for cout, stride in [(16, 1), (32, 2), (64, 2)]:
            for i in range(n):
                layers.append(Block(cin, cout, stride if i == 0 else 1)); cin = cout
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(64, num_classes)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layers(x)
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))


# ---------------------------------------------------------------- io helpers
def find_file(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if not hits: raise FileNotFoundError(f"{name} not found under /kaggle/input")
    return hits[0]

def load_fp32(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = ResNetCifar(3).to(device)
    m.load_state_dict(ck["state_dict"])
    return m.eval(), ck

def get_cifar10(bs=500, workers=2):
    hits = glob.glob("/kaggle/input/**/cifar-10-batches-py", recursive=True)
    root, dl = (os.path.dirname(hits[0]), False) if hits else ("/kaggle/working/data", True)
    tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    tr = torchvision.datasets.CIFAR10(root, train=True, download=dl, transform=tf)
    te = torchvision.datasets.CIFAR10(root, train=False, download=dl, transform=tf)
    mk = lambda d, sh: torch.utils.data.DataLoader(d, bs, shuffle=sh, num_workers=workers)
    return mk(tr, True), mk(te, False)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); c = n = 0
    for x, y in loader:
        c += (model(x.to(device)).argmax(1).cpu() == y).sum().item(); n += len(y)
    return 100 * c / n


# ---------------------------------------------------------------- Gaussian-smoothed surrogate gradient
def gauss_grad(u, qmin, qmax, sig):
    """d/du E[Q(u+eps)], eps~N(0,sig^2), in step units (u = x/Delta, sig = sigma/Delta).
    Q(u)=clamp(round(u),qmin,qmax) has unit jumps at thresholds qmin+.5 .. qmax-.5, so the
    smoothed gradient is a sum of unit-area Gaussian bumps at those thresholds. Sum is truncated
    to the nearest ~3*sig thresholds; thresholds outside the clip range are excluded (-> zero grad
    in the saturated region)."""
    K = max(1, math.ceil(3 * sig))
    j = torch.round(u)
    out = torch.zeros_like(u)
    c = 1.0 / (sig * math.sqrt(2 * math.pi))
    for d in range(-K - 1, K + 1):
        t = j + d + 0.5
        valid = (t >= qmin + 0.5 - 1e-6) & (t <= qmax - 0.5 + 1e-6)
        out = out + valid.to(u.dtype) * c * torch.exp(-0.5 * ((u - t) / sig) ** 2)
    return out

class FakeQuant(torch.autograd.Function):
    """Forward: exact uniform quantizer. Backward: clipped STE or Gaussian-smoothed gradient."""
    @staticmethod
    def forward(ctx, x, scale, qmin, qmax, mode, sig):
        u = x / scale
        ctx.save_for_backward(u)
        ctx.cfg = (qmin, qmax, mode, sig)
        return torch.clamp(torch.round(u), qmin, qmax) * scale
    @staticmethod
    def backward(ctx, g):
        (u,) = ctx.saved_tensors
        qmin, qmax, mode, sig = ctx.cfg
        if mode == "ste":
            m = ((u >= qmin - 0.5) & (u <= qmax + 0.5)).to(g.dtype)
        else:
            m = gauss_grad(u, qmin, qmax, sig)
        return g * m, None, None, None, None, None


# ---------------------------------------------------------------- quantized layer (BN is NOT folded)
class QLayer(nn.Module):
    """Fake-quantizes the layer INPUT (activation) and holds a fake-quantized weight."""
    def __init__(self, layer, a_bits, w_bits, a_signed):
        super().__init__()
        self.layer, self.a_bits, self.w_bits, self.a_signed = layer, a_bits, w_bits, a_signed
        if a_signed: self.qmax = 2 ** (a_bits - 1) - 1; self.qmin = -self.qmax
        else:        self.qmax = 2 ** a_bits - 1;       self.qmin = 0
        self.register_buffer("a_scale", layer.weight.new_tensor(1.0))
        self.register_buffer("w_fp", layer.weight.detach().clone())
        self.calibrating, self.quant_on, self.sigma, self._buf = False, False, None, None
        self.wqm, self.wtrain = 2 ** (w_bits - 1) - 1, False

    def forward(self, x):
        if self.calibrating:
            v = x.detach().flatten()
            idx = torch.randint(0, v.numel(), (100000,), device=v.device)
            self._buf = (self._buf or []) + [v[idx]]
        if self.quant_on:
            sig = self.sigma if self.sigma is not None else QCFG.sigma
            x = FakeQuant.apply(x, self.a_scale, self.qmin, self.qmax, QCFG.mode, sig)
        if self.wtrain:   # trainable latent weight, fake-quantized in forward, STE/Gaussian in backward
            sw = self.sigma if self.sigma is not None else QCFG.sigma
            wq = FakeQuant.apply(self.w_lat, self.w_scale, -self.wqm, self.wqm, QCFG.mode, sw)
            return self._apply(x, wq)
        return self.layer(x)

    def _apply(self, x, w):
        l = self.layer
        if isinstance(l, nn.Conv2d):
            return F.conv2d(x, w, l.bias, l.stride, l.padding, l.dilation, l.groups)
        return F.linear(x, w, l.bias)

    def enable_weight_training(self):
        """Latent FP weight becomes a Parameter; forward starts identical to the PTQ model."""
        self.w_lat = nn.Parameter(self.w_fp.clone()); self.wtrain = True

    def calibrate_act(self):
        v = torch.cat(self._buf); self._buf = None
        mx = v.abs().max()
        best = (float("inf"), None)
        for r in torch.linspace(*CALIB.act):
            s = r * mx / self.qmax
            q = torch.clamp(torch.round(v / s), self.qmin, self.qmax) * s
            e = ((v - q) ** 2).mean().item()
            if e < best[0]: best = (e, s)
        self.a_scale.copy_(best[1])

    def quantize_weight(self):
        w, qm = self.w_fp, 2 ** (self.w_bits - 1) - 1
        shp = (-1,) + (1,) * (w.dim() - 1)
        mx = w.abs().flatten(1).max(1)[0].clamp_min(1e-8)
        best_err = torch.full_like(mx, float("inf")); best_s = mx / qm
        for r in torch.linspace(*CALIB.w):
            s = r * mx / qm
            wq = torch.clamp(torch.round(w / s.view(shp)), -qm, qm) * s.view(shp)
            err = ((w - wq) ** 2).flatten(1).sum(1)
            better = err < best_err
            best_err = torch.where(better, err, best_err); best_s = torch.where(better, s, best_s)
        self.register_buffer("w_scale", best_s.view(shp).clone())
        wq = torch.clamp(torch.round(w / best_s.view(shp)), -qm, qm) * best_s.view(shp)
        self.layer.weight.data.copy_(wq)

def quantize_model(fp_model, w_bits, a_bits, calib_loader, device, n_calib_batches=4, edge_bits=8):
    """PTQ: per-channel MSE-clipped weights, per-tensor MSE-clipped input activations.
    First conv and final FC use edge_bits (8)."""
    m = copy.deepcopy(fp_model).eval()
    first = m.conv1
    qlayers = []
    def wrap(parent):
        for name, child in list(parent.named_children()):
            if isinstance(child, (nn.Conv2d, nn.Linear)):
                edge = (child is first) or isinstance(child, nn.Linear)
                q = QLayer(child, edge_bits if edge else a_bits, edge_bits if edge else w_bits,
                           a_signed=(child is first))
                setattr(parent, name, q); qlayers.append(q)
            else:
                wrap(child)
    wrap(m)
    for q in qlayers: q.calibrating = True
    with torch.no_grad():
        for i, (x, _) in enumerate(calib_loader):
            if i >= n_calib_batches: break
            m(x.to(device))
    for q in qlayers:
        q.calibrating = False; q.calibrate_act(); q.quantize_weight(); q.quant_on = True
    return m, qlayers
