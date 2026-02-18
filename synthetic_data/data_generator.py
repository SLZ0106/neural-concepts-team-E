import json
import random
from math import isclose

def dist_stats(outcomes, probs):
    mu = sum(o*p for o, p in zip(outcomes, probs))
    var = sum(((o - mu) ** 2) * p for o, p in zip(outcomes, probs))
    return mu, var

def make_statement(domain, q, style_id):
    # 保持结构一致，避免“文字本身”成为捷径
    templates = {
        "finance": [
            "An investment has three possible annual returns: -10%, 10%, and 30%, with probabilities {q:.3f}, {p10:.3f}, and {q:.3f}.",
            "A portfolio’s yearly return is -10%, 10%, or 30% with probabilities {q:.3f}, {p10:.3f}, {q:.3f}.",
        ],
        "weather": [
            "Tomorrow’s temperature change is -10, 10, or 30 (units) with probabilities {q:.3f}, {p10:.3f}, {q:.3f}.",
            "A forecast model outputs -10, 10, or 30 with probabilities {q:.3f}, {p10:.3f}, {q:.3f}.",
        ],
        "health": [
            "A treatment effect score is -10, 10, or 30 with probabilities {q:.3f}, {p10:.3f}, {q:.3f}.",
            "A patient outcome score is -10, 10, or 30 with probabilities {q:.3f}, {p10:.3f}, {q:.3f}.",
        ],
        "sports": [
            "A team’s point differential is -10, 10, or 30 with probabilities {q:.3f}, {p10:.3f}, {q:.3f}.",
            "A player’s performance index is -10, 10, or 30 with probabilities {q:.3f}, {p10:.3f}, {q:.3f}.",
        ],
        "education": [
            "A student’s scaled score change is -10, 10, or 30 with probabilities {q:.3f}, {p10:.3f}, {q:.3f}.",
            "A quiz bonus outcome is -10, 10, or 30 with probabilities {q:.3f}, {p10:.3f}, {q:.3f}.",
        ],
    }
    p10 = 1 - 2*q
    tlist = templates[domain]
    tmpl = tlist[style_id % len(tlist)]
    return tmpl.format(q=q, p10=p10)

def generate_dataset(
    N=600,
    seed=7,
    outcomes=(-10, 10, 30),
    domains=("finance","weather","health","sports","education"),
    q_min=0.0,
    q_max=0.5,
    jitter=False
):
    random.seed(seed)

    data = []
    for i in range(1, N+1):
        # 让 variance 在 0~400 上“均匀覆盖”
        # variance = 800*q  =>  q = variance/800
        target_var = (i-1) * (400/(N-1))  # 均匀铺满 [0,400]
        q = target_var / 800.0

        # 可选：加一点很小的随机扰动，避免过于规则（但会稍微破坏均匀）
        if jitter:
            q = min(max(q + random.uniform(-0.002, 0.002), q_min), q_max)

        probs = [q, 1 - 2*q, q]

        mu, var = dist_stats(outcomes, probs)

        # 保护：mean 应该永远是 10（浮点允许微小误差）
        if not isclose(mu, 10.0, abs_tol=1e-9):
            raise RuntimeError(f"Mean drifted: mu={mu} for probs={probs}")

        domain = domains[(i-1) % len(domains)]
        statement = make_statement(domain, q, style_id=i)

        item = {
            "id": i,
            "statement": statement,
            "outcomes": list(outcomes),
            "probabilities": [round(p, 6) for p in probs],
            "mean": round(mu, 6),
            "variance": round(var, 6),
            "domain": domain
        }
        data.append(item)

    return data

if __name__ == "__main__":
    dataset = generate_dataset(N=600, seed=7, jitter=False)
    with open("synthetic_uncertainty_600.json", "w") as f:
        json.dump(dataset, f, indent=2)
    print("Wrote synthetic_uncertainty_600.json with", len(dataset), "items")
    print("First 3 items:\n", json.dumps(dataset[:3], indent=2))
