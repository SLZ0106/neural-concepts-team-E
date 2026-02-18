import json, random, numpy as np
from math import isclose

OUTCOMES = [-10, 10, 30]
N_OBS = 60

def dist_stats(outcomes, probs):
    mu = sum(o*p for o,p in zip(outcomes, probs))
    var = sum(((o-mu)**2)*p for o,p in zip(outcomes, probs))
    return mu, var

def sample_counts(q, n=N_OBS):
    probs = [q, 1-2*q, q]
    return np.random.multinomial(n, probs).tolist()  # aligned to OUTCOMES

def make_statement(domain, counts, rng):
    # randomize outcome order in the text
    triplets = list(zip(OUTCOMES, counts))
    rng.shuffle(triplets)

    # add distractor numbers unrelated to variance
    fiscal_year = rng.choice([2018, 2019, 2020, 2021, 2022, 2023])
    dept_code = rng.randint(100, 999)        # unrelated
    budget = rng.choice([12.5, 18.0, 25.0])  # unrelated

    # varied templates
    templates = [
        "FY{fy} Dept-{dc} budget {bd}M. In 60 observations, outcomes were {a} ({ca} times), {b} ({cb} times), {c} ({cc} times).",
        "Reference FY{fy}-{dc}. Over 60 trials we saw: {a}:{ca}, {b}:{cb}, {c}:{cc}. (Budget {bd}M)",
        "FY{fy} code {dc}. Counts across 60 cases -> {a}={ca}, {b}={cb}, {c}={cc}. Budget={bd}M."
    ]
    t = rng.choice(templates)

    (a, ca), (b, cb), (c, cc) = triplets
    return t.format(fy=fiscal_year, dc=dept_code, bd=budget, a=a, ca=ca, b=b, cb=cb, c=c, cc=cc)

def generate_dataset(N=600, seed=7):
    rng = random.Random(seed)
    np.random.seed(seed)

    domains = ["finance","weather","health","sports","education"]
    data=[]
    for i in range(1, N+1):
        target_var = (i-1) * (400/(N-1))
        q = target_var/800.0
        probs = [q, 1-2*q, q]
        mu, var = dist_stats(OUTCOMES, probs)
        counts = sample_counts(q)
        domain = domains[(i-1)%len(domains)]
        statement = make_statement(domain, counts, rng)
        data.append({
            "id": i,
            "statement": statement,
            "counts": counts,
            "mean": round(mu, 6),
            "variance": round(var, 6),
            "domain": domain
        })
    return data

if __name__ == "__main__":
    ds = generate_dataset()
    with open("synthetic_uncertainty_counts_harder_600.json","w") as f:
        json.dump(ds,f,indent=2)
