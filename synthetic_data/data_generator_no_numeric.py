import json
import random
import numpy as np
from math import isclose

OUTCOMES = (-10, 10, 30)
N_OBS = 60

def dist_stats(outcomes, probs):
    mu = sum(o*p for o, p in zip(outcomes, probs))
    var = sum(((o - mu) ** 2) * p for o, p in zip(outcomes, probs))
    return mu, var

def sample_counts(q, n=N_OBS):
    probs = [q, 1-2*q, q]
    counts = np.random.multinomial(n, probs)
    return counts.tolist()

def make_statement(domain, counts):
    c1, c2, c3 = counts
    templates = {
        "finance": [
            "In 60 past quarters, returns were -10% ({c1} times), 10% ({c2} times), and 30% ({c3} times).",
        ],
        "weather": [
            "In 60 observations, temperature changes were -10 ({c1}), 10 ({c2}), and 30 ({c3}).",
        ],
        "health": [
            "Across 60 patients, outcome scores were -10 ({c1}), 10 ({c2}), and 30 ({c3}).",
        ],
        "sports": [
            "Over 60 games, point differentials were -10 ({c1}), 10 ({c2}), and 30 ({c3}).",
        ],
        "education": [
            "Across 60 quizzes, score changes were -10 ({c1}), 10 ({c2}), and 30 ({c3}).",
        ],
    }
    t = templates[domain][0]
    return t.format(c1=c1, c2=c2, c3=c3)

def generate_dataset(N=600, seed=7):
    random.seed(seed)
    np.random.seed(seed)

    domains = ["finance","weather","health","sports","education"]
    data = []

    for i in range(1, N+1):
        target_var = (i-1) * (400/(N-1))
        q = target_var / 800.0

        probs = [q, 1-2*q, q]
        mu, var = dist_stats(OUTCOMES, probs)

        counts = sample_counts(q)
        domain = domains[(i-1) % len(domains)]

        statement = make_statement(domain, counts)

        item = {
            "id": i,
            "statement": statement,
            "counts": counts,
            "mean": round(mu, 6),
            "variance": round(var, 6),
            "domain": domain
        }
        data.append(item)

    return data

if __name__ == "__main__":
    dataset = generate_dataset()
    with open("synthetic_uncertainty_counts_600.json","w") as f:
        json.dump(dataset,f,indent=2)
    print("Wrote synthetic_uncertainty_counts_600.json")
