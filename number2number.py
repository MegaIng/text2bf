import json
import os
from functools import cache
from pprint import pprint
from typing import Iterable


@cache
def factors(n):
    out = set()
    for i in range(1, int(n ** 0.5) + 1):
        if n % i == 0:
            out.add((i, n // i))
    return frozenset(out)


CLEANUP_PATTERNS = {
    '<>': '',
    '><': '',
    '+-': '',
    '-+': '',
}

@cache
def cleanup(s: str) -> str:
    old = None
    while old != s:
        old = s
        s = s.replace('<>', '').replace('><', '').replace('+-', '').replace('-+', '')
    return s


def repeated(s, n):
    return s[n < 0] * abs(n)


def forward(n):
    return repeated('><', n)


def backward(n):
    return repeated('<>', n)


def add(n):
    return repeated('+-', n)


class Number2Number:
    def __init__(self, extra_cells: Iterable[int], values: tuple[int, int],
                 cache_dir: str | None = None):
        extra_cells = set(extra_cells)
        self.extra_cells = {e: (
            get_n2n({i for i in extra_cells if i != e}, values, cache_dir),  # From result Cell
            get_n2n({i - e for i in extra_cells if i != e}, values, cache_dir)  # From chosen temp Cell
        ) for e in
            extra_cells}
        if cache:
            self.filename = f"{cache_dir}/{values[0]}_{values[1]}~{'_'.join(map(str, extra_cells))}.json"
            if os.path.exists(self.filename):
                with open(self.filename) as f:
                    self.mapping = {int(v): s for v, s in json.load(f).items()}
                return
        else:
            self.filename = None
        self.mapping = {
            i: add(i)
            for i in range(values[0], values[1] + 1)
        }

    def save(self):
        if self.filename is None:
            raise ValueError("Can't not save without a filename")
        with open(self.filename, "w") as f:
            json.dump(self.mapping, f)
        for n1, n2 in self.extra_cells.values():
            n1.save()
            n2.save()

    def __getitem__(self, item):
        if item in self.mapping:
            return self.mapping[item]
        else:  # Fallback to simple multiplication
            return add(item)

    def optimize(self):
        for (n1, n2) in self.extra_cells.values():
            n1.optimize()
            n2.optimize()
        methods = [getattr(self, n) for n in dir(self) if n.startswith("step_")]
        old = None
        while old != self.mapping:
            old = self.mapping.copy()
            for m in methods:
                for i in old:
                    s = cleanup(m(i))
                    if s is not None and len(s) < len(self.mapping[i]):
                        self.mapping[i] = s

    def step_inc(self, i: int):
        if i - 1 not in self.mapping:
            return None
        return self.mapping[i - 1] + '+'

    def step_dec(self, i: int):
        if i + 1 not in self.mapping:
            return None
        return self.mapping[i + 1] + '-'

    def step_factorize(self, i: int):
        if not self.extra_cells or not i:
            return None
        if i < 0:
            f = -1
            i *= -1
        else:
            f = 1
        p, q = min(factors(i), key=sum)
        e = min(self.extra_cells, key=abs)
        base, temp = self.extra_cells[e]
        return f"{forward(e)}{temp[q]}[{backward(e)}{base[p * f]}{forward(e)}{add(-1)}]{backward(e)}"


N2N_cache: dict[tuple[tuple[int, ...], tuple[int, int], str], Number2Number] = {}


def get_n2n(extra_cells: Iterable[int] = (), values: tuple[int, int] = (-256, 256),
            cache_dir: str | None = "Number2Number"):
    extra_cells = tuple(sorted(extra_cells))
    args = (extra_cells, values, cache_dir)
    if args not in N2N_cache:
        N2N_cache[args] = Number2Number(*args)
    return N2N_cache[args]


def optimize_and_save_all():
    for n2n in N2N_cache.values():
        n2n.optimize()
        n2n.save()
