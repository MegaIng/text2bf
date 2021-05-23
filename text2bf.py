import random
from collections import Counter
from dataclasses import dataclass, field
from functools import cache
from itertools import permutations, zip_longest
from math import factorial
from pprint import pprint
from typing import Iterable

import progressbar

from number2number import get_n2n, Number2Number, forward, cleanup


def linear_create(base_text: Iterable[int | str], after=".>"):
    n2n = get_n2n((1, 2), cache_dir="Number2Number")
    for c in base_text:
        yield n2n[ord(c) if not isinstance(c, int) else c] + after


def modify_one(base_text: str):
    n2n = get_n2n((1, 2), cache_dir="Number2Number")
    last = 0
    for c in base_text:
        yield n2n[ord(c) - last] + "."
        last = ord(c)


def modify_two(base_text: str):
    n2n_2 = get_n2n((1, 2), cache_dir="Number2Number")
    n2n_1 = get_n2n((-1, -2), cache_dir="Number2Number")
    yield ">>>"
    last_1, last_2 = ord(base_text[1]), ord(base_text[0])
    yield n2n_1[last_1]
    yield ">"
    yield n2n_2[last_2]
    last = 2
    for c in base_text:
        v = ord(c)
        if abs(v - last_1) <= abs(v - last_2):
            if last == 2:
                yield "<"
            yield n2n_1[v - last_1] + "."
            last_1 = v
            last = 1
        else:
            if last == 1:
                yield ">"
            yield n2n_2[v - last_2] + "."
            last_2 = v
            last = 2


@dataclass(eq=False)
class Cell:
    base: int
    extras: tuple[int, ...]
    current_value: int
    _n2n: Number2Number = field(init=False, repr=False)

    def __post_init__(self):
        self._n2n = get_n2n((e - self.base for e in self.extras))

    def calculate(self, current_index: int, target_value: int) -> str:
        change = target_value - self.current_value
        offset = self.base - current_index
        return cleanup(forward(offset) + self._n2n[change])

    def apply(self, target_value: int):
        self.current_value = target_value
        return self.base


def modify_n(text: str, n: int, init_symbol=None, zero_cost_offset=2):
    cells = [Cell(i * 2, (i * 2 + 1, i * 2 + 2), 0)
             if i % 2 == 1 else
             Cell(i * 2 + 1, (i * 2, i * 2 - 1), 0)
             for i in range(n)]
    current_index = 0
    if init_symbol is not None:
        v = ord(init_symbol) if isinstance(init_symbol, str) else init_symbol
        for cell in cells:
            yield cell.calculate(current_index, v)
            current_index = cell.apply(v)
    for c in text:
        v = ord(c)
        solution = cells[0], cells[0].calculate(current_index, v)
        cost = len(solution[1])
        for cell in cells:
            code = cell.calculate(current_index, v)
            if len(code) < cost or (cell.current_value == 0 and len(code) - zero_cost_offset < cost):
                solution = cell, code
                cost = len(code)
        current_index = solution[0].apply(v)
        yield solution[1] + "."


def find_best(text: str, n_range=(1, 10), init_symbol_range=(32, 127, 30)):
    best = (None, ''.join(linear_create(text)))
    cost = len(best[1])
    bar = progressbar.ProgressBar(max_value=len(range(*n_range)) * len(range(*init_symbol_range)))
    i = 0
    bar.start()
    bar.update(i)
    for n in (range(*n_range)):
        for init in range(*init_symbol_range):
            t = ''.join(modify_n(text, n, init))
            if len(t) < cost:
                best = (n, init), t
                cost = len(t)
            i += 1
            bar.update(i)
    bar.finish()
    return best


def histogram_buckets(text: str, n: int):
    counter = Counter(text)
    mi, ma = ord(min(counter)), ord(max(counter))
    d = ma - mi
    buckets = [(mi + i * d // n, mi + (i + 1) * d // n) for i in range(n)]


def _calc_weight(layout, pair_weights):
    return sum((abs(layout[a] - layout[b]) * w) for (a, b), w in pair_weights.items())


def _optimize_layout(base_layout: dict[str, int], pair_weights: dict[frozenset[str], int], reorder: tuple[str, ...]):
    indices = tuple(base_layout[c] for c in reorder)

    base_order = dict(zip(base_layout, indices))
    best_order = base_order
    best_weight = _calc_weight(base_layout, pair_weights)
    print(best_weight, best_order)
    for p in progressbar.progressbar(permutations(reorder), max_value=factorial(len(reorder))):
        new_order = {c: indices[i] for i, c in enumerate(p)}
        base_layout.update(new_order)
        w = _calc_weight(base_layout, pair_weights)
        if w < best_weight:
            best_weight = w
            best_order = new_order
    base_layout.update(best_order)
    print(best_weight, best_order)


def _rot(layout, chars):
    first = layout[chars[0]]
    for a, b in zip(chars, chars[1:]):
        layout[a] = layout[b]
    layout[chars[-1]] = first
    return chars[::-1]


def _optimize_layout_random(layout: dict[str, int], pair_weights: dict[frozenset[str], int],
                            reorder: tuple[str, ...], attempts: int = 1000_000, swap_range: tuple[int, int] = (2, 5)):
    # best_layout = layout.copy()
    best_weight = _calc_weight(layout, pair_weights)
    print(best_weight)
    pprint(layout)
    for _ in progressbar.progressbar(range(attempts), max_value=attempts):
        r = random.sample(reorder, random.randint(*swap_range))
        inv = _rot(layout, tuple(r))
        w = _calc_weight(layout, pair_weights)
        if w < best_weight:
            best_weight = w
            # best_layout = layout.copy()
        else:
            _rot(layout, inv)
    print(best_weight)
    pprint(layout)
    # layout.update(be)


def histogram_moves(text: str):
    single_weights = Counter(text)
    print(single_weights)
    pair_weights = Counter(frozenset(t) for t in zip(text, text[1:]) if t[0] != t[1])
    pair_weights = {p: w for p, w in pair_weights.items() if w > 10}
    print(pair_weights)
    base = len(single_weights) // 2
    print(base)
    layout = {c: (base + ((i // 2) if i % 2 == 0 else -i // 2))
              for (i, (c, _)) in enumerate(single_weights.most_common())}
    print(layout)
    # _optimize_layout(layout, pair_weights, tuple(t[0] for t in single_weights.most_common(9)))
    _optimize_layout_random(layout, pair_weights, tuple(t[0] for t in single_weights.most_common()))
    assert len(layout) == len(single_weights) == len(set(layout.values()))
    pos_to_char = {i: c for c, i in layout.items()}

    n2n = get_n2n((1, 2))
    for i, c in sorted(pos_to_char.items()):
        yield cleanup(n2n[ord(c)] + ">")
    current_cell = len(pos_to_char)
    for c in text:
        d = layout[c] - current_cell
        yield forward(d) + "."
        current_cell = layout[c]


@cache
def _find_closest(layout: tuple[str], current: int, target: str):
    if target == layout[current]:
        return current
    for i, ri in zip_longest(range(current + 1, len(layout)), range(current - 1, -1, -1)):
        if i is not None and layout[i] == target:
            return i
        elif ri is not None and layout[ri] == target:
            return ri
    else:
        raise ValueError(f"Couldn't find {target!r} in {layout} from {current}")


def _calc_weight_list(layout, pair_weights):
    return sum((abs(i - _find_closest(layout, i, b)) * w)
               for (a, b), w in pair_weights.items()
               for i, c in enumerate(layout)
               if c == a)


def _optimize_layout_list_random(layout: list[str], pair_weights: dict[frozenset[str], int],
                                 reorder: tuple[str, ...], attempts: int = 100_000,
                                 swap_range: tuple[int, int] = (2, 8)):
    # best_layout = layout.copy()
    best_weight = _calc_weight_list(tuple(layout), pair_weights)
    print(best_weight)
    pprint(layout)
    known_weights = {}
    for _ in progressbar.progressbar(range(attempts), max_value=attempts):
        r = random.sample(range(len(layout)), random.randint(*swap_range))
        inv = _rot(layout, tuple(r))
        l = tuple(layout)
        if l not in known_weights:
            w = known_weights[l] = _calc_weight_list(l, pair_weights)
        else:
            w = known_weights[l]
        if w < best_weight:
            best_weight = w
            # best_layout = layout.copy()
        else:
            _rot(layout, inv)
    print(best_weight)
    pprint(layout)
    # layout.update(be)


def histogram_moves_repeated(text: str, n: int):
    single_weights = Counter(text)
    print(single_weights)
    pair_weights = Counter(frozenset(t) for t in zip(text, text[1:]) if t[0] != t[1])
    pair_weights = {p: w for p, w in pair_weights.items() if w > 10}
    print(pair_weights)
    order = single_weights.most_common()
    print(order)
    thresholds = Counter({c: w // 10 for c, w in single_weights.items()})
    count = Counter()
    while len(order) < n:
        (to_add, threshold), = thresholds.most_common(1)
        i = 0
        for i, (_, w) in enumerate(order):
            if threshold > w:
                break
        order.insert(i, (to_add, threshold))
        thresholds[to_add] = threshold // 10
    print(order)

    base = len(order) // 2
    print(base)
    layout: list[str] = [None] * len(order)
    for i, (c, _) in enumerate(order):
        layout[base + ((i // 2) if i % 2 == 0 else -i // 2)] = c
    print(layout)
    # _optimize_layout(layout, pair_weights, tuple(t[0] for t in single_weights.most_common(9)))
    _optimize_layout_list_random(layout, pair_weights, tuple(t[0] for t in single_weights.most_common()))
    # assert len(order) == len(single_weights) == len(set(order.values()))
    # pos_to_char = {i: c for c, i in order.items()}

    n2n = get_n2n((1, 2))
    for i, c in enumerate(layout):
        yield cleanup(n2n[ord(c)] + ">")
    yield "<"
    current_cell = len(layout) - 1
    layout = tuple(layout)
    for c in text:
        d = _find_closest(layout, current_cell, c) - current_cell
        yield forward(d) + "."
        current_cell += d
