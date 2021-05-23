from collections import defaultdict
from dataclasses import dataclass, field
from itertools import zip_longest
from pprint import pprint

from bitarray import bitarray

from huffman import Huffman, HuffmanTree
from number2number import get_n2n, forward, backward
from text2bf import linear_create


@dataclass
class BinaryStateMachine:
    transitions: defaultdict[int, dict[int, tuple[int, str]]] = field(
        default_factory=lambda: defaultdict(dict))

    def add(self, start: int, symbol: int, end: int, action: str = None):
        trans = self.transitions[start]
        assert symbol in (0, 1), symbol
        if symbol in trans:
            raise ValueError(f"Transition for {start} -{symbol}-> already added")
        trans[symbol] = end, (action or "")

    def generate_bf(self):
        # Layout input:  [0, 0, 0, state, 0, symbol, 0, 0]
        #                           ^
        # Layout output: [0, 0, 0, state, 0, 0, 0]
        #                           ^
        cases = sorted(self.transitions.items())
        n2n_12 = get_n2n((1, 2))
        n2n_23 = get_n2n((2, 3))
        current = 0
        yield "<<+>>"
        for state, transition in cases:
            d = current - state
            yield n2n_23[d]
            current = state
            yield "[>]<<"
            # When current_state is zero, we point at a 1, otherwise at a 0
            if True:  # old state is state
                # Layout:  [0, 1, 0, 0, 0, symbol, 0, 0]
                #              ^
                yield "[>>>>"
                # Now we point at symbol
                if True:  # symbol == 1
                    yield "[-<<<<->>"
                    yield n2n_12[transition[1][0]]
                    yield ">" + transition[1][1]
                    yield ">"
                    # Layout:  [0, 0, 0, state, 0, 0, 0]
                    #                              ^
                    yield "<]"
                yield "<<<<"
                if True:  # symbol == 0
                    yield "[->>"
                    # Layout:  [0, 0, 0, 0, 0, 0, 0, 0]
                    #                    ^
                    yield n2n_12[transition[0][0]]
                    yield ">" + transition[0][1]
                    yield "<<<<]"
                # Layout:  [0, 0, 0, state, 0, 0, 0]
                #           ^
                yield "]"
            yield ">"
        yield "[+]>>"

    @classmethod
    def from_huffman_tree(cls, root: HuffmanTree):
        self = cls()
        i = 0
        n2n = get_n2n((1, 2))

        def _rec(tree):
            nonlocal i
            state = i
            i += 1
            for s, v in enumerate(tree):
                if isinstance(v, tuple):
                    self.add(state, s, _rec(v))
                else:
                    self.add(state, s, 0, "" + n2n[ord(v)] + ".[-]")
            return state

        _rec(root)
        return self


def huffman_walker(root: HuffmanTree, decoders):
    # Layout input: [-1, ...symbols, 0, 0...]
    #                           ^
    # Layout SM   : [-1, symbols, 0, 0, 0, state, 0, symbol, 0...]
    sm = BinaryStateMachine.from_huffman_tree(root)
    pprint(root)
    pprint(sm.transitions)
    yield "+[-"
    for decoder in decoders:
        yield decoder
        yield ">>>>"
        yield from sm.generate_bf()
        yield "<<<<"
    yield ">>>>"
    yield "[-<+>]"
    yield "<<<<"
    yield "<"
    yield "+]-"


def huffman_output_bits(text: str):
    h = Huffman.from_dist(text)
    pprint(h.to_bits)
    yield from linear_create((-1, *(int(b) for b in ''.join(h.pack(text))[::-1])), ">")
    yield "<|"
    yield from huffman_walker(h.tree, decoders=["[->>>>>>+<<<<<<]"])


def huffman_output_pairs(text: str):
    h = Huffman.from_dist(text)
    pprint(h.to_bits)
    bits = [int(b) for b in ''.join(h.pack(text))[::-1]]
    pairs = [a + 2 * b for a, b in zip_longest(bits[::2], bits[1::2], fillvalue=0)]
    yield from linear_create((-1, *pairs), ">")
    yield "<|"
    yield from huffman_walker(h.tree, decoders=[
        ">>-<<[->+<[->->>>>>+<<<<<]]+[->+]<[-<+>]<",
        "[->>>>>>+<<<<<<]"
    ])
