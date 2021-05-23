from typing import TypeAlias, Iterable
from collections import Counter

from bitarray import bitarray

from text2bf import linear_create

HuffmanTree: TypeAlias = "tuple[str | HuffmanTree, str | HuffmanTree]"


class Huffman:
    def __init__(self, tree: HuffmanTree):
        self.tree: HuffmanTree = tree
        self.to_bits: dict[str, str] = {}
        self.from_bits: dict[str, str] = {}

        def _rec(prefix: str, t: HuffmanTree):
            for c, s in zip("01", t):
                p = prefix + c
                if isinstance(s, str):
                    self.to_bits[s] = p
                    self.from_bits[p] = s
                else:
                    _rec(p, s)

        _rec("", tree)

    def pack(self, text: str) -> Iterable[str]:
        return (self.to_bits[c] for c in text)

    def unpack(self, bits: str) -> Iterable[str]:
        current = ""
        for b in bits:
            current += b
            if current in self.from_bits:
                yield self.from_bits[current]
                current = ""
        assert not current, current

    @staticmethod
    def create_tree(dist: Counter):
        dist = dist.copy()
        while len(dist) > 1:
            a, b = dist.most_common()[-2:]
            k = a[0], b[0]
            v = a[1] + b[1]
            del dist[a[0]], dist[b[0]]
            dist[k] = v
        (res, _), = dist.most_common()
        return res

    @classmethod
    def from_dist(cls, dist):
        return cls(cls.create_tree(Counter(dist)))


# txt = open("book.txt").read()
#
# hf = Huffman.from_dist(txt)
# print(max(len(v) for v in hf.from_bits))
# bits = "".join(hf.pack(txt))
# print(len(txt), len(bits) / 8, (len(bits) / 8) / len(txt))
# res = "".join(hf.unpack(bits))
# assert txt == res, res[:100]
# bits_as_bytes = bitarray(bits).tobytes()
# bf_cost = bits.count("0")*1+bits.count("1")*2
# print(bf_cost/len(txt))
# print(len(bits_as_bytes)/len(txt))
# print((''.join(linear_create(bitarray(bits),">")))[:100])
# print(len(''.join((*linear_create(bitarray(bits),">"),*linear_create(range(0,255),">"))))/len(txt))
# print(len(''.join(linear_create(bits_as_bytes,">")))/len(txt))