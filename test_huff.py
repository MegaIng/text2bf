from itertools import zip_longest

from huffman import Huffman
from number2number import cleanup
from simple_bf import Interpreter
from statemachine2bf import huffman_output_bits, huffman_output_pairs
from text2bf import linear_create

book = open("book.txt").read()

text = book#[:100]

hu = Huffman.from_dist(text)

bits = [int(c) for s in hu.pack(text) for c in s][::-1]

pairs = [a+2*b for a,b in zip_longest(bits[::2], bits[1::2], fillvalue=0)]
nibbles = [a+4*b for a,b in zip_longest(pairs[::2], pairs[1::2], fillvalue=0)]
print("bits", len("".join(linear_create(bits,">")))/len(text))
print("pairs", len("".join(linear_create(pairs, ">")))/len(text))

for f in (huffman_output_bits, huffman_output_pairs):
    code = cleanup("".join(f(text)))
    data = code.index("|")
    walker = len(code)-data

    print(f)
    print(data/len(code))
    print(walker/len(code))
    print(len(code) / len(text))

# i = Interpreter()
#
# i.execute(code)
