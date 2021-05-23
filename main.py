from number2number import optimize_and_save_all
from simple_bf import Interpreter
from text2bf import linear_create, modify_one, modify_two, modify_n, find_best, histogram_moves, \
    histogram_moves_repeated

book = open("book.txt").read()

text = book[:1000000]

code1 = "".join(linear_create(text))
# config, code2 = find_best(text, (6, 10))
config = (8, 64)
# code2="".join(modify_n(text, *config))
# code3 = "".join(histogram_moves(text))
code3 = "".join(histogram_moves_repeated(text, 100))
# print(code3)

# i = Interpreter(256)
# i.execute(code3)
print("\n" + "-" * 100)
# print(config)
# print(len(code1), len(code2), len(code3))
# print(len(code1) / len(text), len(code2) / len(text), len(code3) / len(text), sep="\n")
print(len(code1) / len(text), len(code3) / len(text), sep="\n")
