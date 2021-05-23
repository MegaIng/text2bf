from collections import defaultdict
from typing import Optional


class Interpreter:
    def __init__(self, wrap: int = None):
        self.wrap: Optional[int] = wrap
        self.cells: defaultdict[int, int] = defaultdict(int)
        self.index: int = 0

    @property
    def value(self):
        return self.cells[self.index]

    def add(self, step: int, offset: int = 0):
        i = self.index + offset
        vn = self.cells[i] + step
        if self.wrap is not None:
            vn %= self.wrap
        self.cells[i] = vn

    def move(self, step: int):
        self.index += step

    def read(self, offset: int = 0):
        raise NotImplementedError

    def write(self, offset: int = 0):
        v = self.cells[self.index + offset]
        print(chr(v), end='')

    def print(self):
        mi, ma = min(*self.cells, self.index), max(*self.cells, self.index)
        w = len(str(self.wrap)) + 2
        print(*(f"{i:^{w}}" if i != self.index else f"[{i:^{w-2}}]" for i in range(mi, ma + 1)), sep="|")
        print(*(f"{self.cells[i]:^{w}}" for i in range(mi, ma + 1)), sep="|")

    def execute(self, code: str):
        def jump_forward():
            nonlocal i, pairs, open_pairs
            if self.value:
                if i not in pairs:
                    open_pairs.append(i)
            elif i in pairs:
                i = pairs[i]
            else:
                sts = len(open_pairs)
                while i < len(code):
                    match code[i]:
                        case '[':
                            open_pairs.append(i)
                        case ']':
                            op = open_pairs.pop()
                            pairs[op] = i
                            pairs[i] = op
                            if len(open_pairs) == sts:
                                break
                    i += 1
        i = 0
        pairs: dict[int, int] = {}
        open_pairs: list[int] = []
        while i < len(code):
            match code[i]:
                case '+':
                    self.add(1)
                case '-':
                    self.add(-1)
                case '>':
                    self.move(1)
                case '<':
                    self.move(-1)
                case '.':
                    self.write()
                case ',':
                    self.read()
                case '[':
                    jump_forward()
                case ']':
                    if i not in pairs:
                        if not open_pairs:
                            raise ValueError(f"Unmatched Closing Parentheses at {i}")
                        op = open_pairs.pop()
                        pairs[op] = i
                        pairs[i] = op
                    if self.value:
                        i = pairs[i]
                case '@':
                    self.print()
                    NL = '\n'
                    input(f"last_line: {code[code.rfind(NL, 0, i)+1:i+1]}")
            i += 1