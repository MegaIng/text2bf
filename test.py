from number2number import get_n2n
from simple_bf import Interpreter
from statemachine2bf import BinaryStateMachine

bsm = BinaryStateMachine()

bsm.add(0, 0, 0, get_n2n((1, 2))[ord("A")] + ".[-]")
bsm.add(0, 1, 1, get_n2n((1, 2))[ord("B")] + ".[-]")
bsm.add(1, 0, 1, get_n2n((1, 2))[ord("C")] + ".[-]")
bsm.add(1, 1, 0, get_n2n((1, 2))[ord("D")] + ".[-]")

code = "".join(bsm.generate_bf())

# print(code)
i = Interpreter(256)
i.execute("+->>>>>+-<<+")
i.print()
print("-" * 100)
i.execute(code)
i.print()
