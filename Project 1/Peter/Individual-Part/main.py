from methods import LineSearch
from sympy import *
"""linesearch = LineSearch("NM")   # Task iv
result = linesearch.testiv()
print(result)
with open("./result4.txt", "w") as f:
    for r in result:
        f.write(f"Iterations: {r['Iterations']}, xk: {r['xk']}, diff_xx: {r['diff_xx']}, calc_xk: {r['calc_xk']}, residual: {r['residual']},\n")"""
"""linesearch = LineSearch("SD")   # Task iii
res = linesearch.testiii()
print(res)"""
"""modi = ["NM","SD"]   # Task ii
for m in modi:
    linesearch = LineSearch(m)
    result = linesearch.testii()
    print(result)
    with open("./result2.txt", "a") as f:
        f.write(f"Second Task {m}:\n")
        for r in result:
            f.write(f"Iterations: {r['Iterations']}, xk: {r['xk']}, residual: {r['residual']}\n")"""


"""modi = ["SD","NM"]   # Task i
for m in modi:
    linesearch = LineSearch(m)
    result = linesearch.testi()
    print(result)
    with open("./result.txt", "a") as f:
        f.write(f"First Task {m}:\n")
        for r in result:
            f.write(f"Iterations: {r['Iterations']}, xk: {r['xk']}, residual: {r['residual']}, problem: {r['function']}\n")

"""
