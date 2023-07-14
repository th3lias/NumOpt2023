import sympy
from matplotlib import pyplot   # for plotting
from sympy import *

class LineSearch:
    def __init__(self,mode:str):
        if mode.upper() == "SD":    # steepest descent
            self.mode = 1
        elif mode.upper() == "NM":      # Newton Method
            self.mode = 2
        else:
            raise ValueError("Only 'SD' for steepest descent and 'NM' for Newton Method are allowed values for method.")
        self.var = None
        self.gradf = None
        self.hessef_inv = None
        self.length_f = None
        self.alphaNM = None


    def pk(self, f, x):
        var = self.var
        if self.mode == 1:  # pk for steepest descent
            return (-1)*self.gradf.subs({var[i]:v[0] for i,v in enumerate(x.tolist())})
        elif self.mode == 2:    # pk for newton method
            return (-1)*self.hessef_inv.subs({var[i]:v[0] for i,v in enumerate(x.tolist())}) * self.gradf.subs({var[i]:v[0] for i,v in enumerate(x.tolist())})


    def d2function(self, f, xo:sympy.Matrix):
        f = Matrix([f])
        if type(xo)!=sympy.Matrix:
            raise ValueError("xo has to be a matrix with dim = len(var)")
        var = self.var
        x = xo
        self.gradf = self.grad(f)
        self.hessef_inv = self.hesse(f).inv()
        self.length_f = sqrt(f.transpose()*f)[0,0]
        df = self.gradf.subs({var[i]:v[0] for i,v in enumerate(x.tolist())}) # calculate df(x)
        #df = gradf.subs({var[i]:v for i,v in enumerate(x.tolist()[0])})
        dfdf = (df.transpose()*df)[0]  # |df(x)|
        k = 1
        while sqrt(dfdf)>10**(-6):  # stop condition
            pk = self.pk(f, x)  # get pk
            alpha = self.alpha_strongWolfe(f,x,pk)  # get step size
            x = x + alpha * pk  # line search method
            xtemp = x.tolist()
            df = self.gradf.subs({var[i]: v[0] for i, v in enumerate(xtemp)})   # calc. for stop condition
            dfdf = (df.transpose()*df)[0,0]
            k += 1
            print(k)
            if k>3000:
                break

        return abs(sqrt(dfdf)), x, k


    def testi(self):
        self.var = [S("x")]
        x = self.var[0]
        problems = [(x - 5)*(x + 3)*(x - 10), (x + 2)*(x - 3)*(x + 14)*(x + 7)*(x - 5),   # 5 different Functions
                    (2*x - 10)*(x + 2)*(x - 6)*(x + 18)*(2*x + 18), (x - 10)*(x + 20)*(x - 30),(x + 10)*(x - 3)*(x - 10)*(x + 7)*(x - 5)]
        problems = [integrate(problem,x) for problem in problems]   # Integrate them to get the zero-points as stable points
        #ranges = [range(-40,150), range(-150, 100), range(-200,100), range(-150,350), range(-120,120)] # for plotting
        result = []
        for i,problem in enumerate(problems):
            print(f"{i+1}/5")
            res = self.d2function(problem,Matrix([[0]]))
            result.append({"function": problem, "Iterations": res[2],"xk": simplify(res[1].values()[0]).evalf(), "residual": simplify(res[0]).evalf()})
        return result
        """fix, ax = pyplot.subplots(5, 1)  # Just for plotting the functions
        for i,p in enumerate(problems):
            if i == 5:
                break
            self.plot2d(ax[i],ranges[i], p, [1, 2], [1, 2])
        pyplot.show()"""

    def testii(self):
        x = S("x")
        functions = [x, x**2,(x - 2) * (x + 3) * (x - 4),2**x,exp(x)]
        self.alphaNM = 1
        q = [10]*len(functions)
        m = [100]*len(functions)
        n = [2, 4, 5, 10, 20]
        a = [[(qx/m[i]) * j * (-1)**j for j in range(1,m[i]+1)]for i,qx in enumerate(q)]
        b = [[f.subs({x: ai}) for ai in a[i]]for i,f in enumerate(functions)]
        result = []
        for i1,temp in enumerate(a):
            ni = n[i1] + 1 # set from k = 0 until k = n
            self.var = symbols(" ".join([f"x{i}" for i in range(ni)]))
            t = S("t")
            PHI = sum([xi*t**i for i,xi in enumerate(self.var)])
            f = 0.5 * sum([(PHI.subs({t: ai}) - b[i1][i2])**2 for i2,ai in enumerate(temp)])
            f = simplify(f)
            print(f"{i1 + 1}/5")
            res = self.d2function(f,Matrix([0]*ni))
            result.append({"function": f, "Iterations": res[2], "xk": res[1].tolist(),
                           "residual": simplify(res[0]).evalf()})

        return result

    def testiii(self):
        nlist = [5, 8, 12, 20, 30]
        for n in nlist:
            s = symbols(" ".join([f"x{n}" for n in range(n)]))
            self.var = s
            Q = Matrix([[1/(i+2+j-1) for j in range(n)] for i in range(n)])
            b = Matrix([1]*n)
            xo = Matrix([0]*n)
            x = Matrix([[v] for v in s])
            f = simplify((1/2)*x.transpose()*Q*x-b.transpose()*x)
            x_star = linsolve(Q*x-b, *self.var)
            return x_star, self.d2function(f,xo) ,Q, x, b

    def testiv(self):
        x,y = symbols("x y", real=True)
        self.var = x,y
        self.alphaNM = 0.5
        q = [(-10) * x ** 2 + 10 * y, (-x) + 1, 10 * x + 2 * (y + 6) * y, (y + 1)**2, -(5 * x) ** 2 + 8 * y+ 4, (x + y)**2]
        functions = [q[i]**2 + q[i+1]**2 for i in range(5)] # define all functions
        zeros = [self.zeros(f) for f in functions]  # calculate analytically the results to compare them with NM
        result = []
        print("Zeros are finished")
        i = 0
        for f in functions:
            print(f"{i + 1}/5")
            f = simplify(f)
            xo = Matrix([0]*2)
            res = self.d2function(f,xo) # apply NM
            result.append({"function": f, "Iterations": res[2], "xk": res[1].tolist(),  # save the result
                           "residual": simplify(res[0]).evalf()})
            i += 1
        for i,z in enumerate(zeros):    # calculate |x - x*| and save the nearest x*
            z_min, diff_xx = find_minddif(z,result[i]["xk"])
            result[i]["diff_xx"] = diff_xx.evalf()
            result[i]["calc_xk"] = z_min[0].evalf(), z_min[1].evalf()
        return result


    def plot2d(self, ax, rang, freal, xk, yk):  # to plot the result
        ax.plot([x/10 for x in rang],[freal.subs(self.var[0], x/10) for x in rang], c = "blue", label = f"f = {freal}")
        #ax.scatter(xk, yk, c="red", label = "steepest descent" if self.mode == 1 else "Newton Method")
        ax.legend()
        ax.grid()

    def zeros(self,f):
        df = self.nabla(f).tolist()
        t = solve((df[0][0],df[1][0]),self.var) # first derivative to zero and solve
        res = []
        for x in t:
            if im(x[0])==0 and im(x[1])==0:     # only real points are interesting
                res.append([x[0], x[1]])
        return res

    def nabla(self,f):
        var = self.var
        matrix_scalar_function = Matrix([f])
        return matrix_scalar_function.jacobian(var).transpose()     # caclulate df with the jacobian matrix

    def grad(self,f):
        var = self.var
        return Matrix([[diff(f,v)]for v in var])

    def hesse(self,f):
        return Matrix([[f.diff(x).diff(y) for x in self.var] for y in self.var])

    def alpha_strongWolfe(self, f, x, pk):  # define step size
        if self.mode == 2:  # In case of NM
            return self.alphaNM
        var = self.var  # for the SD
        """ Initial Values """
        alpha = 1
        roh = 0.5
        c = 10**(-4)
        x_ = x + alpha * pk
        f_new = f.subs({var[i]: v[0] for i, v in enumerate(x_.tolist())})   # f(x + alpha * pk)
        f_old = f.subs({var[i]: v[0] for i, v in enumerate(x.tolist())})    # f(x)
        gradf = self.gradf.subs({var[i]: v[0] for i, v in enumerate(x.tolist())})
        gradfpk = gradf.transpose()*pk  # grad(f)**T * pk
        k = 0
        while not f_new[0] <= f_old[0]+c*alpha*gradfpk[0]:  # stronger wolf condition
            alpha *= roh
            x_ = x + alpha * pk
            f_new = f.subs({var[i]: v[0] for i, v in enumerate(x_.tolist())})
            f_old = f.subs({var[i]: v[0] for i, v in enumerate(x.tolist())})
            gradf = self.gradf.subs({var[i]: v[0] for i, v in enumerate(x.tolist())})
            gradfpk = gradf.transpose() * pk
            k += 1
            if k >= 500:
                print(f"alpha break {alpha}")
                break
        print(f"alpha {alpha}")
        return alpha


def find_minddif(list = None, point = None):    # for task iv, to get the x* value with the smallest squared error to x~
    temp = [None]*2
    i = 0
    for l in list:
        t = sqrt((point[0][0] - l[0])**2 + (point[1][0] - l[1])**2)
        if None in temp or t < temp[0]:
            temp = t, l
        i += 1
    return temp[1], temp[0]



