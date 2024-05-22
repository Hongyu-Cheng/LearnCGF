import numpy as np
np.random.seed(0)
import gurobipy as gp
from gurobipy import GRB
from fractions import Fraction as frac


class IP:
    def __init__(self, A=None, c=None, b=None, is_standard=False, sense='maximize', treesize_limit=100000, presolve=False, verbose=False):
        """
        Input IP problem form: 
        max c^T x
        s.t. Ax <= b, x >= 0, x is integer
        """
        self.sense = sense
        self.verbose = verbose
        self.presolve = presolve
        self.is_standard = is_standard
        self.model = gp.Model()
        if verbose == False:
            self.model.setParam("OutputFlag", 0)
        if presolve == False:
            self.model.setParam("PreCrush", 1)
            self.model.setParam("Heuristics", 0.0)
            self.model.setParam("Cuts", 0)
            self.model.setParam("Presolve", 0)
        self.is_optimized = False
        self.treesize_limit = treesize_limit
        self.model.setParam("NodeLimit", self.treesize_limit)
        self.A, self.c, self.b = A.copy(), c.copy(), b.copy()
        self.m, self.n = self.A.shape
        x = self.model.addVars(self.n, vtype=GRB.INTEGER, lb=0, name="x")
        self.model.setObjective(sum(self.c[i] * x[i] for i in range(self.n)), GRB.MAXIMIZE)
        self.model.addConstrs((gp.quicksum(self.A[i, j] * x[j] for j in range(
            self.n)) <= self.b[i] for i in range(self.m)), name="cons")
        self.model.update()

    def optimize(self):
        self.model.optimize()
        self.is_optimized = True
    
    def add_cut(self, cut):
        a_x = cut[:self.n]
        a_s = cut[self.n:]
        alpha = self.A.T @ a_s - a_x
        beta = self.b @ a_s - 1.0
        self.model.addConstr(gp.quicksum(alpha[i] * self.model.getVars()[i] for i in range(self.n)) <= beta)
        self.model.update()
        self.is_optimized = False
        self.m += 1
        self.A = np.vstack((self.A, alpha))
        self.b = np.append(self.b, beta)
        

    @property
    def treesize(self):
        if self.model.status == GRB.OPTIMAL:
            return self.model.NodeCount
        elif self.model.status == 1:
            print("Warning: The model has not been optimized yet.")
            return -1
        else:
            print("Warning: Tree size limit reached.")
            return self.treesize_limit

    @property
    def info(self):
        return {
            "Num of cons": self.m, 
            "Num of vars": self.n, 
            "Is optimized?": bool(self.is_optimized), 
            "Tree size limit": self.treesize_limit
        }
    
    @property
    def x_LP(self):
        self.LP.optimize()
        return np.array(self.LP.getAttr('x',self.LP.getVars()))

    @property
    def x_IP(self):
        if self.model.status == GRB.OPTIMAL:
            return np.array(self.model.getAttr('x',self.model.getVars()))
        else:
            self.optimize()
            return np.array(self.model.getAttr('x',self.model.getVars()))
        
    @property
    def z_LP(self):
        self.LP.optimize()
        return self.LP.objVal
    
    @property
    def z_IP(self):
        if self.model.status == GRB.OPTIMAL:
            return self.model.objVal
        else:
            self.optimize()
            return self.model.objVal

    def get_candidate_rows(self):
        self.fraction_tableau=get_simplex_tableau(self.A, self.c, self.b)[1:]
        self.np_tableau = np.array(self.fraction_tableau).astype(float)[:-1,:]
        self.candidate_rows_indices = np.where(~np.isclose(self.np_tableau[:,-1], np.round(self.np_tableau[:,-1])))[0].tolist()   
        return self.np_tableau[self.candidate_rows_indices,:-1], self.np_tableau[self.candidate_rows_indices,-1]

def get_simplex_tableau(A, c, b):
    tab = [[-frac(x) for x in c] + [0]*(len(b)+1)] + [[frac(x) for x in r] + [i==j for j in range(len(A))] + [frac(b[i])] for i, r in enumerate(A)]
    while any(x < 0 for x in tab[0][:-1]):
        j_star = min((x, i) for i, x in enumerate(tab[0][:-1]))[1]
        ratios = [(x[-1] / x[j_star], i) for i, x in enumerate(tab) if x[j_star] > 0]
        if not ratios: return None
        i_star = min(ratios)[1]
        tab[i_star] = [x / tab[i_star][j_star] for x in tab[i_star]]
        tab = [x if i == i_star else [y - x[j_star] * tab[i_star][j] for j, y in enumerate(x)] for i, x in enumerate(tab)]
    return tab
