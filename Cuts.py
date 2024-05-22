from IP import *
import numpy as np
np.random.seed(0)

def is_integer(x):
    return np.isclose(x, np.round(x))

class CGF1:
    def __init__(self, r, f, p=2, q=2, slope_clip=30):
        self.r = r - np.floor(r)
        self.f = f - np.floor(f)
        self.p = p
        self.q = q
        self.slope_clip = slope_clip
    def compute_ranges(self,slope_clip=None):
        p, q, f = self.p, self.q, self.f
        if slope_clip is None:
            slope_clip = self.slope_clip
        if (p + q - 1) * f - p < 0:
            mu1_range = [(p + q - 1) / ((p + q - 1) * f - p), 1 / (f - 1)]
        else:
            mu1_range = [1 / (f - 1) - slope_clip, 1 / (f - 1)]
        if q - (p + q - 1) * (1 - f) > 0:
            mu2_range = [1 / f, (p + q - 1) / (q - (p + q - 1) * (1 - f))]
        else:
            mu2_range = [1 / f, 1 / f + slope_clip]
        return mu1_range, mu2_range

    def phi_psi_functions(self, mu1, mu2):
        f, p, q = self.f, self.p, self.q
        def phi1(r, i):
            return mu1 * r + i * (1 - f * mu1) / p
        def phi2(r, i):
            return mu2 * r + (i - 1) * (1 - f * mu2) / (p - 1)
        def psi1(r, j):
            return mu1 * (r - 1) + (j - 1) * (1 + mu1 * (1 - f)) / (q - 1)
        def psi2(r, j):
            return mu2 * (r - 1) + j * (1 + mu2 * (1 - f)) / q
        return phi1, phi2, psi1, psi2

    def pi_function(self, r, phi1, phi2, psi1, psi2):
        term1 = np.array([np.minimum(phi1(r, i), phi2(r, i)) for i in range(1, self.p + 1)])
        term2 = np.array([np.minimum(psi1(r, j), psi2(r, j)) for j in range(1, self.q)])
        return np.max(np.concatenate((term1, term2)))

    def pi(self, s1, s2):
        mu1_range, mu2_range = self.compute_ranges()
        mu1 = mu1_range[1] - s1 * (mu1_range[1] - mu1_range[0])
        mu2 = mu2_range[0] + s2 * (mu2_range[1] - mu2_range[0])
        phi1, phi2, psi1, psi2 = self.phi_psi_functions(mu1, mu2)
        pi_values = [self.pi_function(r, phi1, phi2, psi1, psi2).item() for r in self.r]
        return np.array(pi_values)

def gmi(r, f):
    r = np.array(r) - np.floor(r)
    f = np.array(f) - np.floor(f)
    z = np.where(r < f, r / f, (1 - r) / (1 - f))
    return z

def pik(r, f, mu):
    floor_r = np.floor(r)
    indicator = (r >= f + floor_r).astype(int)
    bar_r = r - floor_r - indicator
    p = np.sum(mu * bar_r)
    q = np.sum(mu * f)
    s = bar_r / (f - 1)
    a = np.max(s)
    i_star = np.argmax(s)
    b = np.max(np.delete(s, i_star))
    lambda_star = (bar_r[i_star] * q - (f[i_star] - 1) * p) / (mu[i_star] * (f[i_star] - 1) - q)
    term1 = np.max([(p + mu[i_star] * np.ceil(lambda_star))/q, (bar_r[i_star] + np.ceil(lambda_star)) / (f[i_star] - 1), b])
    term2 = np.max([(p + mu[i_star] * np.floor(lambda_star))/q, (bar_r[i_star] + np.floor(lambda_star)) / (f[i_star] - 1), b])
    term3 = np.max([p / q, a])
    return np.min([term1, term2, term3])

def cgfk(r, f, mu):
    m, n = r.shape
    f = np.array(f) - np.floor(f)
    if np.allclose(f, 0):
        print("Warning: f should not be all zeros.")
        return 
    elif m != len(f) or m != len(mu):
        print("Warning: f, mu, and r should have the same length.", f.shape, mu.shape, r.shape)
        return
    else:
        pi = np.zeros(n)
        for j in range(n):
            if np.allclose(r[:, j], np.round(r[:, j])):
                pass
            else:
                pi[j] = pik(r[:, j], f, mu)
        return pi



class Cut_size:
    def __init__(self, A, c, b, is_standard=False):
        self.A = A
        self.c = c
        self.b = b
        self.is_standard = is_standard
        self.ip = IP(A, c, b, is_standard=is_standard)
        self.A_bar, self.b_bar = self.ip.get_candidate_rows()
    def reset(self):
        self.ip = IP(self.A, self.c, self.b, is_standard=self.is_standard)

    def get_gmi_treesize(self, index=0):
        self.gmi_cut = gmi(self.A_bar[index], self.b_bar[index])
        self.ip.add_cut(self.gmi_cut)
        self.ip.optimize()
        self.gmi_treesize = self.ip.treesize
        self.reset()
        return self.gmi_treesize
    
    def get_cgfk_treesize(self, mu=np.ones(3)/3, k=3, index=0):
        if self.A_bar.shape[0] < k:
            k = self.A_bar.shape[0]
            mu = mu[:k]/np.sum(mu[:k])
        self.cgfk_cut = cgfk(self.A_bar[index:k+index], self.b_bar[index:k+index], mu)
        self.ip.add_cut(self.cgfk_cut)
        self.ip.optimize()
        self.cgfk_treesize = self.ip.treesize
        self.reset()
        return self.cgfk_treesize
        
    def get_cgf1_treesize(self, s=[0.0,0.0], p=2, q=2, slope_clip=30, index=0):
        cgf = CGF1(self.A_bar[index], self.b_bar[index], p, q, slope_clip)
        self.cgf_cut = cgf.pi(s[0], s[1])
        self.ip.add_cut(self.cgf_cut)
        self.ip.optimize()
        self.cgf1_treesize = self.ip.treesize
        self.reset()
        return self.cgf1_treesize