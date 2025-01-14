import math
import torch
import numpy
import numpy as np
import numba
from lifelines import KaplanMeierFitter
from myb_score import integrated_brier_score

#原代码
# out, time, delta, out_new, time_new, delta_new, Riemann_sum_gap, integral_id=out_train, time_train, delta_train, out_test, time_test, delta_test, Riemann_sum_gap, integral_id
def C_index(out, time, delta, out_new, time_new, delta_new, Riemann_sum_gap, integral_id):
    d = Riemann_sum_gap
    time_new, ind_sort = time_new.sort()
    delta_new = delta_new[ind_sort]
    out_new = out_new[ind_sort, :]
    T_min = min(time.numpy())
    T_max = max(time.numpy())
    ind_min = (time_new > T_min) * ((time_new < T_max))
    time_new = time_new[ind_min]
    delta_new = delta_new[ind_min]
    out_new = out_new[ind_min, :]
    n_new = len(out_new[:, 0])

    time_new = time_new.view(n_new, 1)
    delta_new = delta_new.view(n_new, 1)
    # ia, ib = out.size()

    expg1_new = torch.exp(out_new[:,0].view(n_new,1)) #h1
    expg2_new = torch.exp(out_new[:, 1].view(n_new, 1)) #h2
    expg3_new = torch.exp(out_new[:, 2].view(n_new, 1)) #betaz

    ####################################
    ind_si = torch.zeros(n_new, dtype=int)
    T0 = T_min
    a = (time_new[0].numpy() - T_min)  # (time_new[0].numpy()-T_min)/2

    if d < a:
        TT0 = numpy.arange(T_min, time_new[0].numpy(), min(a, d))
        n0 = len(TT0) - 1
        dt0 = min(a, d) * numpy.ones(len(TT0))
    else:
        TT0 = numpy.append(T_min, time_new[0])
        n0 = 1
        dt0 = numpy.append(T_min, a)

    for k in range(n_new):
        T1 = time_new[k].numpy()
        dd = (T1 - T0)

        if k == 0:
            ind_si[k] = n0
            T = TT0
            dt = dt0
        else:
            if dd == 0:
                ind_si[k] = ind_si[k - 1]
            else:
                if (dd <= d):
                    ind_si[k] = ind_si[k - 1] + 1
                    T = numpy.append(T, T1)
                    dt = numpy.append(dt, dd)
                else:
                    TT0 = numpy.arange(T0, T1, d)
                    dt0 = d * numpy.ones(len(TT0))
                    n0 = len(TT0) - 1
                    ind_si[k] = ind_si[k - 1] + n0
                    T = numpy.append(T, TT0[1:])
                    dt = numpy.append(dt, dt0[1:])
        T0 = T1

    m = len(T)
    T = torch.from_numpy(T).view(1, m)
    T = T.type_as(time)
    dt = torch.from_numpy(dt).view(1, m)
    dt = dt.type_as(time)

    if (integral_id == 0):  # (integral_id == 'gbsg')|(integral_id == 'metabric'):#基于黎曼和方式
        mat_si = torch.zeros(n_new, n_new)
        T_med = torch.zeros(n_new, 1)

        # for i in range(n_new):
        for i in numba.prange(n_new):
            if i == 0:
                A = (torch.exp(
                    -expg2_new[i, 0] * expg3_new[i,0] * torch.cumsum(dt * hazard_fun(out, time, delta, expg1_new[i, 0] * T), dim=1)))
                mat_si[i, :] = A[0, ind_si]
                # T_med[i, :] = max(T[A >= 0.5])
                T_med[i, :] = max(T[A >= 0.1])

            else:
                if ind_si[i] == ind_si[i - 1]:
                    A = (torch.exp(
                        -expg2_new[i, 0] * expg3_new[i,0] * torch.cumsum(dt * hazard_fun(out, time, delta, expg1_new[i, 0] * T), dim=1)))
                    mat_si[i, :] = mat_si[i - 1, :]
                    T_med[i, :] = max(T[A >= 0.1])
                else:
                    A = (torch.exp(
                        -expg2_new[i, 0]* expg3_new[i,0] * torch.cumsum(dt * hazard_fun(out, time, delta, expg1_new[i, 0] * T), dim=1)))
                    mat_si[i, :] = A[0, ind_si]
                    T_med[i, :] = max(T[A >= 0.1])

        Dtime = torch.log(T_med) - torch.log(time_new)
        ap_mean = torch.mean(abs(Dtime[delta_new == 1]) / abs(torch.log(time_new[delta_new == 1])))
        ap_std = torch.std(abs(Dtime[delta_new == 1]) / abs(torch.log(time_new[delta_new == 1])))
        ap_rate = torch.sum(Dtime[delta_new == 0] >= 0) / (n_new - sum(delta_new))

        ss = mat_si.detach().numpy()
        tt = time_new.numpy()
        dlt = delta_new.numpy()
        c_index_antolini = sum_concordant_antolini(ss, tt, dlt)

    else:#使用 Concordance-Discordance 计算 C-index
        count = 0.
        count_d = 0.
        # mat = torch.zeros(n_new,n_new)
        for i in numba.prange(n_new):
            ti = time_new[i].numpy()
            di = delta_new[i].numpy()
            IND_i = ind_si[i] + 1
            Sii = torch.exp(-(expg2_new[i, 0] * expg3_new[i,0] * dt[0, :IND_i] * (
                hazard_fun(out, time, delta, expg1_new[i, 0] * T[0, :IND_i]))).sum())
            # mat[i,i] = Sii
            Sii = Sii.detach().numpy()

            for j in range(n_new):
                if j != i:
                    Sji = torch.exp(-(expg2_new[j, 0] * expg3_new[i,0]* dt[0, :IND_i] * (
                        hazard_fun(out, time, delta, expg1_new[j, 0] * T[0, :IND_i]))).sum())
                    # mat[j,i] = Sji
                    Sji = Sji.detach().numpy()
                    tj = time_new[j].numpy()
                    dj = delta_new[j].numpy()

                    count += is_concordant_antolini(Sii, Sji, ti, tj, di, dj)
                    count_d += is_concordant_dominant_antolini(ti, tj, di, dj)

        c_index_antolini = count / count_d

    ###############################
    a1_ = delta.tolist()
    a2_ = time.tolist()

    b1_ = np.array(a1_)
    b2_ = np.array(a2_)

    c_ = np.transpose(np.vstack((b1_, b2_)))
    dt_ = [('delta', bool), ('time', np.float32)]
    d_ = np.array(list(map(tuple, c_)), dtype=dt_)

    e1_ = delta_new.view(len(delta_new), ).tolist()
    e2_ = time_new.view(len(delta_new), ).tolist()

    f1_ = np.array(e1_)
    f2_ = np.array(e2_)

    g_ = np.transpose(np.vstack((f1_, f2_)))
    h_ = np.array(list(map(tuple, g_)), dtype=dt_)

    mint = max(min(d_['time']), min(h_['time']))
    maxt = min(max(d_['time']), max(h_['time']))
    ts_ = np.arange(mint, maxt, (maxt - mint) / len(h_))
    ibs = integrated_brier_score(d_, h_, ss, ts_)
    #########################
    print(ibs)

    return c_index_antolini, ibs


def hazard_fun(out, time, delta, t):
    # t is a vector of input evaluated time
    n = len(out[:, 0])
    # h = 1.30*0.76 * math.pow(n, -0.2)  ## 1.59 * 0.82 * math.pow(n, -1/5)dui
    h = 1.30* math.pow(n, -0.2) #=1.59*0.82

    time = time.view(n, 1)
    delta = delta.view(n, 1)
    g1 = out[:, 0].view(n, 1)  #h1
    g2 = out[:, 1].view(n, 1)  # h2
    g3 = out[:, 2].view(n, 1)  # beta*z

    # R = g(Xi) + log(Oi)
    R = torch.add(g1, torch.log(time))

    # Rj - Ri
    p = max(len(t), len(torch.t(t)))
    T = t
    T = T.view(1, p)
    log_t = torch.log(T)

    # (1/nh) *sum_j Deltaj * K[(Rj-Ri)/h]
    Dk = torch.sum(torch.mm(delta, torch.ones([1, p], dtype=out.dtype)) \
                   * normal_density((torch.mm(R, torch.ones([1, p], dtype=out.dtype)) - torch.mm(
        torch.ones([n, 1], dtype=out.dtype), log_t)) / h) / h, dim=0) / n

    # Phi((Rj-Ri)/h)
    ncdf = torch.distributions.normal.Normal(torch.tensor([0.0], dtype=out.dtype),
                                             torch.tensor([1.0], dtype=out.dtype)).cdf
    L = torch.exp(g2 + g3 - g1)

    dominator = (torch.sum((torch.mm(L, torch.ones([1, p], dtype=out.dtype))) * \
                           ncdf((torch.mm(R, torch.ones([1, p], dtype=out.dtype)) - torch.mm(
                               torch.ones([n, 1], dtype=out.dtype), log_t)) / h), dim=0) / n) * T  # LP_sum*T
    lam = Dk / dominator
    return lam


### helper function
def normal_density(a):
    b = 0.3989 * torch.exp(-0.5 * torch.pow(a, 2.0))
    return b


def is_comparable_antolini(t_i, t_j, d_i, d_j):
    a = (t_i < t_j) * d_i
    b = (t_i == t_j) * d_i * (d_j == 0.)
    c = max(a, b)
    return c


def is_concordant_antolini(s_i, s_j, t_i, t_j, d_i, d_j):
    return (s_i < s_j) * is_comparable_antolini(t_i, t_j, d_i, d_j)


def is_concordant_dominant_antolini(t_i, t_j, d_i, d_j):
    conc = 0.
    if t_i <= t_j:
        conc = (t_i <= t_j) * 1.
    return conc * is_comparable_antolini(t_i, t_j, d_i, d_j)


def sum_concordant_antolini(s, t, d):
    n = len(t)
    count = 0.
    count_d = 0.
    for i in range(n):
        for j in range(n):
            if j != i:
                count += is_concordant_antolini(s[i, i], s[j, i], t[i], t[j], d[i], d[j])
                count_d += is_concordant_dominant_antolini(t[i], t[j], d[i], d[j])
    ratio = count / count_d
    return ratio


def cox_c_index(out_new, time_new, delta_new):
    n_new = len(out_new[:, 0])
    rawones = torch.ones(1, n_new)
    s = torch.mm(out_new, rawones)
    s = -s.detach().numpy()
    t = time_new.detach().numpy()
    d = delta_new.detach().numpy()

    return sum_concordant_antolini(s, t, d)

def dtget(out, time, delta, out_new, time_new, delta_new,dn):

    dt = (time_new/dn).view(len(time_new),1)
    T = torch.zeros(len(time_new)*dn)
    T = T.view(len(time_new),dn)

    for i in range(len(time_new)):
        t = np.arange(dt[i]/2,time_new[i],dt[i])
        T[i,:] = torch.tensor(t)

    return (dt,T)