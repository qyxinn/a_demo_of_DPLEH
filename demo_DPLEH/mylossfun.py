
import math
import torch
# out, time, delta=out, time_train, delta_train
def l2loss(out, time, delta):
    n = len(delta)            # 总人数，其中死亡=1；删失=0
    time = time.view(n,1)     # 将time表示成n*1的张量
    delta = delta.view(n,1)   # 将time表示成n*1的张量
    A = torch.log(time) - out # 返回e为底的指数：torch.log(torch.Tensor([7.39]))  tensor([2.0001]) exp(2)=7.39
    B = A*A*delta
    B = B.sum()
    return B

def eaftloss(out, time, delta):
    n = len(out[:, 0])  # 任意行第0列
    # h = 1.30*0.76 * math.pow(n,-0.2) ## or 1.59*n^(-1/3)
    h = 1.30* math.pow(n, -0.2) #=1.59*0.82

    time = time.view(n, 1)
    delta = delta.view(n, 1)
    g1 = out[:, 0].view(n, 1)  #h1
    g2 = out[:, 1].view(n, 1)  #h2
    g3 = out[:, 2].view(n, 1)  #beta*z

    # R = g(Xi) + log(Oi)
    R = torch.add(g1, torch.log(time))

    S1 = (delta * (g2+g3)).sum() / n
    S2 = -(delta * R).sum() / n

    # Rj - Ri
    rawones = torch.ones(1, n)
    R1 = torch.mm(R, rawones)
    R2 = torch.mm(torch.t(rawones), torch.t(R))
    DR = R1 - R2

    # K[(Rj-Ri)/h]
    K = normal_density(DR / h)
    Del = torch.mm(delta, rawones)
    DelK = Del * K

    # (1/nh) *sum_j Deltaj * K[(Rj-Ri)/h]
    Dk = torch.sum(DelK, dim=0) / (n * h)  ## Dk would be zero as learning rate too large!

    # log {(1/nh) * Deltaj * K[(Rj-Ri)/h]}
    log_Dk = torch.log(Dk)

    S3 = (torch.t(delta) * log_Dk).sum() / n

    # Phi((Rj-Ri)/h)
    ncdf = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0])).cdf
    # 在计算之前检查 DR 是否包含 NaN 或 Inf 值
    if torch.isnan(DR).any() or torch.isinf(DR).any():
        default_value = 0.0
        DR = torch.nan_to_num(DR, nan=default_value, posinf=default_value, neginf=default_value)
    P = ncdf(DR / h)
    L = torch.exp(g2 + g3 - g1)
    LL = torch.mm(L, rawones)
    LP_sum = torch.sum(LL * P, dim=0) / n
    Q = torch.log(LP_sum)

    S4 = -(delta * Q.view(n, 1)).sum() / n

    S = S1 + S2 + S3 + S4
    S = -S

    return S

# eaftloss_new(h0_train, outy, outz, delta_train, time_train, z_train)
# h0_values = h0_train;f2_values = outy;beta=outz;event = delta_train;time=time_train;Z=z_train
# def eaftloss_new(h0_values, f2_values, beta, event, time, Z):
#     loss = 0
#     for i in range(len(event)):
#         t_i = time[i]
#         event_i = event[i]
#         Z_i = Z[i]
#         f2_i = f2_values[i]
#         # Calculate lambda0(t_i * exp(f1(X_i)))
#         lambda0_i = h0_values[i]
#
#         # Calculate the risk term: lambda0(t_i * exp(f1(X_i))) * exp(β^T * Z_i + f2(Y_i))
#         risk_i = torch.log(lambda0_i)+torch.dot(Z_i, beta[i]) + f2_i
#         denominator_sum = 0
#
#         # Calculate the sum in the denominator for R(ti)
#         if event_i == 1:
#             for j in range(len(event)):
#                 if time[j] >= t_i and event[j] == 0:
#                     Z_j = Z[j]
#                     f2_j = f2_values[j]
#                     lambda0_j = h0_values[j]
#                     denominator_sum += lambda0_j * torch.exp(torch.dot(Z_j, beta[j]) + f2_j)
#             log_likelihood_i = risk_i- torch.log(torch.tensor(denominator_sum))
#         else:
#             log_likelihood_i = 0
#         loss -= log_likelihood_i
#     loss = loss/len(event)
#
#     return loss

def eaftloss_new(h0_values, f2_values, beta, event,time, Z):
    loss = 0
    for i in range(len(time)):
        t_i = time[i]

        Z_i = Z[i]
        f2_i = f2_values[i]
        # Calculate lambda0(t_i * exp(f1(X_i)))
        lambda0_i = h0_values[i]

        # Calculate the risk term: lambda0(t_i * exp(f1(X_i))) * exp(β^T * Z_i + f2(Y_i))
        risk_i = torch.log(lambda0_i)+torch.dot(Z_i, beta[i]) + f2_i
        denominator_sum = 0

        # Calculate the sum in the denominator for R(ti)
        for j in range(len(time)):
            if time[j] >= t_i and event[j] == 0:
                Z_j = Z[j]
                f2_j = f2_values[j]
                lambda0_j = h0_values[j]
                denominator_sum += lambda0_j * torch.exp(torch.dot(Z_j, beta[j]) + f2_j)
        log_likelihood_i = risk_i - torch.log(torch.tensor(denominator_sum))

        loss -= log_likelihood_i
    loss = loss/len(event)

    return loss

def normal_density(a):
    b = 0.3989423*torch.exp(-0.5*torch.pow(a,2.0))
    return b

def coxloss(out, time, delta):
    n = len(delta)
    time = time.view(n,1)
    delta = delta.view(n,1)
    rawones = torch.ones(1,n)
    
    T1 = torch.mm(time,rawones)
    T2 = torch.t(T1)
    Risk = torch.as_tensor((T2 <= T1),dtype=T1.dtype) # 对每一个位置进行0 or 1的判断 输出一个n*n的0 1矩阵 转换为torch tensor
    
    e = torch.sum(torch.mm(torch.exp(out),rawones)*Risk,dim=0)
    v = torch.log(e)
    
    S = -((out-v.view(n,1))*delta).sum()/n
    
    return S








































