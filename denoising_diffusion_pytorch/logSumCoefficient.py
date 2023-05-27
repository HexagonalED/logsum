import torch
import torch.nn as nn


class logSumCoefficient(nn.Module):
    def __init__(self,n):
        super(logSumCoefficient, self).__init__()
        self.n=n
        self.model = torch.jit.load("m"+n+".pt")

    def __init__(self,path):
        super(logSumCoefficient, self).__init__()
        self.model = torch.jit.load(path)

    def predict(self,X_value):
        pred=self.model(X_value)
        log_individual = torch.sum(torch.log(X_value) * pred[:-1])
        log_constant = pred[-1]
        return log_individual+log_constant

    def approximate(self,X_list):
        print("approximiation start")
        print(X_list)
        l = X_list.size(dim=0)
        ret = list(l)
        for t in range(l):
            ts=X_list[t]
            ret.append(self.predict(ts))
        return torch.tensor(ret)

