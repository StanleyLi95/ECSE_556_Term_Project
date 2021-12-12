import torch
import numpy as np


def cov(m, rowvar=False):
    '''Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
   
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()


if __name__ == "__main__":
    r = False
    x = np.random.randn(30, 2)
    xt = torch.from_numpy(x).type(torch.double)
    np_c = np.cov(x, rowvar=r)
    our_c = cov(xt, rowvar=r).numpy()
    print(np.allclose(np_c, our_c))
    print(x, '\n\n\n', our_c, '\n\n\n', torch.var(xt))
