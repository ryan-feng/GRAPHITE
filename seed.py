SEED = None
if SEED is None:
    SEED = 0
    from warnings import warn

    import torch

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import numpy as np

    np.random.seed(SEED)
    import random

    random.seed(SEED)
