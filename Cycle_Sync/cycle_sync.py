import time
import numpy as np
from utils.params import parse_params
from utils.indexing import build_edge_index_matrices
from utils.triangles import sample_triangles, list_triangles
from utils.aab import compute_naive_aab
from utils.reweight import reweight_aab
from utils.irls import run_irls
from object.constructor import CycleSyncOutput

def Cycle_Sync(AdjMat, tijMat, params):
    t_start = time.time()

    params = parse_params(dict(params))
    rng = np.random.default_rng(params['seed'])

    Ind_i, Ind_j, tijMat3d, IndMat = build_edge_index_matrices(AdjMat, tijMat)

    CoIndMat, IndPos = sample_triangles(AdjMat, tijMat3d, Ind_i, Ind_j, params, rng)

    IRAABVec, SAABMat0 = compute_naive_aab(tijMat, tijMat3d, CoIndMat, Ind_i, Ind_j, IndPos)

    IRAABVec = reweight_aab(IRAABVec, SAABMat0, CoIndMat, Ind_i, Ind_j, IndMat, IndPos, params)

    TriEdge, TriK = list_triangles(AdjMat, Ind_i, Ind_j, params, rng)

    t_est, alph, out_extra = run_irls(AdjMat, tijMat, IRAABVec, Ind_i, Ind_j, tijMat3d, IndMat, TriEdge, TriK, params)

    out = CycleSyncOutput(
        t_est=t_est,
        alph=alph,
        TotalTime=time.time() - t_start,
        IRAABVec=IRAABVec,
        **out_extra
    )

    return t_est, out
