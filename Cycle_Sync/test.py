from cycle_sync import Cycle_Sync
from test_case import find_adjacency, align_procrustes, t_true

params = {}
AdjMat, tijMat = find_adjacency()
t_est, out = Cycle_Sync(AdjMat, tijMat, params)

print("Estimated camera locations:")
print(t_est, "\n")

print("Alpha (Edge Scales):")
print(out.alph, "\n")

print("Total time taken:")
print(out.TotalTime, "seconds \n")

X_aligned, R, scale, translation, err = align_procrustes(t_est, t_true)

print("Rotation Matrix:")
print(R, "\n")

print("Scale:")
print(scale, "\n")

print("Translation Vector:")
print(translation, "\n")

print("Aligned estimate:")
print(X_aligned, "\n")

print("RMS alignment error:", err)