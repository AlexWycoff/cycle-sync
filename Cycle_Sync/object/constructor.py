class CycleSyncOutput:
    def __init__(self, t_est, alph, TotalTime, **kwargs):
        self.t_est = t_est
        self.alph = alph
        self.TotalTime = TotalTime
        for k, v in kwargs.items():
            setattr(self, k, v)
