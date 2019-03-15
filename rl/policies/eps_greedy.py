class EpsGreedy(object):
    def __init__(self, eps_decay=.1, eps_init=1., eps_min=0.02):
        self.eps_init = eps_init
        self.eps_min = eps_min
        self.eps_decay = eps_decay

    def value(self, t):
        # fraction  = min(float(t) / self.schedule_timesteps, 1.0)
        return self.eps_init + t * self.eps_decay * (self.eps_min - self.eps_init)