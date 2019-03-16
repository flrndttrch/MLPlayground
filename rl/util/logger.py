import matplotlib.pyplot as plt


class Logger:
    def __init__(self, print_freq=1):
        self.logger = dict(eps_return=[])
        self.print_freq = print_freq

    def print_return(self, num_eps, eps_return):
        self.logger["eps_return"].append(eps_return)

        if num_eps % self.print_freq == 0:
            print("Episode {}:\treturn={}".format(num_eps, eps_return))

    def plot_return(self, title):
        fig, ax = plt.subplots()
        ax.set(title=title, xlabel="Episodes", ylabel="Return")

        ax.plot(list(range(1, len(self.logger["eps_return"]) + 1)), self.logger["eps_return"])

        plt.show()

    @staticmethod
    def print_params(params):
        print()
        for key, param in params.items():
            print("{}={}".format(key, param))
