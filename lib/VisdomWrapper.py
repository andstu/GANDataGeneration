import numpy as np
from visdom import Visdom
from lib.DataCreationWrapper import *

# Visdom Stuff
class VisdomController():
    def __init__(self):
        self.vis = Visdom()
        self.plots = {}
        
    def ClearPlots(self):
        self.plots = {}
        
    def IsConnected(self):
        return self.vis.check_connection()
        
    def CreateLinePlot(self, x, y, title, xlabel, ylabel, win, key, env="main"):
            self.plots[win] = self.vis.line(X=np.array([x,x]), Y=np.array([y,y]), env = env, opts=dict(
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                win=win,
                legend=[key],
                showlegend=True
            ))
            
    def CreateScatterPlot(self, data, title, xlabel, ylabel, win, env="main"):
        self.plots[win] = self.vis.scatter(data, env = env, opts=dict(
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                win=win
            ))
    
    def UpdateLinePlot(self, x, y, win, key, env="main"):
        self.vis.line(np.array([y]), X=np.array([x]), env = env, win=self.plots[win], name=key, update="append") 
        
    def UpdateScatterPlot(self, data, win, env="main"):
        self.vis.scatter(data, env = env, win=self.plots[win], update="replace")

    # Custom Plots    
        
    def PlotLoss(self, key, epoch, loss):
        if not self.IsConnected():
            return

        plot_win = "loss_window"
        if plot_win not in self.plots:
            self.CreateLinePlot(epoch, loss, "Loss Graph", "Epoch", "Loss", plot_win, key)
        else:
            self.UpdateLinePlot(epoch, loss, plot_win, key)
            
    def PlotFakeFeatureDistributionComparison(self, f_idx_0, f_idx_1, gen_nn, batch_size, noise_function):
        if not self.IsConnected():
            return
        
        fake_data_0 = synthesize_data(gen_nn, batch_size, noise_function).detach().cpu().numpy()[:,f_idx_0]
        fake_data_1 = synthesize_data(gen_nn, batch_size, noise_function).detach().cpu().numpy()[:,f_idx_1]
        data = np.array([fake_data_0,fake_data_1]).T
        
        plot_win = str(f_idx_0 + f_idx_1) + "_fake_comp_window"
        title = "fake features : " + str(f_idx_0) + " vs " + str(f_idx_1)
        env = "feature_comparison"
        
        if plot_win not in self.plots:
            self.CreateScatterPlot(data, title, str(f_idx_0), str(f_idx_1), plot_win, env)
        else:
            self.UpdateScatterPlot(data, plot_win, env)
            
    def PlotRealFeatureDistributionComparison(self, f_idx_0, f_idx_1, real_data, num_samples):
        if not self.IsConnected():
            return
        
        rows = np.random.choice(np.arange(0,real_data.shape[0]), size=num_samples, replace=False)
        data = real_data.detach().cpu().numpy()[:,[f_idx_0, f_idx_1]][rows,:]
        
        plot_win = str(f_idx_0 + f_idx_1) + "_real_comp_window"
        title = "real features : " + str(f_idx_0) + " vs " + str(f_idx_1)
        env = "feature_comparison"
        
        if plot_win not in self.plots:
            self.CreateScatterPlot(data, title, str(f_idx_0), str(f_idx_1), plot_win, env)
        else:
            self.UpdateScatterPlot(data, plot_win, env)
        