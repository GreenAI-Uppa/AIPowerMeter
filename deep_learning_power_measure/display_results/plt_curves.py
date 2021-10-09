import matplotlib.pyplot as plt
import numpy as np
from deep_learning_power_measure.power_measure import experiment

def plot_m1_vs_m2(exps, metric_name1, metric_name2, convert_m1=None, convert_m2=None):
    for e_name, e in exps.items():
        m1 = e.get_curve(metric_name1)
        m2 = e.get_curve(metric_name2)
        m1, m2 = experiment.interpolate(m1, m2)
        if convert_m1 is not None:
            y1 = convert_m1(m1)
        else:
            y1 = np.array([p['value'] for p in m1])
        if convert_m2 is not None:
            y2 = convert_m2(m2)
        else:
            y2 = np.array([p['value'] for p in m2])
        plt.plot(y1,y2, label=e_name)
    plt.xlabel(metric_name1)
    plt.ylabel(metric_name2)
    plt.legend()
    #plt.show()
