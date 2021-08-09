
from parse_results import Experiment
#e = Experiment.Experiment('/home/paul/data/uppa/edouard/grad_1_layer/grad_1_layer_0.001_0_power/','/home/paul/data/uppa/edouard/grad_1_layer/grad_1_layer_0.001_0.json')
power_folder = "/home/paul/programmation/uppa/edouard/power/linear/results_0.001_3_power/"
acc_file = "/home/paul/programmation/uppa/edouard/power/linear/results_0.001_3.json"
e_lin = Experiment.Experiment(power_folder,acc_file)

power_folder = "/home/paul/programmation/uppa/edouard/power/results_0.01_0_power/"
acc_file = "/home/paul/programmation/uppa/edouard/power/results_0.01_0.json"
e_1h = Experiment.Experiment(power_folder,acc_file)

power_folder = "/home/paul/programmation/uppa/binary_connect_network/results_0.009_power/"
acc_file = "/home/paul/programmation/uppa/binary_connect_network/results_0.009.json"
e_bc = Experiment.Experiment(power_folder,acc_file)

power_folder = "/home/paul/programmation/uppa/binary_connect_network/binary_network_0.009_power/"
acc_file = "/home/paul/programmation/uppa/binary_connect_network/binary_network_0.009.json"
e_bn = Experiment.Experiment(power_folder,acc_file)


def convert_m2(m):
    return e_lin.wtowh(e_lin.integrate(m))

from display_results import plt_curves
import matplotlib.pyplot as plt
#plt_curves.plot_m1_vs_m2(e_lin, 'test_accuracy', 'intel_power')
#plt_curves.plot_m1_vs_m2(e_1h, 'test_accuracy', 'intel_power')
plt_curves.plot_m1_vs_m2({"1 hidden layer real": e_1h, 'linear real': e_lin, "1 hidden layer binary network": e_bn, 'linear binary connect': e_bc }, 'test_accuracy', 'intel_power', convert_m2=convert_m2)
plt.show()
