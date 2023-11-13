import sys
from deep_learning_power_measure.power_measure import experiment, parsers

output_folder = sys.argv[1]
driver = parsers.JsonParser(output_folder)
exp_result = experiment.ExpResults(driver)


d = exp_result.get_summary() #start = 1698660963.981172, end = 1698660986.022797)
print('WHOLE')
print(d)

#d = exp_result.get_summary(start=1691682321.594161,end=1691682335.915676)
#print('ON PERIOD')
#print(d)


