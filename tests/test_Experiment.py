from deep_learning_power_measure.power_measure import experiment, parsers

driver = parsers.JsonParser('/home/paul/data/power_measure')
exp = experiment.Experiment(driver)
