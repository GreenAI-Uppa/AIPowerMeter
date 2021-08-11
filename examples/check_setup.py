from deep_learning_power_measure.power_measure import experiment, parsers

driver = parsers.JsonParser('/home/paul/data/power_measure')

# will call the init and check if rapl or nvidia are available
exp = experiment.Experiment(driver)
