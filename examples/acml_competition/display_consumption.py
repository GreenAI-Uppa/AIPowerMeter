from deep_learning_power_measure.power_measure import experiment, parsers

# this parser will be in charge to write the results to a json file
output_folder = "measure_power"
driver = parsers.JsonParser(output_folder)
exp_result = experiment.ExpResults(driver)

exp_result.print()

