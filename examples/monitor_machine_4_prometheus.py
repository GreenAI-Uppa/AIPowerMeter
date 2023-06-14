from deep_learning_power_measure.power_measure import experiment, prometheus_client

driver = prometheus_client.PrometheusClient()
exp = experiment.Experiment(driver)
print('launching')
exp.monitor_machine()