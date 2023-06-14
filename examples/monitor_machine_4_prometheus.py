from deep_learning_power_measure.power_measure import experiment, prometheus_client

driver = prometheus_client.PrometheusClient()
exp = experiment.Experiment(driver)
exp.monitor_machine(period=5)