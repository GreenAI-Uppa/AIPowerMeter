from display_results import driver

driver = driver.Driver()
print('model')
print(driver.get_models())
metrics = driver.metrics()
print("metrics")
print(metrics)
print()
m1 = "nvidia_draw_absolute"
print("getting the curve for the metric",m1)
metric1 = driver.get_curve(m1)
print(metric1)
m2 = "test_accuracy"
print("getting the curve for the metric",m2)
metric2 = driver.get_curve(m2)
print(metric2)

print()
metric1, metric2 = driver.interpolate(metric1, metric2)
print('interpolated metric',m1)
print(metric1)
print('interpolated metric',m2)
print(metric2)
