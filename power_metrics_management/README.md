# Power Metric Management 

Make use of *concat_power_measure.py* to merge your multiple power_metrics.json.

## Requirements 

Following architecture is required:

```
__input_format
  |__run_x
     |__power_metrics.json
     |__latency.json
```

Where *latency.json* countains an array with all latencies you need.

## usage

Four arguments have to be specified when you run the main function.
- **output**: can be "csv", "full" or "cube"
  - csv: will write your data into a csv file with input format as row and measure as columns.json
  - cube: will write your data into a json file with a summary done on each power_metrics.json. A mean based on median.
  - full: will write your data into a json file. Everything will be in, no loss.
- **main_folder**: path where the previous required architecture is, 
- **n_iterations**: number iteration inside one power_metrics.json. If various number of iterations have been used then use this format: {'folder': number_of_iteration}
- **file_to_write**: output file where to write your data