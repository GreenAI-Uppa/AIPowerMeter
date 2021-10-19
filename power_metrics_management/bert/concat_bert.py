import sys
sys.path.append('../')
from concat_power_measure import main

# {'details': '',
#  'inputs': [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
#  'iterations': [9500, 9000, 8500, 8000, 7500, 7000, 6500, 6000, 5500, 5000],
#  'model': 'bert'}

main_folder = '/data/mfrancois/measure'
n_iterations = {
    'input_50': 9500,
    'input_100': 9000,
    'input_150': 8500,
    'input_200': 8000,
    'input_250': 7500,
    'input_300': 7000,
    'input_350': 6500,
    'input_400': 6000,
    'input_450': 5500,
    'input_500': 5000,
}

file_to_write = '/data/mfrancois/measure/res.csv'

# apply concatenation and write it in csv file 
main(
    output='csv',
    main_folder=main_folder,
    n_iterations=n_iterations,
    file_to_write=file_to_write
)
