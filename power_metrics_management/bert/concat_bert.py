import sys
sys.path.append('../')
from concat_power_measure import main

main_folder = '/data/mfrancois/measure'
n_iterations = {
    'input_100': 9000,
    'input_200': 8000,
    'input_300': 7000,
    'input_400': 6000,
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