from bert import main
import json

iterations = []
inputs = []

for i in range(5):
    print(f'Running on i: {i*100}')
    main(n_input=i*100)
    iterations.append(int(10000-(i*100*10)))
    inputs.append(i*100)

meta = {
    'iterations': iterations,
    'inputs': inputs,
    'model': 'bert',
    'details': '',  
}
f = open('/data/mfrancois/measure/meta.json', 'w')
json.dump()
f.close()

print()
print('Done.')