from bert import main
import json

iterations = []
inputs = []

for i in range(10):
    i += 1
    print(f'Running on i: {i*50}')
    main(n_input=i*50)
    iterations.append(int(10000-(i*50*10)))
    inputs.append(i*50)

meta = {
    'iterations': iterations,
    'inputs': inputs,
    'model': 'bert',
    'details': '',  
}
f = open('/data/mfrancois/measure/meta.json', 'w')
json.dump(meta, f)
f.close()

print()
print('Done.')