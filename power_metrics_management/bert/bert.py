import torch, sys, time, os, time, json, argparse
sys.path.append('../..')
from deep_learning_power_measure.power_measure import experiment, parsers
from transformers import BertTokenizer, BertForSequenceClassification

# main params
device = 'cuda' if torch.cuda.is_available() else 'cpu'
root = '/data/mfrancois/measure'

def main(n_input: int):
    n_iterations = int(10000-(n_input*10))
    sentence = 'yes '
    parent_folder = f'input_{n_input}'
    print(f'device: {device}')
    print(f'number of iteration: {n_iterations:.0f}')

    # 10 itérations afin d'obtenir une médiane robuste
    for n in range(10):
        output_folder = f'run_{n+1}'
        print('*'*30)
        print(f'RUNNING ITERATION n: {n+1:.0f}')
        print('*'*30)

        # chargement de bert
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        model.to(device)
        inputs = tokenizer(sentence*n_input, return_tensors="pt") # Tokenization + format input 
        inputs = inputs.to(device)

        # création des dossiers d'output si besoin
        if not os.path.isdir(f'{root}/{parent_folder}'):
            os.mkdir(f'{root}/{parent_folder}')
        
        if not os.path.isdir(f'{root}/{parent_folder}/{output_folder}'):
            os.mkdir(f'{root}/{parent_folder}/{output_folder}')
        
        latency = []
        # mesure de consommation + itération sur l'inférence
        start_measuring(output_folder=f'{root}/{parent_folder}/{output_folder}')

        # model prediction
        for _ in range(n_iterations): # Attention : un range de float ne plante pas mais tourne indéfiniment
             
            # latency calc for each predict
            since = time.time()
            
            with torch.no_grad():
                _ = model(**inputs)
            
            latency.append(time.time() - since)


        end_measuring(output_folder=f'{root}/{parent_folder}/{output_folder}')

        f = open(f'{root}/{parent_folder}/{output_folder}/latency.json', 'w')
        json.dump(latency, f)
        f.close()

def start_measuring(output_folder: str):
    """start the consumption measure

    Args:
        output_folder (str): path where power_metrics.json will be save.
    """
    global parser, driver, q
    driver = parsers.JsonParser(output_folder)
    exp = experiment.Experiment(driver)
    p, q = exp.measure_yourself(period=2)
    print('*'*10)
    print('Power Meter running...')
    print('*'*10)

def end_measuring(output_folder: str):
    """end the consumption measure

    Args:
        output_folder (str): path where power_metrics.json will be save.
    """
    print('*'*10)
    print('Power Meter ending...')
    print('*'*10)
    q.put(experiment.STOP_MESSAGE)
    driver = parsers.JsonParser(output_folder)
    exp_result = experiment.ExpResults(driver)
    exp_result.print()


if __name__ == "__main__":    
    try:
        try:
            global args
            parser = argparse.ArgumentParser(
                description='Run convolution layer on random input and record the energy consumption'
                )
            parser.add_argument('--input',
                                help='input size',
                                default='100', type=str)
            args = parser.parse_args()
            # choix de la taille en input du modèle
            n_input = int(args.input)
            print("============================================================")
            print(f"===> WORKING ON: {n_input} inputs")
            print("============================================================")
            main(n_input=n_input)
        except ValueError:
            raise ValueError("not an int provided")
            
    except Exception as e:
        print(e)
    
