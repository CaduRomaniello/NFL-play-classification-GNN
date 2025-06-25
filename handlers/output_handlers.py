import json
import os

import pandas as pd

def json2csv(data, timestamp, n):
    """
    Convert JSON data to CSV format.
    
    Args:
        json_data (list): List of dictionaries representing JSON data.
        
    Returns:
        str: CSV formatted string.
    """

    best_gcn_results = data.get('best_gcn_results', {})
    last_gcn_results = data.get('last_gcn_results', {})
    rf_results = data.get('rf_results', {})
    mlp_results = data.get('rf_results', {})

    best_gcn_dict = {
        "model_name": "GCN",
        "rush_precision": best_gcn_results['Rush']['precision'],
        "rush_recall": best_gcn_results['Rush']['recall'],
        "rush_f1_score": best_gcn_results['Rush']['f1-score'],
        "rush_support": best_gcn_results['Rush']['support'],

        "pass_precision": best_gcn_results['Pass']['precision'],
        "pass_recall": best_gcn_results['Pass']['recall'],
        "pass_f1_score": best_gcn_results['Pass']['f1-score'],
        "pass_support": best_gcn_results['Pass']['support'],

        "accuracy": best_gcn_results['accuracy'],

        "macro_precision": best_gcn_results['macro avg']['precision'],
        "macro_recall": best_gcn_results['macro avg']['recall'],
        "macro_f1_score": best_gcn_results['macro avg']['f1-score'],
        "macro_support": best_gcn_results['macro avg']['support'],

        "weighted_precision": best_gcn_results['weighted avg']['precision'],
        "weighted_recall": best_gcn_results['weighted avg']['recall'],
        "weighted_f1_score": best_gcn_results['weighted avg']['f1-score'],
        "weighted_support": best_gcn_results['weighted avg']['support'],

        "config": data.get('config', {})
    }

    last_gcn_dict = {
        "model_name": "GCN",
        "rush_precision": last_gcn_results['Rush']['precision'],
        "rush_recall": last_gcn_results['Rush']['recall'],
        "rush_f1_score": last_gcn_results['Rush']['f1-score'],
        "rush_support": last_gcn_results['Rush']['support'],

        "pass_precision": last_gcn_results['Pass']['precision'],
        "pass_recall": last_gcn_results['Pass']['recall'],
        "pass_f1_score": last_gcn_results['Pass']['f1-score'],
        "pass_support": last_gcn_results['Pass']['support'],

        "accuracy": last_gcn_results['accuracy'],

        "macro_precision": last_gcn_results['macro avg']['precision'],
        "macro_recall": last_gcn_results['macro avg']['recall'],
        "macro_f1_score": last_gcn_results['macro avg']['f1-score'],
        "macro_support": last_gcn_results['macro avg']['support'],

        "weighted_precision": last_gcn_results['weighted avg']['precision'],
        "weighted_recall": last_gcn_results['weighted avg']['recall'],
        "weighted_f1_score": last_gcn_results['weighted avg']['f1-score'],
        "weighted_support": last_gcn_results['weighted avg']['support'],

        "config": data.get('config', {})
    }

    rf_dict = {
        "model_name": "GCN",
        "rush_precision": rf_results['Rush']['precision'],
        "rush_recall": rf_results['Rush']['recall'],
        "rush_f1_score": rf_results['Rush']['f1-score'],
        "rush_support": rf_results['Rush']['support'],

        "pass_precision": rf_results['Pass']['precision'],
        "pass_recall": rf_results['Pass']['recall'],
        "pass_f1_score": rf_results['Pass']['f1-score'],
        "pass_support": rf_results['Pass']['support'],

        "accuracy": rf_results['accuracy'],

        "macro_precision": rf_results['macro avg']['precision'],
        "macro_recall": rf_results['macro avg']['recall'],
        "macro_f1_score": rf_results['macro avg']['f1-score'],
        "macro_support": rf_results['macro avg']['support'],

        "weighted_precision": rf_results['weighted avg']['precision'],
        "weighted_recall": rf_results['weighted avg']['recall'],
        "weighted_f1_score": rf_results['weighted avg']['f1-score'],
        "weighted_support": rf_results['weighted avg']['support'],

        "config": data.get('config', {})
    }

    mlp_dict = {
        "model_name": "GCN",
        "rush_precision": mlp_results['Rush']['precision'],
        "rush_recall": mlp_results['Rush']['recall'],
        "rush_f1_score": mlp_results['Rush']['f1-score'],
        "rush_support": mlp_results['Rush']['support'],

        "pass_precision": mlp_results['Pass']['precision'],
        "pass_recall": mlp_results['Pass']['recall'],
        "pass_f1_score": mlp_results['Pass']['f1-score'],
        "pass_support": mlp_results['Pass']['support'],

        "accuracy": mlp_results['accuracy'],

        "macro_precision": mlp_results['macro avg']['precision'],
        "macro_recall": mlp_results['macro avg']['recall'],
        "macro_f1_score": mlp_results['macro avg']['f1-score'],
        "macro_support": mlp_results['macro avg']['support'],

        "weighted_precision": mlp_results['weighted avg']['precision'],
        "weighted_recall": mlp_results['weighted avg']['recall'],
        "weighted_f1_score": mlp_results['weighted avg']['f1-score'],
        "weighted_support": mlp_results['weighted avg']['support'],

        "config": data.get('config', {})
    }

    csv_data = [
        best_gcn_dict,
        last_gcn_dict,
        rf_dict,
        mlp_dict
    ]

    cur_path = os.getcwd()
    out_path = os.path.abspath(os.path.join(cur_path, f'eniac/n{n}/{timestamp}'))

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    df = pd.DataFrame(csv_data)
    df.to_csv(os.path.join(out_path, 'results.csv'))


# model_name,rush_precision,rush_recall,rush_f1_score,rush_support,pass_precision,pass_recall,pass_f1_score,
# pass_support,accuracy,macro_precision,macro_recall,macro_f1_score,weighted_precision,weighted_recall,weighted_f1_score,config

def save_data_to_json(data, timestamp, n):
    """
    Salva um dicionário Python em um arquivo JSON.
    
    Args:
        data (dict): O dicionário Python a ser salvo.
        file_path (str): O caminho completo para o arquivo JSON de saída.
    """
    # Cria o diretório se não existir
    cur_path = os.getcwd()
    out_path = os.path.abspath(os.path.join(cur_path, f'eniac/n{n}/{timestamp}'))

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    data['last_gcn_results']['confusion_matrix'] = data['last_gcn_results']['confusion_matrix'].tolist()
    data['best_gcn_results']['confusion_matrix'] = data['best_gcn_results']['confusion_matrix'].tolist()
    data['rf_results']['confusion_matrix'] = data['rf_results']['confusion_matrix'].tolist()
    data['mlp_results']['confusion_matrix'] = data['mlp_results']['confusion_matrix'].tolist()

    # Salva o dicionário como JSON
    with open(f'{out_path}/results.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)