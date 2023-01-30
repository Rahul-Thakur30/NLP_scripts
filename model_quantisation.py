from torch.quantization import quantize_dynamic
from torch.nn import Embedding, Linear
import os
import torch
from scipy.spatial import distance
from sentence_transformers import SentenceTransformer


def get_cosine_similarity_qmodel(sen_1,sen_2):
    en_1 = q_model.encode(sen_1)
    en_2 = q_model.encode(sen_2)
    return 1-distance.cosine(en_1,en_2)


def print_size_of_model(model):
    torch.save(model.state_dict(),"temp.p")
    print("Size (MB):",os.path.getsize("temp.p")/1e6)
    model_size = os.path.getsize("temp.p")/1e6
    os.remove("temp.p")
    return model_size

if __name__=="__main__":
    models_path = 'C:/Users/rahul/Desktop/Unremot/Assisto/assisto_intent_updated/online_trained_model/'
    models = os.listdir(models_path)
    for model_name in models:
        print("Original Model Name: {}".format(model_name))
        if model_name!='create_model.py':
            original_model = SentenceTransformer("{}{}/".format(models_path,model_name))
            q_model = quantize_dynamic(original_model, {Linear})
            torch.save(q_model,"quantised_{}.pt".format(model_name))
            print("Size of original Model")
            og_model_size = print_size_of_model(original_model)
            print("Size of quantised Model")
            q_model_size = print_size_of_model(q_model)
            print("Percentage of Reduction ={} ".format((og_model_size-q_model_size)/og_model_size*100 ))
            print("*******************************************************")

