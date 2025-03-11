# Standard modules
import sys
import os
import random
import math

# Third-party modules
import torch

# Local modules
from helpers import get_mutants
from visuals import plot_embeddings

def find_latent_distance_to_stable_point(sequences, model, tokeniser, device, embedding_layer):
    
    for sequence in sequences:
        
        local_stable_point = find_stable_point(sequence, model, tokeniser, device, embedding_layer)
        latent_delta = embedding - local_stable_point

    return latent_delta

def find_stable_point(sequence, model, tokeniser, device, embedding_layer):
    
    embedding_path = metropolis_hastings_search(sequence, model, tokeniser, device, embedding_layer)
    sampled_embeddings = sample_path(embedding_path)
    stable_point = mean_stable_point(sampled_embeddings)
    
    return stable_point

def metropolis_hastings_search(input_sequence, model, tokeniser, device, embedding_layer):
    
    metropolis_hastings_search_length = 1000
    residues = ["G", "A", "V", "L", "I", "T", "S", "M", "C", "P", "F", "Y", "W", "H", "K", "R", "D", "E", "N", "Q"]
    vocab = tokeniser.get_vocab()
    vocab = {key: vocab[key] for key in residues}
    search_breadth = 100
    
    searched_sequences, searched_embeddings, searched_heuristics = [], [], []
    previous_heuristic = math.inf
    
    for search in range(metropolis_hastings_search_length):
        
        print("Searching: ", "[", search, "/", metropolis_hastings_search_length, "]", input_sequence)
        
        neighbour_sequences = get_mutants(input_sequence, vocab, search_breadth)
        input_embedding = fetch_embedding(input_sequence, model, tokeniser, device, embedding_layer)
        neighbour_embeddings = [fetch_embedding(sequence, model, tokeniser, device, embedding_layer) for sequence in neighbour_sequences]
        input_heuristic = find_distance_to_centroid(input_embedding, neighbour_embeddings)
        searched_sequences.append(input_sequence)
        searched_embeddings.append(input_embedding)
        
        if get_metropolis_hastings_acceptance_ratio(previous_heuristic, input_heuristic):

            previous_sequence = input_sequence
            previous_heuristic = input_heuristic
            input_sequence = random.choice(neighbour_sequences)
    
    return searched_sequences, searched_embeddings, searched_heuristics

def fetch_embedding(sequence, model, tokeniser, device, embedding_layer):
    
    inputs = tokeniser([sequence], padding = True, truncation = True, return_tensors = "pt", max_length = 1024)
    inputs = inputs["input_ids"].to(device)
    model = model.to(device)
    output = model(inputs, output_hidden_states = True)
    batch_embeddings = output.hidden_states[embedding_layer]
    
    return batch_embeddings
    
def get_metropolis_hastings_acceptance_ratio(current_density_metric, new_density_metric) -> bool:

    acceptance_ratio = min(1, math.exp(current_density_metric - new_density_metric))
    acceptance = random.random() < acceptance_ratio

    return acceptance

def sample_path(embeddings):
    
    num_drop_head = 100
    num_sample_period =  1
    
    assert len(embeddings) > num_drop_head
    sampled_embeddings = embeddings[num_drop_head::num_sample_period]
    
    return sampled_embeddings

def mean_stable_point(sampled_embeddings):
    
    mean_embedding = sampled_embeddings.mean(dim = 1).float().cpu()
    
    return mean_embedding

def spatial_interpolation(sampled_embeddings):
    
    pca = PCA(n_components = 2)
    projected_embeddings = pca.fit_transform(sampled_embeddings.flat())
    kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer = 10)

def find_distance_to_centroid(input_embedding, mutant_embeddings):

    embeddings = [input_embedding] + mutant_embeddings
    embeddings = [embedding.squeeze(0).mean(dim = 0) for embedding in embeddings]
    tensorised_embeddings = torch.stack(embeddings)
    latent_mean = torch.mean(tensorised_embeddings, dim = 0)    
    distance_to_mean = math.dist(tensorised_embeddings[0], latent_mean)

    return distance_to_mean