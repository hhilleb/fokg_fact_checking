import transform
import embedding
import torch
import os

# ------- Transform input data to easier format ----------
data_train = transform.load_data("./fokg-sw-train-2023.nt")[:500].copy()
data_test = transform.load_data("./fokg-sw-test-2023.nt")

transform.write_data(data_train, "./train.txt")
transform.write_data(data_test, "./test.txt")


# ------- Train TransE embedding of knowledge graph ------
kg = embedding.KG(data_dir='./')

os.remove("./train.txt")
os.remove("./test.txt")

dataset_train = embedding.Dataset(data=kg.train_idx, num_entities=kg.num_entities)

hparam = {
    'embedding_dim': 50,
    'num_entities': kg.num_entities,
    'num_relations': kg.num_relations,
    'gamma': 1.0,  # margin for loss
    'lr': .01,  # learning rate for optimizer
    'batch_size': 256,
    'num_epochs': 1000
}

model = embedding.TransE(**hparam)
embedding.train(model, dataset_train, hparam)


# --- Compute truth values for every fact in test file ----
distance_sum = 0
for train_triple_idx in kg.train_idx:
    s_idx, p_idx, o_idx = train_triple_idx
    distance_sum += model.forward(torch.tensor([s_idx]), torch.tensor([p_idx]), torch.tensor([o_idx]))
average_distance = (distance_sum / len(kg.train_idx)).item()


truth_values = []
for test_triple_idx in kg.test_idx:
    s_idx, p_idx, o_idx = test_triple_idx
    distance = model.forward(torch.tensor([s_idx]), torch.tensor([p_idx]), torch.tensor([o_idx])).item()
    truth_values.append(min(1, average_distance/distance))


# ----------------- Create output file --------------------
fact_iris = []
with open("./fokg-sw-test-2023.nt", "r") as f:
    for line in f:
        iri = line.strip().split()[0]
        if iri not in fact_iris:
            fact_iris.append(iri)

output_data = [(fact_iri, "<http://swc2017.aksw.org/hasTruthValue>", '"' + str(truth_value) + '"^^<http://www.w3.org/2001/XMLSchema#double> .') for fact_iri, truth_value in zip(fact_iris, truth_values)]

transform.write_data(output_data, "./result.ttl")