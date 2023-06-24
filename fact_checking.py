import transform
import embedding

# ----- Transform input data to easier format --------
data = transform.load_data("./fokg-sw-train-2023.nt")
transform.write_data(data)

# --- Compute TransE embedding of knowledge graph ----
kg = embedding.KG(data_dir='./')
print(kg.train[:2])
print(kg.train_idx[:2])

dataset_train = embedding.Dataset(data=kg.train_idx, num_entities=kg.num_entities)

hparam = {
    'embedding_dim': 25,
    'num_entities': kg.num_entities,
    'num_relations': kg.num_relations,
    'gamma': 1.0,  # margin for loss
    'lr': .01,  # learning rate for optimizer
    'batch_size': 256,
    'num_epochs': 500
}

model = embedding.TransE(**hparam)

embedding.train(model, dataset_train, hparam)