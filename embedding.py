import torch

class KG:
    def __init__(self, data_dir=None):
        
        # 1. Parse the benchmark dataset
        s = '------------------- Description of Dataset' + data_dir + '----------------------------'
        print(f'\n{s}')
        self.train = self.load_data(data_dir + 'train.txt')
        #self.valid = self.load_data(data_dir + 'valid.txt')
        #self.test = self.load_data(data_dir + 'test.txt')
        
        self.all_triples = self.train #+ self.valid + self.test
        self.entities = self.get_entities(self.all_triples)
        self.relations = self.get_relations(self.all_triples)

        # 2. Index entities and relations
        self.entity_idxs = {self.entities[i]: i for i in range(len(self.entities))}
        self.relation_idxs = {self.relations[i]: i for i in range(len(self.relations))}

        print(f'Number of triples: {len(self.all_triples)}')
        print(f'Number of entities: {len(self.entities)}')
        print(f'Number of relations: {len(self.relations)}')
        print(f'Number of triples on train set: {len(self.train)}')
        #print(f'Number of triples on valid set: {len(self.valid)}')
        #print(f'Number of triples on test set: {len(self.test)}')
        s = len(s) * '-'
        print(f'{s}\n')

        # 3. Index train, validation and test sets 
        self.train_idx = [(self.entity_idxs[s], self.relation_idxs[p], self.entity_idxs[o]) for s, p, o in self.train]
        #self.valid_idx = [(self.entity_idxs[s], self.relation_idxs[p], self.entity_idxs[o]) for s, p, o in self.valid]
        #self.test_idx = [(self.entity_idxs[s], self.relation_idxs[p], self.entity_idxs[o]) for s, p, o in self.test]

        # 4. Create mappings for the filtered link prediction
        self.sp_vocab = dict()
        self.po_vocab = dict()
        self.so_vocab = dict()

        for i in self.all_triples:
            s, p, o = i[0], i[1], i[2]
            s_idx, p_idx, o_idx = self.entity_idxs[s], self.relation_idxs[p], self.entity_idxs[o]
            self.sp_vocab.setdefault((s_idx, p_idx), []).append(o_idx)
            self.so_vocab.setdefault((s_idx, o_idx), []).append(p_idx)
            self.po_vocab.setdefault((p_idx, o_idx), []).append(s_idx)


    @staticmethod
    def load_data(data_dir):
        with open(data_dir, "r") as f:
            data = f.read().strip().split("\n")
            data = [i.split() for i in data]
        return data

    @staticmethod
    def get_relations(data):
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    @staticmethod
    def get_entities(data):
        entities = sorted(list(set([d[0] for d in data] + [d[2] for d in data])))
        return entities

    @property
    def num_entities(self):
        return len(self.entities)
    
    @property
    def num_relations(self):
        return len(self.relations)
    

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, num_entities=None):
        data = torch.Tensor(data).long()
        self.head_idx = data[:, 0]
        self.rel_idx = data[:, 1]
        self.tail_idx = data[:, 2]
        self.num_entities = num_entities
        assert self.head_idx.shape == self.rel_idx.shape == self.tail_idx.shape

        self.length = len(self.head_idx)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        h = self.head_idx[idx]
        r = self.rel_idx[idx]
        t = self.tail_idx[idx]
        return h, r, t

    def collate_fn(self, batch):
        """ Generate Negative Triples"""
        batch = torch.LongTensor(batch)
        h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]
        size_of_batch, _ = batch.shape
        assert size_of_batch > 0
        label = torch.ones((size_of_batch, ))
        # Generate negative/corrupted Triples
        corr = torch.randint(0, self.num_entities, (size_of_batch, 1))
        
        if torch.rand(1).item() > 0.5:
            # Corrupt head
            h_corr = corr[:, 0]
            r_corr = r
            t_corr = t
            label_corr = -torch.ones(size_of_batch, )
        else:
            # Corrupt tail
            h_corr = h
            r_corr = r
            t_corr = corr[:, 0]
            label_corr = -torch.ones(size_of_batch, )

        # 3. Stack True and Corrupted Triples
        h = torch.cat((h, h_corr), 0)
        r = torch.cat((r, r_corr), 0)
        t = torch.cat((t, t_corr), 0)
        label = torch.cat((label, label_corr), 0)
        return h, r, t, label


class TransE(torch.nn.Module):
    def __init__(self, embedding_dim, num_entities, num_relations, **kwargs):
        super(TransE, self).__init__()
        self.name = 'TransE'
        self.embedding_dim = embedding_dim
        self.num_entities = num_entities
        self.num_relations = num_relations

        self.emb_ent = torch.nn.Embedding(self.num_entities, self.embedding_dim)
        self.emb_rel = torch.nn.Embedding(self.num_relations, self.embedding_dim)
        
        self.low = -6 / torch.sqrt(torch.tensor(self.embedding_dim)).item()
        self.high = 6 / torch.sqrt(torch.tensor(self.embedding_dim)).item()
        
        self.emb_ent.weight.data.uniform_(self.low, self.high)
        self.emb_rel.weight.data.uniform_(self.low, self.high)
        
        
    def forward(self, e1_idx, rel_idx, e2_idx):
        # (1) Embeddings of head, relation and tail
        emb_head = self.emb_ent(e1_idx)
        emb_rel = self.emb_rel(rel_idx)
        emb_tail = self.emb_ent(e2_idx)
               
        # (2) Normalize head and tail entities
        emb_head = torch.nn.functional.normalize(emb_head, p=2, dim=1)
        emb_tail = torch.nn.functional.normalize(emb_tail, p=2, dim=1)
        
        # (3) Compute distance, i.e., the L2 norm of head + relation - tail
        distance = torch.norm(emb_head + emb_rel - emb_tail, p=2, dim=1)
        
        return distance
    


def train(model, dataset_train, hparams):
    # Disabled parallel loads because it throws an error otherwise
    dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=hparams['batch_size'],
        #num_workers=4,
        shuffle=True,
        drop_last=True,
        collate_fn=dataset_train.collate_fn)

    gamma = torch.nn.Parameter(torch.Tensor([ hparams['gamma'] ]), requires_grad=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'])


    def loss_function(gamma, pos_distance, neg_distance):
        return torch.nn.functional.relu(gamma + pos_distance - neg_distance).sum()


    for e in range(1, hparams['num_epochs']):
        epoch_loss = 0.0
        i = 1
        # iterate over batches (h, r, t are vectors of indices)
        for h, r, t, labels in dataloader:
            print("Batch " + str(i))
            optimizer.zero_grad()
            
            # Compute Distance based on translation, i.e. h + r \approx t provided that h,r,t \in G.
            distance = model.forward(h, r, t)    
            
            pos_distance = distance[labels == 1]
            neg_distance = distance[labels == -1]

            loss = loss_function(gamma, pos_distance, neg_distance)
            
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            
            i = i + 1
            
        if e % 1 == 0:
            print(f'{e}.th epoch sum of loss: {epoch_loss}')
