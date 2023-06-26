## A simple Fact Checking Engine
This is a simple fact checking engine created for the course "Foundations of Knowledge Graphs" at Paderborn University. It receives a knowledge graph represented in RDF as input and then computes veracity values in [0,1] for given facts based on the knowledge graph.

### Approach
The engine trains a [TransE embedding](https://dl.acm.org/doi/10.5555/2999792.2999923) using the input knowledge graph. This means that every property $p$ can be translated to a vector $\overrightarrow{p}$, such that for the triples $(s,p,o)$ of the knowledge graph $\overrightarrow{s} + \overrightarrow{p} \approx \overrightarrow{o}$ holds. After the training is complete, the engine computes the average distance $average^+$ between $\overrightarrow{s} + \overrightarrow{p}$ and $\overrightarrow{o}$ of all positive training triples $(s,p,o)$, which is then used to compute the veracity values. A new fact $(s', p', o')$ is assigned a veracity value by translating each property to a vector using the learned embedding, computing the distance between $\overrightarrow{s'} + \overrightarrow{p'}$ and $\overrightarrow{o'}$ and comparing it to the average distance: $$veracity(s',p',o') = \min(1, \frac{average^+}{distance(\overrightarrow{s'} + \overrightarrow{p'}, \overrightarrow{o'})})$$
This way, facts for which the embedding behaves poorly get a small veracity value, whereas facts with a good performance get a high veracity value.

### How to execute
A local installation of [Python](https://www.python.org/) with [PyTorch](https://pytorch.org/) is required. The code was tested using Python 3.8 (64 bit). The code files (```fact_checking.py```, ```embedding.py```, ```transform.py```) and input files in the project format (```fokg-sw-train-2023.nt```, ```fokg-sw-test-2023.nt```) have to be in the same folder. Simply running ```fact_checking.py``` will result in the ```result.ttl``` output file:

```python fact_checking.py```
