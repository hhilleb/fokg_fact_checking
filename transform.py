def load_data(data_dir):
    # Load the data in the project format into a list of (s,p,o) triples
    transformed_data = []
    with open(data_dir, "r") as f:
        new_fact = tuple()
        for line in f:          
            s, p, o = line.strip()[:-1].split()
            if "subject" in p or "predicate" in p:
                new_fact += (o,)
            elif "object" in p:
                new_fact += (o,)
                transformed_data.append(new_fact)
                new_fact = tuple()
                
    return transformed_data


def write_data(data, dir):
    # Write list of (s,p,o) triples into a new file
    with open(dir, "a") as f:
        for fact in data:
            f.write(fact[0] + " " + fact[1] + " " + fact[2] + "\n")