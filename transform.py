def load_data(data_dir):
    # Load the positive data in the project format into a list of (s,p,o) triples
    transformed_data = []
    with open(data_dir, "r") as f:
        i = 0
        new_fact = tuple()
        for line in f:          
            s, p, o = line.strip()[:-1].split()
            i = i + 1
            if i in [2,3,4]:
                new_fact += (o,)
            elif i == 5:
                i = 0
                transformed_data.append(new_fact)
                new_fact = tuple()
    return transformed_data[:500].copy()


def write_data(data):
    # Write list of (s,p,o) triples into a new file
    with open("train.txt", "a") as f:
        for fact in data:
            f.write(fact[0] + " " + fact[1] + " " + fact[2] + "\n")