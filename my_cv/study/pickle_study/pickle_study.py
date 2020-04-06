import pickle


def write(file_name, args):
    with open(file_name, 'wb') as f:
        pickle.dumps(args, f)


def read(file_name):
    with open(file_name, 'rb') as f:
        tmp = pickle.loads(f)
    return tmp 