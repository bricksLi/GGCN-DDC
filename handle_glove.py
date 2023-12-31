import numpy as np
import joblib


def load_data(embedding_dim):
    words, vectors = [], []
    with open(f"source/glove.6B.{embedding_dim}d.txt", "r", encoding="utf-8") as f:
        line = f.readline()
        while line != "":
            line = line.strip().split()
            words.append(line[0])
            vectors.append(np.array(line[1:], dtype=np.float))
            line = f.readline()
    vectors = np.array(vectors)
    return words, vectors


if __name__ == '__main__':
    for embedding_dim in [50, 100, 200, 300]:
        print(embedding_dim)
        words, vectors = load_data(embedding_dim)
        joblib.dump(words, f"source/glove.6B.words.pkl")
        # print(type(words),len(words),type(words[1]))  # List，400000，str
        # print(vectors.shape)  # (400000,50)
        np.save(f"source/glove.6B.{embedding_dim}d.npy", vectors)

    data = np.load('source/glove.6B.50d.npy')
    print(len(data[0]),type(data),data)
