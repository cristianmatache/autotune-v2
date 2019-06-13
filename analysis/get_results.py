import pickle

path = "D:/datasets/outputresults.pkl"


if __name__ == "__main__":
    with open(path, "rb") as f:
        optimizer = pickle.load(f)[0]
        print(optimizer.arms)
        print(optimizer.val_loss)
