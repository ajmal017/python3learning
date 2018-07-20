
import pickle
# from .timer import timer
import importlib
import sys

def train(model, x, y):
    model.fit(x, y)

def save(model, path):
    pickle.dump(model, open(path, 'wb'))

def load(path):
    return pickle.load(open(path, 'rb'))

# @timer
def use():
    # test use
    from sklearn.tree.tree import DecisionTreeClassifier
    import sklearn.datasets
    path = 'model.pkl'
    iris = sklearn.datasets.load_iris()
    model = DecisionTreeClassifier()
    # a = None
    # try:
    #     a.test()
    # except Exception as e:
    #     traceback.print_exc()

    train(model, iris.data, iris.target)
    save(model, path)
    model = load(path)
    print(model.predict(iris.data))

def import_module(path, package):
    sys.path.append(path)
    return importlib.import_module(package)

if __name__ == '__main__':
    use()
