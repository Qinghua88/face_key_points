from utils import load_data
import kmodel

# load training data
X_train, y_train = load_data()

# create neural networks
my_model = kmodel.create_model()

# compile model
kmodel.compile_model(my_model)

# training model
kmodel.train_model(my_model, X_train, y_train)

# save model
kmodel.save_model(my_model, 'my_model')
