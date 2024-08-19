from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV, cross_val_score
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score

#removed param X
def baseline_model(optimizer='adam', learn_rate=0.1):
    model = Sequential()
    model.add(Dense(100, input_dim=X.shape[1], activation='relu'))
    model.add(Dense(50, activation='relu'))  # 8 is the dim/ the number of hidden units (units are the kernel)
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model


def get_probability_labels(x, y, optimizer='adam'):
    all_predictions = []
    cv_5 = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
    estimator = KerasClassifier(optimizer=optimizer, batch_size=32, epochs=100, build_fn=baseline_model, verbose=0)
    for train_index, test_index in cv_5.split(x, y):
        X_train, X_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        estimator.fit(X_train, y_train)
        # change predict_proba (deprecated) and remove list(predictions[:, 1])
        predictions = estimator.predict(X_test)
        all_predictions.append(predictions)
        a = [j for i in all_predictions for j in i] #remove nested list
    return a

def add_labels(real_data, synthetic_data):

    # add labels 0 for real and 1 for synthetic
    data = pd.concat([real_data, synthetic_data], ignore_index=True)
    o_labels = np.zeros((len(real_data)), dtype=int)
    s_labels = np.ones((len(synthetic_data)), dtype=int)
    labels = np.concatenate([o_labels, s_labels], axis=0)
    data['class'] = labels
    x = data.drop('class', axis=1)
    y = data['class']

    return x, y

# other file
def main():
    X, Y = add_labels(df, df_synth)
    probability_labels = get_probability_labels(X, Y)
    print(probability_labels)