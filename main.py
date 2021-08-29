import pandas as pd
import numpy as np
from functions import generate_data, get_weighted_sum, sigmoid, cross_entropy, update_bias, update_weights

bias = 0.5
l_rate = 0.01
epochs = 30
epoch_loss = []
data, weights = generate_data(50, 3)


for e in range(epochs):
    individual_loss = []
    for i in range(len(data)):
        feature = data.loc[i][:-1]
        target = data.loc[i][-1]
        w_sum = get_weighted_sum(feature, weights, bias)
        prediction = sigmoid(w_sum)
        loss = cross_entropy(target, prediction)
        individual_loss.append(loss)
        # gradient descent
        weights = update_weights(weights, l_rate, target, prediction, feature)
        bias = update_bias(bias, l_rate, target, prediction)
    average_loss = sum(individual_loss)/len(individual_loss)
    epoch_loss.append(average_loss)
    print("==============================================")
    print("epoch:", e, "- loss:", average_loss)

df = pd.DataFrame(epoch_loss, columns=["loss"])
df_plot = df.plot(kind = "line", grid = True, legend = True).get_figure()
df_plot.savefig("training_loss.png")