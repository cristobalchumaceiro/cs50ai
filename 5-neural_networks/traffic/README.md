# TensorFlow Sequential Model Experimentation

This model is trained on the [German Traffic Sign Recognition Benchmark](https://benchmark.ini.rub.de/?section=gtsrb&subsection=news) (GTSRB) dataset, which contains thousands of images of 43 different kinds of road signs. The directory used is available [here](https://cdn.cs50.net/ai/2023/x/projects/5/gtsrb.zip). You can run my code by placing the ```gtrsb``` directory inside the ```traffic``` directory and executing:

```
python traffic.py gtsrb
```

with an optional 3rd argument of a file name to save the compiled model.

I first started with the following model introduced in the lecture on Convolutional Neural Networks for handwriting recognition using the MNIST dataset. 

``` 
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
```

Though it was a good start to at least get everything up and running, it didn't produce very good results, with the final accuracy and loss shown below:

```
Epoch 1/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 2s 4ms/step - accuracy: 0.0544 - loss: 5.1667      
...
Epoch 10/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 3s 5ms/step - accuracy: 0.0576 - loss: 3.4975 
333/333 - 1s - 2ms/step - accuracy: 0.0543 - loss: 3.4998
```

## Convolutional Layers

I decided to include another convolutional layer, this time increasing the size, following the idea of each layer filtering for increasingly more detail. First going for basic borders, curves, shapes, and other low-level features, then during a second pass, identifying high-level features such as objects. 

Moreover, when max-pooling by a factor of two, you are effectively downsampling the image by two, and so with the subsequent convolutional layer I multiply the number of channels by 2.

```
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
])
```

And we can see below, that extra layer results in an incredible difference, with a similar amount of computational time taken and an accuracy of around 96%

```
Epoch 1/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 3s 4ms/step - accuracy: 0.1948 - loss: 3.4867         
Epoch 2/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 2s 5ms/step - accuracy: 0.4902 - loss: 1.7624 
...
Epoch 9/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 3s 5ms/step - accuracy: 0.8968 - loss: 0.3451 
Epoch 10/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 3s 6ms/step - accuracy: 0.9082 - loss: 0.3030 
333/333 - 1s - 2ms/step - accuracy: 0.9672 - loss: 0.1366
```

My next step was to add one more convolutional layer  of 128 filters, to try and bring the accuracy up, and below are the results.

```
Epoch 1/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 3s 5ms/step - accuracy: 0.2533 - loss: 3.0741         
Epoch 2/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 3s 5ms/step - accuracy: 0.6079 - loss: 1.3162 
...
Epoch 9/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 3s 7ms/step - accuracy: 0.9418 - loss: 0.2122 
Epoch 10/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 4s 7ms/step - accuracy: 0.9489 - loss: 0.1901 
333/333 - 1s - 3ms/step - accuracy: 0.9748 - loss: 0.0955
```

This achieved a slightly higher degree of accuracy, which I'm happy with, so I moved on from toying with the convolutional layers and began to test with different values for the hidden layer. 

## Hidden Layer

I increased it to 256, but as evidenced below, though the model began with a higher level of accuracy at Epoch 1/10, and carried it forward, it reached a plateau quickly, and saw diminishing returns from later epochs.

```
Epoch 1/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 3s 5ms/step - accuracy: 0.4721 - loss: 2.1249     
Epoch 2/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 3s 6ms/step - accuracy: 0.8231 - loss: 0.6083 
...
Epoch 9/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 4s 7ms/step - accuracy: 0.9669 - loss: 0.1345 
Epoch 10/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 4s 8ms/step - accuracy: 0.9660 - loss: 0.1339 
333/333 - 1s - 3ms/step - accuracy: 0.9721 - loss: 0.0984
```

## Dropout Rate

Now below we test 3 different dropout rates (the probability of temporarily deactivating random neurons in a layer during each training step). A dropout rate is set for the purpose of safeguarding a model from overfitting. This situation arises when the model is very highly trained on the data and does not generalise well to new, unseen data. By avoiding over-reliance on specific neurons, we drop them randomly to make the model more robust.

Dropout rate: 0.6

```
Epoch 10/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 4s 8ms/step - accuracy: 0.9220 - loss: 0.2644 
333/333 - 1s - 3ms/step - accuracy: 0.9694 - loss: 0.1180
```

Dropout rate: 0.4

```
Epoch 10/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 4s 7ms/step - accuracy: 0.9531 - loss: 0.1780 
333/333 - 1s - 3ms/step - accuracy: 0.9698 - loss: 0.1170
```

Dropout rate: 0.3

```
Epoch 10/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 3s 6ms/step - accuracy: 0.9707 - loss: 0.1135 
333/333 - 1s - 3ms/step - accuracy: 0.9791 - loss: 0.1010
```

We see the model become slightly more accurate as we lower the dropout rate, but to strike a balance I decide to keep it at 0.5.

## Average Pooling vs. Max Pooling

Finally, just for comparison's sake, I change the pooling layers to Average Pooling from Max Pooling, whilst keeping the pooling window at (2, 2) and the stride length at 2.

```
Epoch 10/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 4s 7ms/step - accuracy: 0.9705 - loss: 0.1091 
333/333 - 1s - 3ms/step - accuracy: 0.9883 - loss: 0.0426
```

After running the model several times, I ended up with a higher degree of accuracy on all of the runs. Most likely due to the fact that the exercise of identifying road signs benefits more from a higher degree of contextual information being retained about a particular patch. Instead of just the max value of all inputs in a patch.

## Conclusion
I'm happy about the level of accuracy I've obtained through all the tweaking. More refinement could be made, such as a mix of pooling methods, integrating a dropout after convolutional layers, in addition to the one after flattening. We could also make changes to the activation functions, or increase the amount of dense layers. Finally, we could also adjust learning rates, or turn the Convolutional Neural Network into a Recurrent Neural Network (RNN) or even use a mixture of both in a model. 

For a simple categorisation problem like this one however, the parameters chosen are adequate enough.
