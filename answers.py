r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    lr = 0.02
    reg = 0.01
    wstd = 0.1
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0, 0, 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr_vanilla = 0.02
    lr_momentum = 0.0015
    lr_rmsprop = 0.0001
    reg = 0.01
    # ========================
    return dict(wstd=wstd, lr_vanilla=lr_vanilla, lr_momentum=lr_momentum,
                lr_rmsprop=lr_rmsprop, reg=reg)


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr = 0.00075
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
No dropout had lower loss on training and testing and had a better acuracy on training but did worse on testing. That make sense since it overfits more that it will have higher certainty and lower generalization. it remmbers more then it actually trains.

The low dropout vs high dropout is actually the more surprising one. the loss of 0.4 was much higher then 0.8 (both higher then no dropout). 0.4 trains less well then the 0.8 (expected it to surpass it since overfits more) and 0.8 does better then 0.4 on testing (make sense because of better generalization probably). so this answer was the more surprising one.

"""

part2_q2 = r"""

**Yes. It is possibole. Example:

1 epoch for example has 10 examples with a binary result:

the first got 8/10 right

with probability almost 1 for correct answers and probabiliy almost 0.5 for wrong answers. therefore a low loss of about 1.

The second got 9/10 right

with probability of almost 0.5 for all the answers so the loss is much higer and around 6.

So in this this example the loss has increase but also the accuracy.**






"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
Generaly adding more conv layers help us find more features. As the features are more specific,
we can classify a given sample more accurately.  
Adding too much layers, will cause ovefitting as a result of finding irregularities in the samples and "mark" them 
as features.

Yes, some models weren't trainable. L values 2 and 16 made the model not trainable. 
Too deep convolution network also involves many pooling layers which downsample the features.
If the features are downsampled too much we lose too much information, thus the samples can't 
be classified properly. 
We could solve it by using less pooling layers, but this will cause too many neurons
int the input layers of the classifier, which increase training time.
   
Too shallow conv network (L=2) doesn't produce features which are "good enough".
i.e it finds features which are common to most of the samples anyway.

"""

part3_q2 = r"""
We can clearly see that deeper conv net produce more features, thus our 
samples can be classified more precisely. More filters it have, more features
it produces. 
Of course we should be able to see same results for same configurations.

"""

part3_q3 = r"""
We can see that deeper conv nets tend to overfit faster, as a result of finding 
irregularities in the samples.

"""


part3_q4 = r"""
- Batch normalization added.
In experiment 1 we saw an overfitting problem for K64-128-256 conv net.
Normilizing the samples before injecting them to classifier solved the problem.
Unfortunately we could not perform experiments on all required configurations,
But still we can see better performance.

- Padding disabled. 
Disabling paddings cause less number of features injected to classifier.
We guess this has minor influence on accuracy, but slightly shorter training time.
"""
# ==============
