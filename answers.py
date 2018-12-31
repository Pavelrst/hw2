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
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part3_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============
