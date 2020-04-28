# Homework 2

Course: 💡ML\
Dates: Mar 30, 2020 → Apr 06, 2020\
Done: Yes\
Remain: -29 days\
Type: 📌 Assignment\


# Naive Bayes Classifier

Create a Naive Bayes classifier for each handwritten digit that support discrete and continuous features.

## Input

1. Training image data from MNIST
2. Training label data from MNIST
3. Testing image from MNIST
4. Testing label from MNIST
5. Toggle option
    - 0: discrete mode
    - 1: continuous mode

## Output

1. Print out the the posterior (in log scale to avoid underflow) of the ten categories (0-9) for each image in INPUT 3. Don't forget to marginalize them so sum it up will equal to 1
2. For each test image, print out your prediction which is the category having the highest posterior, and tally the prediction by comparing with INPUT 4
3. Print out the imagination of numbers in your Bayes classifier
    - For each digit, print a 28 x 28 binary image which 0 represents a white pixel, and 1 represents a black pixel
    - The pixel is 0 when Bayes classifier expect the pixel in this position should less then 128 in original image, otherwise is 1
4. Calculate and report the error rate in the end

## Function

- Discrete mode
Tally the frequency of the values of each pixel into 32 bins each bin cross 8 gray level. Then perform Naive Bayes classifier.
To avoid empty bin, you can use a pesudocount (such as the minimum value in other bins) for instead.
- Continuous mode
Use MLE to fit a Gaussian distribution for the value of each pixel. Perform Naive Bayes classifier.

## Reference

[貝葉斯推斷和各類機率 Bayesian Inference](https://brohrer.mcknote.com/zh-Hant/statistics/how_bayesian_inference_works.html)

[朴素贝叶斯分类器](https://zh.wikipedia.org/wiki/%E6%9C%B4%E7%B4%A0%E8%B4%9D%E5%8F%B6%E6%96%AF%E5%88%86%E7%B1%BB%E5%99%A8)

[Bayes classifier and Naive Bayes tutorial (using the MNIST dataset) - Lazy Programmer](https://lazyprogrammer.me/bayes-classifier-and-naive-bayes-tutorial-using/)

## Implement

### Read binary byte into integer

```python
with open(filename, 'rb') as f:
    byte = f.read()
    print(int.from_bytes(byte[:4], byteorder='big', signed=False))
```

### Extract image and display

```python
import numpy as np
import matplotlib.pyplot as plt

with open(filename, "rb") as f:
        f.seek(index*28*28+16)
        bytes = f.read(28*28)
img = np.array([i for i in bytes]).reshape((28,28))
plt.imshow(img, cmap='gray')
plt.show()
```

![$$P(C|x) = {P(x|C)P(C)\over P(x)}$$](http://www.sciweavers.org/upload/Tex2Img_1588067637/render.png)

x is the vector formatted by image with 784 slots each could be 0...255

C is the digit

P(C|x) is the posterior means the probability that the class is C given the data x

P(x|C) is the likelihood means the probability that the data x belong to class C

![$$C^*=argmax_CP(C|x)=argmax_c{P(x|C)P(C)\over P(x)}$$](http://www.sciweavers.org/upload/Tex2Img_1588067771/render.png)

P(x) is a constant and can be ignored

### Calculate the prior

P(C) is the prior means the probability that digit C appear among all images

```python
labels = list()
with open(filename, "rb") as f:
    bytes = f.read()
    size = int.from_bytes(bytes[4:8],  byteorder='big', signed=False)
for i in range(size):
    labels.append(bytes[i + 8])

def prior(n):
    assert (0 <= n and n < 10)
    counter = Counter(labels)
    return (counter[n] / sum(counter.values()))
priors = [prior(i) for i in range(10)]
```

### Calculate the likelihood

For Gaussian case

![$$P(x|C)=\prod_{i=1}^{784}{1\over \sqrt{2\pi\sigma_i^2}}exp(-{(x_i-\mu_i)^2\over 2\sigma_i^2})$$](http://www.sciweavers.org/upload/Tex2Img_1588067836/render.png)

# Online Learning

Use online learning to learn the beta distribution of the parameter p (chance to see 1) of the coin tossing trails in batch.

## Input

1. A file contains many lines of binary outcomes
2. a
3. b

a, b are the parameters for the initial beta prior

## Output

Print out the

- Binomial likelihood (based on MLE, of course),
- Beta prior
- posterior probability (parameters only)

for each line

## Function

Use Beta-Binomial conjugation to perform online learning

## Implement

TL;DR
