# homework 3

Course: 💡ML\
Dates: Apr 13, 2020 → Apr 20, 2020\
Done: Yes\
Remain: -15 days\
Type: 📌 Assignment\


# Random data generator

## Univariant gaussian data generator

### Input

- Mean (Expectation) : `float` m
- Variance : `float` s

### Output

A data point from N(m,s)

### Reference

  can use uniform distribution function (Numpy)

[Normal distribution](https://en.wikipedia.org/wiki/Normal_distribution#Generating_values_from_normal_distribution)

## Polynomial basis linear model data generator

### Concept

![$$y=W^T\phi(x)+e$$](http://www.sciweavers.org/upload/Tex2Img_1588069366/render.png)

- W is a `nx1` vector
- e ~ N(0,a)

### Input

- n : `int` basis

![$$n=3\rightarrow y=w_0x^0+w_1x^1+w_2x^2$$](http://www.sciweavers.org/upload/Tex2Img_1588069398/render.png)

- a: `float` variance of e
- w: `array like` parameters of above equation with

### Output

y form a random generate x (-1<x<1)

# Sequential Estimator

Sequential estimate the mean and variance.

Data is given from the univariate **gaussian data generator.**

### Input

m, s for gaussian data generator

### Function

- Generate a data point from N(m,s)
- Use sequential estimation to find the current estimates to m and s
- Repeat steps above until the estimates converge

### Output

Print the new data point and the current estimates of m and s in each iteration.

### Reference

[Algorithms for calculating variance](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm)

### Implementation

For each iteration, get the data point(x,y) from gaussian generator, and update the data count, mean, variance with these equation:

Initially count(N), mean(m), variance(σ²), M₂ are 0  

![$$N' = N +1$$ $$\Delta_1=y-m$$ $$m' = {m+\Delta\over N'}$$ $$\Delta_2=y-m'$$ $$M_2'=M_2+\Delta_1\times\Delta_2$$ $$\sigma^2={M_2\over N'}$$](http://www.sciweavers.org/upload/Tex2Img_1588069423/render.png)

The criteria to break the loop is both mean and variance are converge, that is it's different between previous value is smaller then 0.0001.

# Baysian Linear regression

w is a `nx1` matrix

### prior:

![$$p(w)=N(w|m_0,S_0)$$ $$m_0=0, S_0=\alpha^{-1}I$$](http://www.sciweavers.org/upload/Tex2Img_1588069460/render.png)

N is the gaussian distribution.

α is the precision parameter

### posterior:

After obtain a data point (x,y)

![$$p(w|t)=N(w|m_N,S_N)$$ $$m_N=S_N(S_0^{-1}m_0+\beta\Phi^Tt)$$ $$S_N^{-1}=S_0^{-1}+\beta\Phi^T\Phi$$](http://www.sciweavers.org/upload/Tex2Img_1588069491/render.png)

Φ is the design matrix with size `1xn`

![$$\Phi=[x^0\ x^1\ ...\ x^{n-1}]$$](http://www.sciweavers.org/upload/Tex2Img_1588069507/render.png)

t is y

β is noise precision parameter

### Implementation

- Initialization
    - N = 0
    - β = 1 / b
    - mean(m) is a nx1 matrix with all zero
    - covariance(S) is a nxn matrix with identity matrix multiply 1/b
- For each  iteration
    - Get data point (x,y) from polynomial_generator
    - N += 1
    - Create the design matrix

    ![$$\Phi=[x^0\ x^1\ \cdot \cdot \cdot\ x^{n-1}]^T$$](http://www.sciweavers.org/upload/Tex2Img_1588069518/render.png)

    - Update S with

    ![$$S'^{-1}=S^{-1}+\beta\cdot\Phi^T\cdot\Phi$$](http://www.sciweavers.org/upload/Tex2Img_1588069563/render.png)

    - Update m with

    ![$$m'=S\cdot S^{-1}\cdot m+\beta\cdot \Phi^T\cdot y$$](http://www.sciweavers.org/upload/Tex2Img_1588069590/render.png)

    The criteria to break the loop is both mean and variance are converge, that is it's different between previous value is smaller then 0.00001 and count greater then 100.
