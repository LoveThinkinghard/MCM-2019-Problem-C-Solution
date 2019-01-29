# MCM-2019-Problem-C-Solution

We participated in the MCM 2019, and chose the [Problem](https://www.comap.com/undergraduate/contests/mcm/contests/2019/problems/) C. This is our solution.

Also, we made some beautiful gifs to visualize the spread of opioids. You can download them and their code here.

## Acknowledgement

The latitude and longitude data of 461 counties come from [SimpleMaps.com](https://simplemaps.com/data/us-cities), which is for free. Great Thanks!

Thanks, my teammates, without whom I have no way to finish this.

## Examples of spread map gifs

### All drugs

(which I mean `TotalDrugReportsCounty` in the [problem](https://www.comap.com/undergraduate/contests/mcm/contests/2019/problems/) file `MCM_NFLIS_Data.xlsx`)

![All drugs](https://github.com/LoveThinkinghard/MCM-2019-Problem-C-drug-spread-maps/blob/master/spread%20maps/All%20Drugs%20map.gif)

### Heroin

![Heroin](https://github.com/LoveThinkinghard/MCM-2019-Problem-C-drug-spread-maps/blob/master/solution/gifs/Heroin%20map.gif)

### Fentanyl

![Fentanyl](https://github.com/LoveThinkinghard/MCM-2019-Problem-C-drug-spread-maps/blob/master/solution/gifs/Fentanyl%20map.gif)

## Explanation of our model

### Overall

We use one `state-vector` to describe the drug use state of one year. And a `transfer-matrix` to make the transfer of one year to the next year. And when you want to predict before just use a `inverse-transfer-matrix`, which is very convenient. So our model would be like this:

![formula](https://github.com/LoveThinkinghard/MCM-2019-Problem-C-drug-spread-maps/blob/master/pics/formula.png)

Or as the figure below illustrated (461 is the number of counties).

![model](https://github.com/LoveThinkinghard/MCM-2019-Problem-C-drug-spread-maps/blob/master/pics/model.jpg)

Of course the `transfer-matrix` is not just a matrix, there are more details.

### Details of first model

`transfer-matrix` of our first model, which uesd to solve the first part, consist of two part: `distance-matrix` and `correction-matrix`. And you might think of it that our `correction-matrix` is actually a fully connected network of 461 inputs and 461 outputs, which is trainable while our `distance-matrix` is designed by hand. Figure below illustrates this.

![model1_detail](https://github.com/LoveThinkinghard/MCM-2019-Problem-C-drug-spread-maps/blob/master/pics/model1_detial.jpg)

`distance-matrix` is designed to represent to distance of two counties. For example, if two counties are close to each other, we think that one county's drugs might be more likely to spread to anthor, which is a natural thinking. And about how exactly we design this `distance-matrix`, the answer would be: just try.

`correction-matrix` is designed to correct the `distance-matrix` and represent some information we don't consider. But actually if you say: `distance-matrix` is designed to accelerate the learning of network, I will also agree with that.

### Details of second model

Based on the first model, we add a new part, called `economic-matrix`, which represent the influence of socio-economic parameters, and mainly to one county itself.
So the `economic-matrix` is diagonal matrix. The Value is determinded by one county's socio-economic data. And how to get each parameter's weight? The solution is another network. And to explain how each parameter influence the `economic-matrix`, we choose fully connected network again, because it's linear and the weights of network can represent the weights of socio-economic parameters. So the structure would be like this:

![model2_detail](https://github.com/LoveThinkinghard/MCM-2019-Problem-C-drug-spread-maps/blob/master/pics/model2_detial.jpg)

### Part of results

#### loss of the first model

![model1_loss](https://github.com/LoveThinkinghard/MCM-2019-Problem-C-drug-spread-maps/blob/master/pics/model1_loss.png)

#### loss of the second model

![model2_loss](https://github.com/LoveThinkinghard/MCM-2019-Problem-C-drug-spread-maps/blob/master/pics/model2_loss.png)

#### prediction of the first model

![prediction](https://github.com/LoveThinkinghard/MCM-2019-Problem-C-drug-spread-maps/blob/master/pics/prediction.png)

#### Summary of results

This first model performs not bad, as you can see. But not on every sample. But the second model is not that good, the loss is very high, though decreased a lot. That might because we make some mistakes when pre-process the socio-economic data. We thought the tags (like 'HC01_VC03') are the same in each file, however, we are wrong. when we find this mistake it's very close to the deadline. So we leave that problem. But we do believe that the second model would be a good model if fed well.
Also, when we try to inverse the model to predict before, we failed again, some bad results come out, and it's not solved now.

### Small details about our model

We don't use any activation function, because we want to make inverse to predict before. Though we know it will be more reasonable if we put a relu at the final output, because the number of drug use is all positive.

Why we don't use more complex network like CNN, or even LSTM? It's also for making inverse.
