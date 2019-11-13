# Evaluation metrics summary

## Confusion matrix

| | predicted $\hat{c}=0$ | predicted $\hat{c}=1$ |
--|--|--|--|
**actual** $c=0$ | TN | FP |
**actual** $c=1$ | FN | TP |

### Precision 
* How many out of *classified* positives were true. 
* What proportion of positive identifications was actually correct?
* Important when we want to be sure that whenever we classify instance as positive we are correct, e.g. in making recommendations we want a given recommendation to be good even if we miss out on giving all recommendations.
  $$P = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FP}}$$
### Recall 
* How many true positives *in total* has the classifier detected.
* What proportion of actual positives was identified correctly?
* Important when we want to be safe about catching as many positive cases as we can, e.g. cases of cancer.
  $$R = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FN}}$$
### $F_1$ score 
* For combining precision and recall.
  $$F_1 = 2\times \frac{P\times R}{P+R}$$

  More generally, 
  $$F_\beta = (1 + \beta^2)\frac{P \times R}{\beta^2 \times P + R}$$ 
  where $\beta < 1$ emphasises precision and $\beta > 1$ emphasises recall.

## Receiver Operating Characteristic (ROC)
### True positive rate (recall, sensitivity)
$$\mathrm{TPR} = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FN}}$$

### False positive rate
$$\mathrm{FPR} = 1 - \mathrm{specificity} = 1 - \frac{\mathrm{TN}}{\mathrm{TN} + \mathrm{FP}} = \frac{\mathrm{FP}}{\mathrm{TN} + \mathrm{FP}}$$

### ROC curve
Plotting TPR and FPR for varying classification thresholds




