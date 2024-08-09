# Peaks-and-lowlands

Python realization of the lowlands algorithm for multimodality detection, based on quantile-respectful density estimation.

The algorithm was proposed by Andrey Akinshin. See for more details: https://aakinshin.net/posts/lowland-multimodality-detection/

Documentations can be found at the https://iasivkov.github.io/peaks-and-lowlands

## General description of the algorithm

* Determine the quantiles based on the Harrell-Davis estimator.

* Using these quantiles, estimate the density function.

* Locate all the maxima (peaks) in the estimated density function.

* Fill the areas between peaks with “water,” with the water level defined by the height of the lower peak.

* Ponds where the ratio of the water area to the area of the ground under the water exceeds a certain threshold are marked as “lowlands.” These ponds cannot be merged with others.

* For each pond, starting from the left, check if it can merge with the next pond (the left peak of the current pond must be higher than the right peak of the next pond, and the next pond must not be a lowland). If the conditions are met, merge the next pond with the current one, adjusting the water level based on the new right peak.

## Examples

For examples see documentaion section "Examples", example_script.py and example_notebook.ipynb
