
# Metropolis-Hastings In-Class Project


* sample a 1d distribution where you already know the answer
    * check the mean and variance of your samples against the exact values
* vary the proposal width
    * what acceptance rate do you get, and what does the trace look like?
    * too small and too large both fail, in different ways
* find the burn-in by eye, discard it, see if your answer changes
* estimate the autocorrelation length and thin by it
    * how many independent samples did you actually get out of the chain?
    * (look at the example autorcorrelation script)
* Implement an ensemble sampler with 500 walkers
      * Run on the model and data we used for the prior 'fitting' exercise.
      * Plot the distribution of each parameter after 'burn in' 
      * Identify 'burn in' by eye by looking at trace plots (plot of the walker as a function of iteration). 
* Compare your result to using an off the shelf sampler like `emcee`.

* (1) write the likelihood for the fitting data
    * gaussian noise likelihood
    * your choice of priors
* (2) run your own model with many walkers
    * say 100-500, start from random initial positions
    * trace plot per parameter, identify burn-in by eye
* (3) corner plot from the walkers after burn-in
    * `pip install -r ../11-10/requirements.txt`
    * does the peak sit where your minimizer put the best fit?
    * Are there correlated parameters? 
* (4) pick one convergence test from the list above and apply it
    * different individuals take different ones, compare within your group
* (5) if there is time, run the same model through `emcee`
