# Chapter 1: Single Point Positioning

In this project, we will compute the **Single Point Positioning** solution, which consists in:
- using code observations
- using the broadcast navigation message for satellite position and correction computation
- using a Weighted Least Squares algorithm to compute the solution at each epoch
- estimating the receiver position and clock

To do so, the following steps are proposed:
- apply corrections to code observations
- compute the Jacobian matrix of the observation model
- compute the SPP solution (snapshot solution)

Additionally, along the project, we will observe:
- the characteristics of some of the error terms affecting the code observations: iono/tropo delay, noise
- compute the Dilution of Precision factors over time

