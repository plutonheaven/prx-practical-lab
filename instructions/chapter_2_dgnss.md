# Chapter 2: SPP with differential corrections

In this project, we will compute the **Differntial GNSS** solution, which consists in:
- using code observations
- using the observations from a nearby stations to correct our own observations
- using a Weighted Least Squares algorithm to compute the solution at each epoch
- estimating the receiver position and clock

To do so, the following steps are proposed:
- apply corrections to code observations
- compute the Jacobian matrix of the observation model
- compute the DGNSS solution for each epoch (snapshot solution)

Additionally, along the project, we will:
- observe the characteristics of the differentially-corrected code observations
- compare the DGNSS and the SPP solution

