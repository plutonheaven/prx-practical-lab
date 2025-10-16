# Chapter 3: Real-Time Kinematics

In this project, we will compute a **Real-Time Kinematics** solution, which consists in:
- using code and carrier observations
- using the observations from a nearby stations to correct our own observations
- using a Weighted Least Squares algorithm to compute the solution on a set of epochs with continuous phase tracking
- estimating the receiver position and clock, as well as the carrier phase ambiguity over the set of selected epochs.

To do so, the following steps are proposed:
- apply corrections to code and carrier observations
- identify a set of epochs with continuous phase tracking
- compute the Jacobian matrix of the observation model
- compute the WLS on a set of epochs (batch solution)
- try to fix the carrier phase ambiguities to integer numbers

Additionally, along the project, we will observe:
- the impact of cycle slips on a carrier phase-based solution
- compare this solution to SPP and DGNSS.


