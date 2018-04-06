#!/usr/bin/env python
import numpy as np
import ex13

Nf     = 1
dofs   = []
times  = []
flops  = []
errors = []
for f in range(Nf): errors.append([])
level  = 0
while level >= 0:
  stageName = "ConvEst Refinement Level "+str(level)
  if stageName in ex13.Stages:
    dofs.append(ex13.Stages[stageName]["ConvEst Error"][0]["dof"])
    times.append(ex13.Stages[stageName]["SNESSolve"][0]["time"])
    flops.append(ex13.Stages[stageName]["SNESSolve"][0]["flop"])
    for f in range(Nf): errors[f].append(ex13.Stages[stageName]["ConvEst Error"][0]["error"][f])
    level = level + 1
  else:
    level = -1

dofs   = np.array(dofs)
times  = np.array(times)
flops  = np.array(flops)
errors = np.array(errors)
print dofs
print times
print flops
print errors

import matplotlib.pyplot as plt

plt.title('Mesh Convergence')
plt.xlabel('Problem Size $\log N$')
plt.ylabel('Error $\log |x - x^*|$')
plt.loglog(dofs, errors[0])
plt.show()

plt.title('Static Scaling')
plt.xlabel('Time (s)')
plt.ylabel('Flop Rate (F/s)')
plt.loglog(times, flops/times)
plt.show()

plt.title('Efficacy')
plt.xlabel('Time (s)')
plt.ylabel('Action (s)')
plt.loglog(times, errors[0]*times)
plt.show()
