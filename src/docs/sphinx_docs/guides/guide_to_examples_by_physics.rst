============================================
Guide to PETSc Tutorial Examples, by Physics
============================================
.. highlight:: none

Below we list examples which simulate particular physics problems so that users interested in a particular set of governing equations can easily locate a relevant example. Often PETSc will have several examples looking at the same physics using different numerical tools, such as different discretizations, meshing strategy, closure model, or parameter regime.


Poisson
=======

The Poisson equation

.. math::

  -\Delta u = f

is used to model electrostatics, steady-state diffusion, and other physical processes. Many PETSc examples solve this equation.

  Finite Difference
    :2D: `SNES example 5 <https://www.mcs.anl.gov/petsc/petsc-current/src/snes/tutorials/ex5.c.html>`_
    :3D: `KSP example 45 <https://www.mcs.anl.gov/petsc/petsc-current/src/ksp/ksp/tutorials/ex45.c.html>`_

  Finite Element
    :2D: `SNES example 12 <https://www.mcs.anl.gov/petsc/petsc-current/src/snes/tutorials/ex12.c.html>`_
    :3D: `SNES example 12 <https://www.mcs.anl.gov/petsc/petsc-current/src/snes/tutorials/ex12.c.html>`_

Elastostatics
=============

The equation for elastostatics balances body forces against stresses in the body

.. math::

  -\nabla\cdot \bm \sigma = \bm f

where :math:`\bm\sigma` is the stress tensor. Linear, isotropic elasticity governing infinitesimal strains has the particular stress-strain relation

.. math::

  -\nabla\cdot \left( \lambda I \operatorname{trace}(\bm\varepsilon) + 2\mu \bm\varepsilon \right) = \bm f

where the strain tensor :math:`\bm \varepsilon` is given by

.. math::

  \bm \varepsilon = \frac{1}{2} \left(\nabla \bm u + (\nabla \bm u)^T \right)

where :math:`\bm u` is the infinitesimal displacement of the body.

Finite Element
  :2D: `SNES example 17 <https://www.mcs.anl.gov/petsc/petsc-current/src/snes/tutorials/ex17.c.html>`_
  :3D: `SNES example 17 <https://www.mcs.anl.gov/petsc/petsc-current/src/snes/tutorials/ex17.c.html>`_
  :3D: `SNES example 56 <https://www.mcs.anl.gov/petsc/petsc-current/src/snes/tutorials/ex56.c.html>`_

If we allow finite strains in the body, we can express the stress-strain relation in terms of the Jacobian of the deformation gradient

.. math::

  J = \mathrm{det}(F) = \mathrm{det}\left(\nabla u\right)

and the right Cauchy-Green deformation tensor

.. math::

  C = F^T F

so that

.. math::

  \frac{\mu}{2} \left( \mathrm{Tr}(C) - 3 \right) + J p + \frac{\kappa}{2} (J - 1) = 0

In the example itself, everything can be expressed in terms of determinants and cofactors of :math:`F`.

  Finite Element
    :3D: `SNES example 77 <https://www.mcs.anl.gov/petsc/petsc-current/src/snes/tutorials/ex77.c.html>`_


Stokes
======

The Stokes equations

.. math::

    -\frac{\mu}{2} \left(\nabla u + \nabla u^T \right) + \nabla p + f = 0

.. math::

    \nabla\cdot u = 0

describe slow flow of an incompressible fluid with velocity :math:`u`, pressure :math:`p`, and body force :math:`f`.

  Finite Element
    :2D: `SNES example 62 <https://www.mcs.anl.gov/petsc/petsc-current/src/snes/tutorials/ex62.c.html>`_
    :3D: `SNES example 62 <https://www.mcs.anl.gov/petsc/petsc-current/src/snes/tutorials/ex62.c.html>`_

    See :doc:`physics/guide_to_stokes` for more.

Euler
=====

Heat equation
=============

The heat equation

.. math::

  \frac{\partial u}{\partial t} - \Delta u = f

is used to model heat flow, time-dependent diffusion, and other physical processes.

  Finite Element
    :2D: `TS example 45 <https://www.mcs.anl.gov/petsc/petsc-current/src/ts/tutorials/ex45.c.html>`_
    :3D: `TS example 45 <https://www.mcs.anl.gov/petsc/petsc-current/src/ts/tutorials/ex45.c.html>`_

Navier-Stokes
=============

The incompressible Navier-Stokes equations

.. math::

    \frac{\partial u}{\partial t} + u\cdot\nabla u - \frac{\mu}{2} \left(\nabla u + \nabla u^T\right) + \nabla p + f = 0

.. math::

    \nabla\cdot u = 0

are appropriate for flow of an incompressible fluid at low to moderate Reynolds number.

  Finite Element
    :2D: `TS example 46 <https://www.mcs.anl.gov/petsc/petsc-current/src/ts/tutorials/ex46.c.html>`_
    :3D: `TS example 46 <https://www.mcs.anl.gov/petsc/petsc-current/src/ts/tutorials/ex46.c.html>`_
