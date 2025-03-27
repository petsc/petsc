# Tutorials, by Physics

```{highlight} none
```

Below we list examples which simulate particular physics problems so that users interested in a particular set of governing equations can easily locate a relevant example. Often PETSc will have several examples looking at the same physics using different numerical tools, such as different discretizations, meshing strategy, closure model, or parameter regime.

## Poisson

The Poisson equation

$$
-\Delta u = f
$$

is used to model electrostatics, steady-state diffusion, and other physical processes. Many PETSc examples solve this equation.

> Finite Difference
> : ```{eval-rst}
>
>   :2D: `SNES example 5 <PETSC_DOC_OUT_ROOT_PLACEHOLDER/src/snes/tutorials/ex5.c.html>`_
>   :3D: `KSP example 45 <PETSC_DOC_OUT_ROOT_PLACEHOLDER/src/ksp/ksp/tutorials/ex45.c.html>`_
>   ```
>
> Finite Element
> : ```{eval-rst}
>
>   :2D: `SNES example 12 <PETSC_DOC_OUT_ROOT_PLACEHOLDER/src/snes/tutorials/ex12.c.html>`_
>   :3D: `SNES example 12 <PETSC_DOC_OUT_ROOT_PLACEHOLDER/src/snes/tutorials/ex12.c.html>`_
>   ```

## Elastostatics

The equation for elastostatics balances body forces against stresses in the body

$$
-\nabla\cdot \bm \sigma = \bm f
$$

where $\bm\sigma$ is the stress tensor. Linear, isotropic elasticity governing infinitesimal strains has the particular stress-strain relation

$$
-\nabla\cdot \left( \lambda I \operatorname{trace}(\bm\varepsilon) + 2\mu \bm\varepsilon \right) = \bm f
$$

where the strain tensor $\bm \varepsilon$ is given by

$$
\bm \varepsilon = \frac{1}{2} \left(\nabla \bm u + (\nabla \bm u)^T \right)
$$

where $\bm u$ is the infinitesimal displacement of the body. The resulting discretizations use PETSc's nonlinear solvers

Finite Element
: ```{eval-rst}

  :2D: `SNES example 17 <PETSC_DOC_OUT_ROOT_PLACEHOLDER/src/snes/tutorials/ex17.c.html>`_
  :3D: `SNES example 17 <PETSC_DOC_OUT_ROOT_PLACEHOLDER/src/snes/tutorials/ex17.c.html>`_
  :3D: `SNES example 56 <PETSC_DOC_OUT_ROOT_PLACEHOLDER/src/snes/tutorials/ex56.c.html>`_
  ```

If we allow finite strains in the body, we can express the stress-strain relation in terms of the Jacobian of the deformation gradient

$$
J = \mathrm{det}(F) = \mathrm{det}\left(\nabla u\right)
$$

and the right Cauchy-Green deformation tensor

$$
C = F^T F
$$

so that

$$
\frac{\mu}{2} \left( \mathrm{Tr}(C) - 3 \right) + J p + \frac{\kappa}{2} (J - 1) = 0
$$

In the example everything is expressed in terms of determinants and cofactors of $F$.

> Finite Element
> :

## Stokes

{doc}`physics/guide_to_stokes`

## Euler

Not yet developed

## Heat equation

The time-dependent heat equation

$$
\frac{\partial u}{\partial t} - \Delta u = f
$$

is used to model heat flow, time-dependent diffusion, and other physical processes.

> Finite Element
> : ```{eval-rst}
>
>   :2D: `TS example 45 <PETSC_DOC_OUT_ROOT_PLACEHOLDER/src/ts/tutorials/ex45.c.html>`_
>   :3D: `TS example 45 <PETSC_DOC_OUT_ROOT_PLACEHOLDER/src/ts/tutorials/ex45.c.html>`_
>   ```

## Navier-Stokes

The time-dependent incompressible Navier-Stokes equations

$$
\begin{aligned}
\frac{\partial u}{\partial t} + u\cdot\nabla u - \nabla \cdot \left(\mu \left(\nabla u + \nabla u^T\right)\right) + \nabla p + f &= 0 \\
\nabla\cdot u &= 0 \end{aligned}
$$

are appropriate for flow of an incompressible fluid at low to moderate Reynolds number.

> Finite Element
> : ```{eval-rst}
>
>   :2D: `TS example 46 <PETSC_DOC_OUT_ROOT_PLACEHOLDER/src/ts/tutorials/ex46.c.html>`_
>   :3D: `TS example 46 <PETSC_DOC_OUT_ROOT_PLACEHOLDER/src/ts/tutorials/ex46.c.html>`_
>   ```
