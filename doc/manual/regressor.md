(ch_regressor)=

# PetscRegressor: Regression Solvers

The ``PetscRegressor`` library provides a framework for the scalable solution of
regression and classification problems. Methods are available for

- {any}`sec_regressor_linear`

Note that by regressor, we mean an algorithm or implementation
used to fit a regression model, following notation from machine-learning community.
Regressor here does NOT mean independent (or predictor) variable, as it often does in the
statistics community.

(sec_regressor_usage)=

## Basic Regressor Usage

Here, we introduce a simple example to demonstrate `PetscRegressor` usage.
Please read {any}`sec_regressor_solvers` for more in-depth discussion.
The code presented {any}`below <regressor-ex3>` solves ordinary linear
regressoion problem, with various regularization options.

In the simplest usage of the regressor solver, the user simply needs to
provide target matrix (`Mat`), and a target vector (`Vec`) to fit
the regressor against. With fitted regressor, then the user can obtain
predicted value vector.

PETSc's default method for solving regression problem is ordinary least squares,
`REGRESSOR_LINEAR_OLS`, which is a sub-type of linear regressor,
`PETSCREGRESSORLINEAR`.

Note that data creation, option parsings, and cleaning stages are omiited for
display purposes. The complete code is available in {ref}`ex3.c <regressor-ex3>`.

(regressor-ex3)=
:::{admonition} Listing: `src/ml/regressor/tests/ex3.c`
```{literalinclude} /../src/ml/regressor/tests/ex3.c
:prepend: '#include <petscregressor.h>'
:start-at: int main
:end-at: PetscFinalize
:append: return 0;}
```
:::

To create a `PetscRegressor` solver, one must first call `PetscRegressorCreate()`
as follows:

```
PetscRegressorCreate(MPI_Comm comm, PetscRegressor *regressor);
```

To choose a solver type, the user can either call

```
PetscRegressorSetType(PetscRegressor regressor, PetscRegressorType type);
```

or use the option `-regressor_type <method>`, where details regarding the
available methods are presented in {any}`sec_regressor_solvers`.
The application code can take complete control of the linear and nonlinear
techniques used in the Newton-like method by calling

```
PetscRegressorSetFromOptions(regressor);
```

This routine provides an interface to the PETSc options database, so
that at runtime the user can select a particular regression solver, set
various parameters and customized routines. With this routine the user
can also control all inner solver options in the `KSP`, and `Tao`
modules, as discussed in {any}`ch_ksp`, {any}`ch_tao`.

After having set these routines and options, the user can fit the problem
by calling

```
PetscRegressorFit(PetscRegressor regressor, Mat X, Vec y);
```

where `X` is training data, and `y` is target values.
Finally, after fitting the regressor solver, the user can compute
prediction, that is, perform inference, using a fitted regressor.

```
PetscRegressorPredict(PetscRegressor regressor, Mat X, Vec y_predicted);
```

Finally, after the user is done using predicting the regressor solver,
the user should destroy the `PetscRegressor` context with

```
PetscRegressorDestroy(PetscRegressor *regressor);
```

Note that the user should not destroy `y_predicted` from previous section,
as this is done internally.

(sec_regressor_solvers)=

## Regression Solvers

One can see the list of regressor solver types in Table
{any}`tab-regressordefaults`. Currently, we only support one type,
`PETSCREGRESSORLINEAR`.

```{eval-rst}
.. list-table:: PETSc Regressor
   :name: tab-regressordefaults
   :header-rows: 1

   * - Method
     - PetscRegressorType
     - Options Name
   * - Linear
     - ``PETSCREGRESSORLINEAR``
     - ``linear``
```

If the particular method that the user is using supports regularizer,
the user can set regularizer's weight via

```
PetscRegressorSetRegularizerWeight(PetscRegressor regressor, PetscReal weight);
```

or with the option `-regresor_regularizer_weight <weight>`.

(sec_regressor_linear)=

## Linear regressor

The method `PETSCREGRESSORLINEAR` (`-regressor_type linear`)
constructs a linear model to reduce the sum of squared differences
between the actual target values in the dataset and the target
values estimated by the linear approximation. By default,
this method will use bound-constrained regularized Gauss-Newton
`TAOBRGN` to solve the regression problem.

Currently, linear regressor has three types, which are described
in Table {any}`tab-lineartypes`.

```{eval-rst}
.. list-table:: Linear Regressor types
   :name: tab-lineartypes
   :header-rows: 1

   * - Linear method
     - ``PetscRegressorLinearType``
     - Options Name
   * - Ordinary
     - ``REGRESSOR_LINEAR_OLS``
     - ``ols``
   * - Lasso
     - ``REGRESSOR_LINEAR_LASSO``
     - ``lasso``
   * - Ridge
     - ``REGRESSOR_LINEAR_RIDGE``
     - ``ridge``
```

If one wishes, the user can (when appropriate) use `KSP` to solve the problem, instead of `Tao`,
via

```
PetscRegressorLinearSetUseKSP(PetscRegressor regressor, PetscBool flg);
```

or with the option `-regressor_linear_use_ksp <true,false>`.

The user can also compute the intercept, also known as the bias or offset), via

```
PetscRegressorLinearSetFitIntercept(PetscRegressor regressor, PetscBool flg);
```

or with the option `-regressor_linear_fit_intercept <true,false>`.

After the regressor has been fitted and predicted, one can obtain intercept and
a vector of the fitted coefficients from a linear regression model.

```
PetscRegressorLinearGetCoefficients(PetscRegressor regressor, Vec *coefficients);
PetscRegressorLinearGetIntercept(PetscRegressor regressor, PetscScalar *intercept);
```
