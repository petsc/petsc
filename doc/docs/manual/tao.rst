.. _chapter_tao:

TAO: Optimization Solvers
=========================

The Toolkit for Advanced Optimization (TAO) focuses on algorithms for the
solution of large-scale optimization problems on high-performance
architectures.  Methods are available for unconstrained, bound-constrained and
generally constrained optimization, nonlinear least squares problems,
variational inequalities and complementarity constraints.

.. _sec_tao_getting_started:

Getting Started: A Simple TAO Example
-------------------------------------

To help the user start using TAO immediately, we introduce here a simple
uniprocessor example. Please read :any:`sec_tao_solvers`
for a more in-depth discussion on using the TAO solvers. The code
presented :any:`below <tao-example1>` minimizes the
extended Rosenbrock function :math:`f: \mathbb R^n \to \mathbb R`
defined by

.. math::

   f(x) = \sum_{i=0}^{m-1} \left( \alpha(x_{2i+1}-x_{2i}^2)^2 + (1-x_{2i})^2 \right),

where :math:`n = 2m` is the number of variables. Note that while we use
the C language to introduce the TAO software, the package is fully
usable from C++ and Fortran 77/90.
:any:`chapter_fortran` discusses additional
issues concerning Fortran usage.

The code in :any:`the example <tao-example1>` contains many of
the components needed to write most TAO programs and thus is
illustrative of the features present in complex optimization problems.
Note that for display purposes we have omitted some nonessential lines
of code as well as the (essential) code required for the routine
``FormFunctionGradient``, which evaluates the function and gradient, and
the code for ``FormHessian``, which evaluates the Hessian matrix for
Rosenbrock’s function. The complete code is available in
`$TAO_DIR/src/unconstrained/tutorials/rosenbrock1.c <../../src/tao/unconstrained/tutorials/rosenbrock1.c.html>`__.
The following sections annotate the lines of code in
:any:`the example <tao-example1>`.

.. _tao-example1:
.. admonition:: Listing: ``src/tao/unconstrained/tutorials/rosenbrock1.c``

   .. literalinclude:: /../src/tao/unconstrained/tutorials/rosenbrock1.c
      :prepend: #include <petsctao.h>
      :start-at: typedef struct
      :end-at: PetscFinalize
      :append: return ierr;}

.. _sec_tao_workflow:

TAO Workflow
------------

Many TAO applications will follow an ordered set of procedures for
solving an optimization problem: The user creates a ``Tao`` context and
selects a default algorithm. Call-back routines as well as vector
(``Vec``) and matrix (``Mat``) data structures are then set. These
call-back routines will be used for evaluating the objective function,
gradient, and perhaps the Hessian matrix. The user then invokes TAO to
solve the optimization problem and finally destroys the ``Tao`` context.
A list of the necessary functions for performing these steps using TAO
is shown below.

.. code::

      TaoCreate(MPI_Comm comm, Tao *tao);
      TaoSetType(Tao tao, TaoType type);
      TaoSetSolution(Tao tao, Vec x);
      TaoSetObjectiveAndGradient(Tao tao, Vec g,
           PetscErrorCode (*FormFGradient)(Tao,Vec,PetscReal*,Vec,void*),
           void *user);
      TaoSetHessian(Tao tao, Mat H, Mat Hpre,
           PetscErrorCode (*FormHessian)(Tao,Vec,Mat,Mat,
           void*), void *user);
      TaoSolve(Tao tao);
      TaoDestroy(Tao tao);

Note that the solver algorithm selected through the function
``TaoSetType()`` can be overridden at runtime by using an options
database. Through this database, the user not only can select a
minimization method (e.g., limited-memory variable metric, conjugate
gradient, Newton with line search or trust region) but also can
prescribe the convergence tolerance, set various monitoring routines,
set iterative methods and preconditions for solving the linear systems,
and so forth. See :any:`sec_tao_solvers` for more
information on the solver methods available in TAO.

Header File
~~~~~~~~~~~

TAO applications written in C/C++ should have the statement

.. code::

      #include <petsctao.h>

in each file that uses a routine in the TAO libraries.

Creation and Destruction
~~~~~~~~~~~~~~~~~~~~~~~~

A TAO solver can be created by calling the

.. code::

      TaoCreate(MPI_Comm, Tao*);

routine. Much like creating PETSc vector and matrix objects, the first
argument is an MPI *communicator*. An MPI [#mpi]_
communicator indicates a collection of processors that will be used to
evaluate the objective function, compute constraints, and provide
derivative information. When only one processor is being used, the
communicator ``PETSC_COMM_SELF`` can be used with no understanding of
MPI. Even parallel users need to be familiar with only the basic
concepts of message passing and distributed-memory computing. Most
applications running TAO in parallel environments can employ the
communicator ``PETSC_COMM_WORLD`` to indicate all processes known to
PETSc in a given run.

The routine

.. code::

      TaoSetType(Tao, TaoType);

can be used to set the algorithm TAO uses to solve the application. The
various types of TAO solvers and the flags that identify them will be
discussed in the following sections. The solution method should be
carefully chosen depending on the problem being solved. Some solvers,
for instance, are meant for problems with no constraints, whereas other
solvers acknowledge constraints in the problem and handle them
accordingly. The user must also be aware of the derivative information
that is available. Some solvers require second-order information, while
other solvers require only gradient or function information. The command
line option ``-tao_type`` followed by
a TAO method will override any method specified by the second argument.
The command line option ``-tao_type bqnls``, for instance, will
specify the limited-memory quasi-Newton line search method for
bound-constrained problems. Note that the ``TaoType`` variable is a string that
requires quotation marks in an application program, but quotation marks
are not required at the command line.

Each TAO solver that has been created should also be destroyed by using
the

.. code::

      TaoDestroy(Tao tao);

command. This routine frees the internal data structures used by the
solver.

Command-line Options
~~~~~~~~~~~~~~~~~~~~

Additional options for the TAO solver can be be set from the command
line by using the

.. code::

      TaoSetFromOptions(Tao)

routine. This command also provides information about runtime options
when the user includes the ``-help`` option on the command line.

In addition to common command line options shared by all TAO solvers, each TAO
method also implements its own specialized options. Please refer to the
documentation for individual methods for more details.

Defining Variables
~~~~~~~~~~~~~~~~~~

In all the optimization solvers, the application must provide a ``Vec``
object of appropriate dimension to represent the variables. This vector
will be cloned by the solvers to create additional work space within the
solver. If this vector is distributed over multiple processors, it
should have a parallel distribution that allows for efficient scaling,
inner products, and function evaluations. This vector can be passed to
the application object by using the

.. code::

      TaoSetSolution(Tao, Vec);

routine. When using this routine, the application should initialize the
vector with an approximate solution of the optimization problem before
calling the TAO solver. This vector will be used by the TAO solver to
store the solution. Elsewhere in the application, this solution vector
can be retrieved from the application object by using the

.. code::

      TaoGetSolution(Tao, Vec*);

routine. This routine takes the address of a ``Vec`` in the second
argument and sets it to the solution vector used in the application.

User Defined Call-back Routines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Users of TAO are required to provide routines that perform function
evaluations. Depending on the solver chosen, they may also have to write
routines that evaluate the gradient vector and Hessian matrix.

Application Context
^^^^^^^^^^^^^^^^^^^

Writing a TAO application may require use of an *application context*.
An application context is a structure or object defined by an
application developer, passed into a routine also written by the
application developer, and used within the routine to perform its stated
task.

For example, a routine that evaluates an objective function may need
parameters, work vectors, and other information. This information, which
may be specific to an application and necessary to evaluate the
objective, can be collected in a single structure and used as one of the
arguments in the routine. The address of this structure will be cast as
type ``(void*)`` and passed to the routine in the final argument. Many
examples of these structures are included in the TAO distribution.

This technique offers several advantages. In particular, it allows for a
uniform interface between TAO and the applications. The fundamental
information needed by TAO appears in the arguments of the routine, while
data specific to an application and its implementation is confined to an
opaque pointer. The routines can access information created outside the
local scope without the use of global variables. The TAO solvers and
application objects will never access this structure, so the application
developer has complete freedom to define it. If no such structure or
needed by the application then a NULL pointer can be used.

.. _sec_fghj:

Objective Function and Gradient Routines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TAO solvers that minimize an objective function require the application
to evaluate the objective function. Some solvers may also require the
application to evaluate derivatives of the objective function. Routines
that perform these computations must be identified to the application
object and must follow a strict calling sequence.

Routines should follow the form

.. code::

      PetscErrorCode EvaluateObjective(Tao, Vec, PetscReal*, void*);

in order to evaluate an objective function
:math:`f: \, \mathbb R^n \to \mathbb R`. The first argument is the TAO
Solver object, the second argument is the :math:`n`-dimensional vector
that identifies where the objective should be evaluated, and the fourth
argument is an application context. This routine should use the third
argument to return the objective value evaluated at the point specified
by the vector in the second argument.

This routine, and the application context, should be passed to the
application object by using the

.. code::

      TaoSetObjective(Tao, PetscErrorCode(*)(Tao,Vec,PetscReal*,void*), void*);

routine. The first argument in this routine is the TAO solver object,
the second argument is a function pointer to the routine that evaluates
the objective, and the third argument is the pointer to an appropriate
application context. Although the final argument may point to anything,
it must be cast as a ``(void*)`` type. This pointer will be passed back
to the developer in the fourth argument of the routine that evaluates
the objective. In this routine, the pointer can be cast back to the
appropriate type. Examples of these structures and their usage are
provided in the distribution.

Many TAO solvers also require gradient information from the application
The gradient of the objective function is specified in a similar manner.
Routines that evaluate the gradient should have the calling sequence

.. code::

      PetscErrorCode EvaluateGradient(Tao, Vec, Vec, void*);

where the first argument is the TAO solver object, the second argument
is the variable vector, the third argument is the gradient vector, and
the fourth argument is the user-defined application context. Only the
third argument in this routine is different from the arguments in the
routine for evaluating the objective function. The numbers in the
gradient vector have no meaning when passed into this routine, but they
should represent the gradient of the objective at the specified point at
the end of the routine. This routine, and the user-defined pointer, can
be passed to the application object by using the

.. code::

      TaoSetGradient(Tao, Vec, PetscErrorCode (*)(Tao,Vec,Vec,void*), void*);

routine. In this routine, the first argument is the Tao object, the second
argument is the optional vector to hold the computed gradient, the
third argument is the function pointer, and the fourth object is the
application context, cast to ``(void*)``.

Instead of evaluating the objective and its gradient in separate
routines, TAO also allows the user to evaluate the function and the
gradient in the same routine. In fact, some solvers are more efficient
when both function and gradient information can be computed in the same
routine. These routines should follow the form

.. code::

      PetscErrorCode EvaluateFunctionAndGradient(Tao, Vec, PetscReal*, Vec, void*);

where the first argument is the TAO solver and the second argument
points to the input vector for use in evaluating the function and
gradient. The third argument should return the function value, while the
fourth argument should return the gradient vector. The fifth argument is
a pointer to a user-defined context. This context and the name of the
routine should be set with the call

.. code::

      TaoSetObjectiveAndGradient(Tao, Vec PetscErrorCode (*)(Tao,Vec,PetscReal*,Vec,void*), void*);

where the arguments are the TAO application, the optional vector to be
used to hold the computed gradient, a function pointer, and a
pointer to a user-defined context.

The TAO example problems demonstrate the use of these application
contexts as well as specific instances of function, gradient, and
Hessian evaluation routines. All these routines should return the
integer :math:`0` after successful completion and a nonzero integer if
the function is undefined at that point or an error occurred.

.. _sec_tao_matrixfree:

Hessian Evaluation
^^^^^^^^^^^^^^^^^^

Some optimization routines also require a Hessian matrix from the user.
The routine that evaluates the Hessian should have the form

.. code::

      PetscErrorCode EvaluateHessian(Tao, Vec, Mat, Mat, void*);

where the first argument of this routine is a TAO solver object. The
second argument is the point at which the Hessian should be evaluated.
The third argument is the Hessian matrix, and the sixth argument is a
user-defined context. Since the Hessian matrix is usually used in
solving a system of linear equations, a preconditioner for the matrix is
often needed. The fourth argument is the matrix that will be used for
preconditioning the linear system; in most cases, this matrix will be
the same as the Hessian matrix. The fifth argument is the flag used to
set the Hessian matrix and linear solver in the routine
``KSPSetOperators()``.

One can set the Hessian evaluation routine by calling the

.. code::

      TaoSetHessian(Tao, Mat, Mat, PetscErrorCode (*)(Tao,Vec,Mat,Mat,void*), void*);

routine. The first argument is the TAO Solver object. The second and
third arguments are, respectively, the Mat object where the Hessian will
be stored and the Mat object that will be used for the preconditioning
(they may be the same). The fourth argument is the function that
evaluates the Hessian, and the fifth argument is a pointer to a
user-defined context, cast to ``(void*)``.

Finite Differences
""""""""""""""""""

Finite-difference approximations can be used to compute the gradient and
the Hessian of an objective function. These approximations will slow the
solve considerably and are recommended primarily for checking the
accuracy of hand-coded gradients and Hessians. These routines are

.. code::

      TaoDefaultComputeGradient(Tao, Vec, Vec, void*);

and

.. code::

      TaoDefaultComputeHessian(Tao, Vec, Mat*, Mat*,void*);

respectively. They can be set by using ``TaoSetGradient()`` and
``TaoSetHessian()`` or through the options database with the
options ``-tao_fdgrad`` and ``-tao_fd``, respectively.

The efficiency of the finite-difference Hessian can be improved if the
coloring of the matrix is known. If the application programmer creates a
PETSc ``MatFDColoring`` object, it can be applied to the
finite-difference approximation by setting the Hessian evaluation
routine to

.. code::

      TaoDefaultComputeHessianColor(Tao, Vec, Mat*, Mat*, void*);

and using the ``MatFDColoring`` object as the last (``void *``) argument
to ``TaoSetHessian()``.

One also can use finite-difference approximations to directly check the
correctness of the gradient and/or Hessian evaluation routines. This
process can be initiated from the command line by using the special TAO
solver ``tao_fd_test`` together with the option ``-tao_test_gradient``
or ``-tao_test_hessian``.

Matrix-Free Methods
"""""""""""""""""""

TAO fully supports matrix-free methods. The matrices specified in the
Hessian evaluation routine need not be conventional matrices; instead,
they can point to the data required to implement a particular
matrix-free method. The matrix-free variant is allowed *only* when the
linear systems are solved by an iterative method in combination with no
preconditioning (``PCNONE`` or ``-pc_type none``), a user-provided
preconditioner matrix, or a user-provided preconditioner shell
(``PCSHELL``). In other words, matrix-free methods cannot be used if a
direct solver is to be employed. Details about using matrix-free methods
are provided in the :doc:`/docs/manual/index`.

.. _sec_bounds:

Constraints
^^^^^^^^^^^

Some optimization problems also impose constraints on the variables or
intermediate application states. The user defines these constraints through
the appropriate TAO interface functions and call-back routines where necessary.

Variable Bounds
"""""""""""""""

The simplest type of constraint on an optimization problem puts lower or
upper bounds on the variables. Vectors that represent lower and upper
bounds for each variable can be set with the

.. code::

      TaoSetVariableBounds(Tao, Vec, Vec);

command. The first vector and second vector should contain the lower and
upper bounds, respectively. When no upper or lower bound exists for a
variable, the bound may be set to ``PETSC_INFINITY`` or ``PETSC_NINFINITY``.
After the two bound vectors have been set, they may be accessed with the
command ``TaoGetVariableBounds()``.

Since not all solvers recognize the presence of bound constraints on
variables, the user must be careful to select a solver that acknowledges
these bounds.

.. _sec_tao_programming:

General Constraints
"""""""""""""""""""

Some TAO algorithms also support general constraints as a linear or nonlinear
function of the optimization variables. These constraints can be imposed either
as equalities or inequalities. TAO currently does not make any distinctions
between linear and nonlinear constraints, and implements them through the
same software interfaces.

In the equality constrained case, TAO assumes that the constraints are
formulated as :math:`c_e(x) = 0` and requires the user to implement a call-back
routine for evaluating :math:`c_e(x)` at a given vector of optimization
variables,

.. code::

      PetscErrorCode EvaluateEqualityConstraints(Tao, Vec, Vec, void*);

As in the previous call-back routines, the first argument is the TAO solver
object. The second and third arguments are the vector of optimization variables
(input) and vector of equality constraints (output), respectively. The final
argument is a pointer to the user-defined application context, cast into
``(void*)``.

Generally constrained TAO algorithms also require a second user call-back
function to compute the constraint Jacobian matrix :math:`\nabla_x c_e(x)`,

.. code::

      PetscErrorCode EvaluateEqualityJacobian(Tao, Vec, Mat, Mat, void*);

where the first and last arguments are the TAO solver object and the application
context pointer as before. The second argument is the vector of optimization
variables at which the computation takes place. The third and fourth arguments
are the constraint Jacobian and its pseudo-inverse (optional), respectively. The
pseudoinverse is optional, and if not available, the user can simply set it
to the constraint Jacobian itself.

These call-back functions are then given to the TAO solver using the
interface functions

.. code::

      TaoSetEqualityConstraintsRoutine(Tao, Vec, PetscErrorCode (*)(Tao,Vec,Vec,void*), void*);

and

.. code::

      TaoSetJacobianEqualityRoutine(Tao, Mat, Mat, PetscErrorCode (*)(Tao,Vec,Mat,Mat,void*), void*);

Inequality constraints are assumed to be formulated as :math:`c_i(x) \leq 0`
and follow the same workflow as equality constraints using the
``TaoSetInequalityConstraintsRoutine()`` and ``TaoSetJacobianInequalityRoutine()``
interfaces.

Some TAO algorithms may adopt an alternative double-sided
:math:`c_l \leq c_i(x) \leq c_u` formulation and require the lower and upper
bounds :math:`c_l` and :math:`c_u` to be set using the
``TaoSetInequalityBounds(Tao,Vec,Vec)`` interface. Please refer to the
documentation for each TAO algorithm for further details.

Solving
~~~~~~~

Once the application and solver have been set up, the solver can be

.. code::

      TaoSolve(Tao);

routine. We discuss several universal options below.

.. _sec_customize:

Convergence
^^^^^^^^^^^

Although TAO and its solvers set default parameters that are useful for
many problems, the user may need to modify these parameters in order to
change the behavior and convergence of various algorithms.

One convergence criterion for most algorithms concerns the number of
digits of accuracy needed in the solution. In particular, the
convergence test employed by TAO attempts to stop when the error in the
constraints is less than :math:`\epsilon_{crtol}` and either

.. math::

   \begin{array}{lcl}
   ||g(X)|| &\leq& \epsilon_{gatol}, \\
   ||g(X)||/|f(X)| &\leq& \epsilon_{grtol}, \quad \text{or} \\
   ||g(X)||/|g(X_0)| &\leq& \epsilon_{gttol},
   \end{array}

where :math:`X` is the current approximation to the true solution
:math:`X^*` and :math:`X_0` is the initial guess. :math:`X^*` is
unknown, so TAO estimates :math:`f(X) - f(X^*)` with either the square
of the norm of the gradient or the duality gap. A relative tolerance of
:math:`\epsilon_{frtol}=0.01` indicates that two significant digits are
desired in the objective function. Each solver sets its own convergence
tolerances, but they can be changed by using the routine
``TaoSetTolerances()``. Another set of convergence tolerances terminates
the solver when the norm of the gradient function (or Lagrangian
function for bound-constrained problems) is sufficiently close to zero.

Other stopping criteria include a minimum trust-region radius or a
maximum number of iterations. These parameters can be set with the
routines ``TaoSetTrustRegionTolerance()`` and
``TaoSetMaximumIterations()`` Similarly, a maximum number of function
evaluations can be set with the command
``TaoSetMaximumFunctionEvaluations()``. ``-tao_max_it``, and
``-tao_max_funcs``.

Viewing Status
^^^^^^^^^^^^^^

To see parameters and performance statistics for the solver, the routine

.. code::

      TaoView(Tao tao)

can be used. This routine will display to standard output the number of
function evaluations need by the solver and other information specific
to the solver. This same output can be produced by using the command
line option ``-tao_view``.

The progress of the optimization solver can be monitored with the
runtime option ``-tao_monitor``. Although monitoring routines can be
customized, the default monitoring routine will print out several
relevant statistics to the screen.

The user also has access to information about the current solution. The
current iteration number, objective function value, gradient norm,
infeasibility norm, and step length can be retrieved with the following
command.

.. code::

      TaoGetSolutionStatus(Tao tao, PetscInt* iterate, PetscReal* f,
                        PetscReal* gnorm, PetscReal* cnorm, PetscReal* xdiff,
                        TaoConvergedReason* reason)

The last argument returns a code that indicates the reason that the
solver terminated. Positive numbers indicate that a solution has been
found, while negative numbers indicate a failure. A list of reasons can
be found in the manual page for ``TaoGetConvergedReason()``.

Obtaining a Solution
^^^^^^^^^^^^^^^^^^^^

After exiting the ``TaoSolve()`` function, the solution and the gradient can be
recovered with the following routines.

.. code::

      TaoGetSolution(Tao, Vec*);
      TaoGetGradient(Tao, Vec*, NULL, NULL);

Note that the ``Vec`` returned by ``TaoGetSolution()`` will be the
same vector passed to ``TaoSetSolution()``. This information can be
obtained during user-defined routines such as a function evaluation and
customized monitoring routine or after the solver has terminated.

Special Problem structures
~~~~~~~~~~~~~~~~~~~~~~~~~~

Certain special classes of problems solved with TAO utilize specialized
code interfaces that are described below per problem type.

PDE-constrained Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TAO solves PDE-constrained optimization problems of the form

.. math::

   \begin{array}{ll}
   \displaystyle \min_{u,v} & f(u,v) \\
   \text{subject to} & g(u,v) = 0,
   \end{array}

where the state variable :math:`u` is the solution to the discretized
partial differential equation defined by :math:`g` and parametrized by
the design variable :math:`v`, and :math:`f` is an objective function.
The Lagrange multipliers on the constraint are denoted by :math:`y`.
This method is set by using the linearly constrained augmented
Lagrangian TAO solver ``tao_lcl``.

We make two main assumptions when solving these problems: the objective
function and PDE constraints have been discretized so that we can treat
the optimization problem as finite dimensional and
:math:`\nabla_u g(u,v)` is invertible for all :math:`u` and :math:`v`.

Unlike other TAO solvers where the solution vector contains only the
optimization variables, PDE-constrained problems solved with ``tao_lcl``
combine the design and state variables together in a monolithic solution vector
:math:`x^T = [u^T, v^T]`. Consequently, the user must provide index sets to
separate the two,

.. code::

      TaoSetStateDesignIS(Tao, IS, IS);

where the first IS is a PETSc IndexSet containing the indices of the
state variables and the second IS the design variables.

PDE constraints have the general form :math:`g(x) = 0`,
where :math:`c: \mathbb R^n \to \mathbb R^m`. These constraints should
be specified in a routine, written by the user, that evaluates
:math:`g(x)`. The routine that evaluates the constraint equations
should have the form

.. code::

      PetscErrorCode EvaluateConstraints(Tao, Vec, Vec, void*);

The first argument of this routine is a TAO solver object. The second
argument is the variable vector at which the constraint function should
be evaluated. The third argument is the vector of function values
:math:`g(x)`, and the fourth argument is a pointer to a user-defined
context. This routine and the user-defined context should be set in the
TAO solver with the

.. code::

      TaoSetConstraintsRoutine(Tao, Vec, PetscErrorCode (*)(Tao,Vec,Vec,void*), void*);

command. In this function, the first argument is the TAO solver object,
the second argument a vector in which to store the constraints, the
third argument is a function point to the routine for evaluating the
constraints, and the fourth argument is a pointer to a user-defined
context.

The Jacobian of :math:`g(x)` is the matrix in
:math:`\mathbb R^{m \times n}` such that each column contains the
partial derivatives of :math:`g(x)` with respect to one variable. The
evaluation of the Jacobian of :math:`g` should be performed by calling
the

.. code::

      PetscErrorCode JacobianState(Tao, Vec, Mat, Mat, Mat, void*);
      PetscErrorCode JacobianDesign(Tao, Vec, Mat*, void*);

routines. In these functions, The first argument is the TAO solver
object. The second argument is the variable vector at which to evaluate
the Jacobian matrix, the third argument is the Jacobian matrix, and the
last argument is a pointer to a user-defined context. The fourth and
fifth arguments of the Jacobian evaluation with respect to the state
variables are for providing PETSc matrix objects for the preconditioner
and for applying the inverse of the state Jacobian, respectively. This
inverse matrix may be ``PETSC_NULL``, in which case TAO will use a PETSc
Krylov subspace solver to solve the state system. These evaluation
routines should be registered with TAO by using the

.. code::

      TaoSetJacobianStateRoutine(Tao, Mat, Mat, Mat,
                              PetscErrorCode (*)(Tao,Vec,Mat,Mat,void*),
                              void*);
      TaoSetJacobianDesignRoutine(Tao, Mat,
                              PetscErrorCode (*)(Tao,Vec,Mat*,void*),
                              void*);

routines. The first argument is the TAO solver object, and the second
argument is the matrix in which the Jacobian information can be stored.
For the state Jacobian, the third argument is the matrix that will be
used for preconditioning, and the fourth argument is an optional matrix
for the inverse of the state Jacobian. One can use ``PETSC_NULL`` for
this inverse argument and let PETSc apply the inverse using a KSP
method, but faster results may be obtained by manipulating the structure
of the Jacobian and providing an inverse. The fifth argument is the
function pointer, and the sixth argument is an optional user-defined
context. Since no solve is performed with the design Jacobian, there is
no need to provide preconditioner or inverse matrices.

.. _sec_evalsof:

Nonlinear Least Squares
^^^^^^^^^^^^^^^^^^^^^^^

For nonlinear least squares applications, we are solving the
optimization problem

.. math:: \min_{x} \;\frac{1}{2}||r(x)||_2^2.

For these problems, the objective function value should be computed as a
vector of residuals, :math:`r(x)`, computed with a function of the form

.. code::

      PetscErrorCode EvaluateResidual(Tao, Vec, Vec, void*);

and set with the

.. code::

      TaoSetResidualRoutine(Tao, PetscErrorCode (*)(Tao,Vec,Vec,void*), void*);

routine. If required by the algorithm, the Jacobian of the residual,
:math:`J = \partial r(x) / \partial x`, should be computed with a
function of the form

.. code::

      PetscErrorCode EvaluateJacobian(Tao, Vec, Mat, void*);

and set with the

.. code::

      TaoSetJacobianResidualRoutine(Tao, PetscErrorCode (*)(Tao,Vec,Mat,void*), void *);

routine.

Complementarity
^^^^^^^^^^^^^^^

Complementarity applications have equality constraints in the form of
nonlinear equations :math:`C(X) = 0`, where
:math:`C: \mathbb R^n \to \mathbb R^m`. These constraints should be
specified in a routine written by the user with the form

.. code::

      PetscErrorCode EqualityConstraints(Tao, Vec, Vec, void*);

that evaluates :math:`C(X)`. The first argument of this routine is a TAO
Solver object. The second argument is the variable vector :math:`X` at
which the constraint function should be evaluated. The third argument is
the output vector of function values :math:`C(X)`, and the fourth
argument is a pointer to a user-defined context.

This routine and the user-defined context must be registered with TAO by
using the

.. code::

      TaoSetConstraintRoutine(Tao, Vec, PetscErrorCode (*)(Tao,Vec,Vec,void*), void*);

command. In this command, the first argument is TAO Solver object, the
second argument is vector in which to store the function values, the
third argument is the user-defined routine that evaluates :math:`C(X)`,
and the fourth argument is a pointer to a user-defined context that will
be passed back to the user.

The Jacobian of the function is the matrix in
:math:`\mathbb R^{m \times n}` such that each column contains the
partial derivatives of :math:`f` with respect to one variable. The
evaluation of the Jacobian of :math:`C` should be performed in a routine
of the form

.. code::

      PetscErrorCode EvaluateJacobian(Tao, Vec, Mat, Mat, void*);

In this function, the first argument is the TAO Solver object and the
second argument is the variable vector at which to evaluate the Jacobian
matrix. The third argument is the Jacobian matrix, and the sixth
argument is a pointer to a user-defined context. Since the Jacobian
matrix may be used in solving a system of linear equations, a
preconditioner for the matrix may be needed. The fourth argument is the
matrix that will be used for preconditioning the linear system; in most
cases, this matrix will be the same as the Hessian matrix. The fifth
argument is the flag used to set the Jacobian matrix and linear solver
in the routine ``KSPSetOperators()``.

This routine should be specified to TAO by using the

.. code::

      TaoSetJacobianRoutine(Tao, Mat, Mat, PetscErrorCode (*)(Tao,Vec,Mat,Mat,void*), void*);

command. The first argument is the TAO Solver object; the second and
third arguments are the Mat objects in which the Jacobian will be stored
and the Mat object that will be used for the preconditioning (they may
be the same), respectively. The fourth argument is the function pointer;
and the fifth argument is an optional user-defined context. The Jacobian
matrix should be created in a way such that the product of it and the
variable vector can be stored in the constraint vector.

.. _sec_tao_solvers:

TAO Algorithms
--------------

TAO includes a variety of optimization algorithms for several classes of
problems (unconstrained, bound-constrained, and PDE-constrained
minimization, nonlinear least-squares, and complementarity). The TAO
algorithms for solving these problems are detailed in this section, a
particular algorithm can chosen by using the ``TaoSetType()`` function
or using the command line arguments ``-tao_type <name>``. For those
interested in extending these algorithms or using new ones, please see
:any:`sec_tao_addsolver` for more information.

.. _sec_tao_unconstrained:

Unconstrained Minimization
~~~~~~~~~~~~~~~~~~~~~~~~~~

Unconstrained minimization is used to minimize a function of many
variables without any constraints on the variables, such as bounds. The
methods available in TAO for solving these problems can be classified
according to the amount of derivative information required:

#. Function evaluation only – Nelder-Mead method (``tao_nm``)

#. Function and gradient evaluations – limited-memory, variable-metric
   method (``tao_lmvm``) and nonlinear conjugate gradient method
   (``tao_cg``)

#. Function, gradient, and Hessian evaluations – Newton Krylov methods:
   Netwon line search (``tao_nls``), Newton trust-region (``tao_ntr``),
   and Newton trust-region line-search (``tao_ntl``)

The best method to use depends on the particular problem being solved
and the accuracy required in the solution. If a Hessian evaluation
routine is available, then the Newton line search and Newton
trust-region methods will likely perform best. When a Hessian evaluation
routine is not available, then the limited-memory, variable-metric
method is likely to perform best. The Nelder-Mead method should be used
only as a last resort when no gradient information is available.

Each solver has a set of options associated with it that can be set with
command line arguments. These algorithms and the associated options are
briefly discussed in this section.

Newton-Krylov Methods
^^^^^^^^^^^^^^^^^^^^^

TAO features three Newton-Krylov algorithms, separated by their globalization methods
for unconstrained optimization: line search (NLS), trust region (NTR), and trust
region with a line search (NTL). They are available via the TAO solvers
``TAONLS``, ``TAONTR`` and ``TAONTL``, respectively, or the ``-tao_type``
``nls``/``ntr``/``ntl`` flag.

Newton Line Search Method (NLS)
"""""""""""""""""""""""""""""""

The Newton line search method solves the symmetric system of equations

.. math:: H_k d_k = -g_k

to obtain a step :math:`d_k`, where :math:`H_k` is the Hessian of the
objective function at :math:`x_k` and :math:`g_k` is the gradient of the
objective function at :math:`x_k`. For problems where the Hessian matrix
is indefinite, the perturbed system of equations

.. math:: (H_k + \rho_k I) d_k = -g_k

is solved to obtain the direction, where :math:`\rho_k` is a positive
constant. If the direction computed is not a descent direction, the
(scaled) steepest descent direction is used instead. Having obtained the
direction, a Moré-Thuente line search is applied to obtain a step
length, :math:`\tau_k`, that approximately solves the one-dimensional
optimization problem

.. math:: \min_\tau f(x_k + \tau d_k).

The Newton line search method can be selected by using the TAO solver
``tao_nls``. The options available for this solver are listed in
:numref:`table_nlsoptions`. For the best efficiency, function and
gradient evaluations should be performed simultaneously when using this
algorithm.

 .. table:: Summary of ``nls`` options
    :name: table_nlsoptions

    +--------------------------+----------------+--------------------+-------------------------+
    | Name                     | Value          | Default            | Description             |
    +==========================+================+====================+=========================+
    | ``-tao_nls_ksp_type``    | cg, nash,      | stcg               | Type of Krylov          |
    |                          | stcg, gltr,    |                    | subspace                |
    |                          | gmres, ...     |                    | method to use           |
    |                          |                |                    | when solving            |
    |                          |                |                    | linear system           |
    +--------------------------+----------------+--------------------+-------------------------+
    | ``-tao_nls_pc_type``     | none, jacobi,  | lmvm               | Type of                 |
    |                          | icc, ilu, lmvm |                    | preconditioner          |
    |                          |                |                    | to use when             |
    |                          |                |                    | solving linear          |
    |                          |                |                    | system                  |
    +--------------------------+----------------+--------------------+-------------------------+
    | ``-tao_nls_sval``        | real           | :math:`0`          | Initial                 |
    |                          |                |                    | perturbation            |
    |                          |                |                    | value                   |
    +--------------------------+----------------+--------------------+-------------------------+
    | ``-tao_nls_imin``        | real           | :math:`10^{-4}`    | Minimum                 |
    |                          |                |                    | initial                 |
    |                          |                |                    | perturbation            |
    |                          |                |                    | value                   |
    +--------------------------+----------------+--------------------+-------------------------+
    | ``-tao_nls_imax``        | real           | :math:`100`        | Maximum                 |
    |                          |                |                    | initial                 |
    |                          |                |                    | perturbation            |
    |                          |                |                    | value                   |
    +--------------------------+----------------+--------------------+-------------------------+
    | ``-tao_nls_imfac``       | real           | :math:`0.1`        | Factor applied          |
    |                          |                |                    | to norm of              |
    |                          |                |                    | gradient when           |
    |                          |                |                    | initializing            |
    |                          |                |                    | perturbation            |
    +--------------------------+----------------+--------------------+-------------------------+
    | ``-tao_nls_pmax``        | real           | :math:`100`        | Maximum                 |
    |                          |                |                    | perturbation            |
    |                          |                |                    | when                    |
    |                          |                |                    | increasing              |
    |                          |                |                    | value                   |
    +--------------------------+----------------+--------------------+-------------------------+
    | ``-tao_nls_pgfac``       | real           | :math:`10`         | Growth factor           |
    |                          |                |                    | applied to              |
    |                          |                |                    | perturbation            |
    |                          |                |                    | when                    |
    |                          |                |                    | increasing              |
    |                          |                |                    | value                   |
    +--------------------------+----------------+--------------------+-------------------------+
    | ``-tao_nls_pmgfac``      | real           | :math:`0.1`        | Factor applied          |
    |                          |                |                    | to norm of              |
    |                          |                |                    | gradient when           |
    |                          |                |                    | increasing              |
    |                          |                |                    | perturbation            |
    +--------------------------+----------------+--------------------+-------------------------+
    | ``-tao_nls_pmin``        | real           | :math:`10^{-12}`   | Minimum                 |
    |                          |                |                    | perturbation            |
    |                          |                |                    | when                    |
    |                          |                |                    | decreasing              |
    |                          |                |                    | value; smaller          |
    |                          |                |                    | values set to           |
    |                          |                |                    | zero                    |
    +--------------------------+----------------+--------------------+-------------------------+
    | ``-tao_nls_psfac``       | real           | :math:`0.4`        | Shrink factor           |
    |                          |                |                    | applied to              |
    |                          |                |                    | perturbation            |
    |                          |                |                    | when                    |
    |                          |                |                    | decreasing              |
    |                          |                |                    | value                   |
    +--------------------------+----------------+--------------------+-------------------------+
    | ``-tao_nls_pmsfac``      | real           | :math:`0.1`        | Factor applied          |
    |                          |                |                    | to norm of              |
    |                          |                |                    | gradient when           |
    |                          |                |                    | decreasing              |
    |                          |                |                    | perturbation            |
    +--------------------------+----------------+--------------------+-------------------------+
    | ``-tao_nls_init_type``   | constant,      | interpolation      | Method used to          |
    |                          | direction,     |                    | initialize              |
    |                          | interpolation  |                    | trust-region            |
    |                          |                |                    | radius when             |
    |                          |                |                    | using                   |
    |                          |                |                    | ``nash``,               |
    |                          |                |                    | ``stcg``, or            |
    |                          |                |                    | ``gltr``                |
    +--------------------------+----------------+--------------------+-------------------------+
    | ``-tao_nls_mu1_i``       | real           | 0.35               | :math:`\mu_1`           |
    |                          |                |                    | in                      |
    |                          |                |                    | ``interpolation``       |
    |                          |                |                    | init                    |
    +--------------------------+----------------+--------------------+-------------------------+
    | ``-tao_nls_mu2_i``       | real           | 0.50               | :math:`\mu_2`           |
    |                          |                |                    | in                      |
    |                          |                |                    | ``interpolation``       |
    |                          |                |                    | init                    |
    +--------------------------+----------------+--------------------+-------------------------+
    | ``-tao_nls_gamma1_i``    | real           | 0.0625             | :math:`\gamma_1`        |
    |                          |                |                    | in                      |
    |                          |                |                    | ``interpolation``       |
    |                          |                |                    | init                    |
    +--------------------------+----------------+--------------------+-------------------------+
    | ``-tao_nls_gamma2_i``    | real           | 0.50               | :math:`\gamma_2`        |
    |                          |                |                    | in                      |
    |                          |                |                    | ``interpolation``       |
    |                          |                |                    | init                    |
    +--------------------------+----------------+--------------------+-------------------------+
    | ``-tao_nls_gamma3_i``    | real           | 2.00               | :math:`\gamma_3`        |
    |                          |                |                    | in                      |
    |                          |                |                    | ``interpolation``       |
    |                          |                |                    | init                    |
    +--------------------------+----------------+--------------------+-------------------------+
    | ``-tao_nls_gamma4_i``    | real           | 5.00               | :math:`\gamma_4`        |
    |                          |                |                    | in                      |
    |                          |                |                    | ``interpolation``       |
    |                          |                |                    | init                    |
    +--------------------------+----------------+--------------------+-------------------------+
    | ``-tao_nls_theta_i``     | real           | 0.25               | :math:`\theta`          |
    |                          |                |                    | in                      |
    |                          |                |                    | ``interpolation``       |
    |                          |                |                    | init                    |
    +--------------------------+----------------+--------------------+-------------------------+
    | ``-tao_nls_update_type`` | step,          | step               | Method used to          |
    |                          | reduction,     |                    | update                  |
    |                          | interpolation  |                    | trust-region            |
    |                          |                |                    | radius when             |
    |                          |                |                    | using                   |
    |                          |                |                    | ``nash``,               |
    |                          |                |                    | ``stcg``, or            |
    |                          |                |                    | ``gltr``                |
    +--------------------------+----------------+--------------------+-------------------------+
    | ``-tao_nls_nu1``         | real           | 0.25               | :math:`\nu_1`           |
    |                          |                |                    | in ``step``             |
    |                          |                |                    | update                  |
    +--------------------------+----------------+--------------------+-------------------------+
    | ``-tao_nls_nu2``         | real           | 0.50               | :math:`\nu_2`           |
    |                          |                |                    | in ``step``             |
    |                          |                |                    | update                  |
    +--------------------------+----------------+--------------------+-------------------------+
    | ``-tao_nls_nu3``         | real           | 1.00               | :math:`\nu_3`           |
    |                          |                |                    | in ``step``             |
    |                          |                |                    | update                  |
    +--------------------------+----------------+--------------------+-------------------------+
    | ``-tao_nls_nu4``         | real           | 1.25               | :math:`\nu_4`           |
    |                          |                |                    | in ``step``             |
    |                          |                |                    | update                  |
    +--------------------------+----------------+--------------------+-------------------------+
    | ``-tao_nls_omega1``      | real           | 0.25               | :math:`\omega_1`        |
    |                          |                |                    | in ``step``             |
    |                          |                |                    | update                  |
    +--------------------------+----------------+--------------------+-------------------------+
    | ``-tao_nls_omega2``      | real           | 0.50               | :math:`\omega_2`        |
    |                          |                |                    | in ``step``             |
    |                          |                |                    | update                  |
    +--------------------------+----------------+--------------------+-------------------------+
    | ``-tao_nls_omega3``      | real           | 1.00               | :math:`\omega_3`        |
    |                          |                |                    | in ``step``             |
    |                          |                |                    | update                  |
    +--------------------------+----------------+--------------------+-------------------------+
    | ``-tao_nls_omega4``      | real           | 2.00               | :math:`\omega_4`        |
    |                          |                |                    | in ``step``             |
    |                          |                |                    | update                  |
    +--------------------------+----------------+--------------------+-------------------------+
    | ``-tao_nls_omega5``      | real           | 4.00               | :math:`\omega_5`        |
    |                          |                |                    | in ``step``             |
    |                          |                |                    | update                  |
    +--------------------------+----------------+--------------------+-------------------------+
    | ``-tao_nls_eta1``        | real           | :math:`10^{-4}`    | :math:`\eta_1`          |
    |                          |                |                    | in                      |
    |                          |                |                    | ``reduction``           |
    |                          |                |                    | update                  |
    +--------------------------+----------------+--------------------+-------------------------+
    | ``-tao_nls_eta2``        | real           | 0.25               | :math:`\eta_2`          |
    |                          |                |                    | in                      |
    |                          |                |                    | ``reduction``           |
    |                          |                |                    | update                  |
    +--------------------------+----------------+--------------------+-------------------------+
    | ``-tao_nls_eta3``        | real           | 0.50               | :math:`\eta_3`          |
    |                          |                |                    | in                      |
    |                          |                |                    | ``reduction``           |
    |                          |                |                    | update                  |
    +--------------------------+----------------+--------------------+-------------------------+
    | ``-tao_nls_eta4``        | real           | 0.90               | :math:`\eta_4`          |
    |                          |                |                    | in                      |
    |                          |                |                    | ``reduction``           |
    |                          |                |                    | update                  |
    +--------------------------+----------------+--------------------+-------------------------+
    | ``-tao_nls_alpha1``      | real           | 0.25               | :math:`\alpha_1`        |
    |                          |                |                    | in                      |
    |                          |                |                    | ``reduction``           |
    |                          |                |                    | update                  |
    +--------------------------+----------------+--------------------+-------------------------+
    | ``-tao_nls_alpha2``      | real           | 0.50               | :math:`\alpha_2`        |
    |                          |                |                    | in                      |
    |                          |                |                    | ``reduction``           |
    |                          |                |                    | update                  |
    +--------------------------+----------------+--------------------+-------------------------+
    | ``-tao_nls_alpha3``      | real           | 1.00               | :math:`\alpha_3`        |
    |                          |                |                    | in                      |
    |                          |                |                    | ``reduction``           |
    |                          |                |                    | update                  |
    +--------------------------+----------------+--------------------+-------------------------+
    | ``-tao_nls_alpha4``      | real           | 2.00               | :math:`\alpha_4`        |
    |                          |                |                    | in                      |
    |                          |                |                    | ``reduction``           |
    |                          |                |                    | update                  |
    +--------------------------+----------------+--------------------+-------------------------+
    | ``-tao_nls_alpha5``      | real           | 4.00               | :math:`\alpha_5`        |
    |                          |                |                    | in                      |
    |                          |                |                    | ``reduction``           |
    |                          |                |                    | update                  |
    +--------------------------+----------------+--------------------+-------------------------+
    | ``-tao_nls_mu1``         | real           | 0.10               | :math:`\mu_1`           |
    |                          |                |                    | in                      |
    |                          |                |                    | ``interpolation``       |
    |                          |                |                    | update                  |
    +--------------------------+----------------+--------------------+-------------------------+
    | ``-tao_nls_mu2``         | real           | 0.50               | :math:`\mu_2`           |
    |                          |                |                    | in                      |
    |                          |                |                    | ``interpolation``       |
    |                          |                |                    | update                  |
    +--------------------------+----------------+--------------------+-------------------------+
    | ``-tao_nls_gamma1``      | real           | 0.25               | :math:`\gamma_1`        |
    |                          |                |                    | in                      |
    |                          |                |                    | ``interpolation``       |
    |                          |                |                    | update                  |
    +--------------------------+----------------+--------------------+-------------------------+
    | ``-tao_nls_gamma2``      | real           | 0.50               | :math:`\gamma_2`        |
    |                          |                |                    | in                      |
    |                          |                |                    | ``interpolation``       |
    |                          |                |                    | update                  |
    +--------------------------+----------------+--------------------+-------------------------+
    | ``-tao_nls_gamma3``      | real           | 2.00               | :math:`\gamma_3`        |
    |                          |                |                    | in                      |
    |                          |                |                    | ``interpolation``       |
    |                          |                |                    | update                  |
    +--------------------------+----------------+--------------------+-------------------------+
    | ``-tao_nls_gamma4``      | real           | 4.00               | :math:`\gamma_4`        |
    |                          |                |                    | in                      |
    |                          |                |                    | ``interpolation``       |
    |                          |                |                    | update                  |
    +--------------------------+----------------+--------------------+-------------------------+
    | ``-tao_nls_theta``       | real           | 0.05               | :math:`\theta`          |
    |                          |                |                    | in                      |
    |                          |                |                    | ``interpolation``       |
    |                          |                |                    | update                  |
    +--------------------------+----------------+--------------------+-------------------------+

The system of equations is approximately solved by applying the
conjugate gradient method, Nash conjugate gradient method,
Steihaug-Toint conjugate gradient method, generalized Lanczos method, or
an alternative Krylov subspace method supplied by PETSc. The method used
to solve the systems of equations is specified with the command line
argument ``-tao_nls_ksp_type <cg,nash,stcg,gltr,gmres,...>``; ``stcg``
is the default. See the PETSc manual for further information on changing
the behavior of the linear system solvers.

A good preconditioner reduces the number of iterations required to solve
the linear system of equations. For the conjugate gradient methods and
generalized Lanczos method, this preconditioner must be symmetric and
positive definite. The available options are to use no preconditioner,
the absolute value of the diagonal of the Hessian matrix, a
limited-memory BFGS approximation to the Hessian matrix, or one of the
other preconditioners provided by the PETSc package. These
preconditioners are specified by the command line arguments
``-tao_nls_pc_type <none,jacobi,icc,ilu,lmvm>``, respectively. The
default is the ``lmvm`` preconditioner, which uses a BFGS approximation
of the inverse Hessian. See the PETSc manual for further information on
changing the behavior of the preconditioners.

The perturbation :math:`\rho_k` is added when the direction returned by
the Krylov subspace method is not a descent direction, the Krylov method
diverged due to an indefinite preconditioner or matrix, or a direction
of negative curvature was found. In the last two cases, if the step
returned is a descent direction, it is used during the line search.
Otherwise, a steepest descent direction is used during the line search.
The perturbation is decreased as long as the Krylov subspace method
reports success and increased if further problems are encountered. There
are three cases: initializing, increasing, and decreasing the
perturbation. These cases are described below.

#. If :math:`\rho_k` is zero and a problem was detected with either the
   direction or the Krylov subspace method, the perturbation is
   initialized to

   .. math:: \rho_{k+1} = \text{median}\left\{\text{imin}, \text{imfac} * \|g(x_k)\|, \text{imax}\right\},

   where :math:`g(x_k)` is the gradient of the objective function and
   ``imin`` is set with the command line argument
   ``-tao_nls_imin <real>`` with a default value of :math:`10^{-4}`,
   ``imfac`` by ``-tao_nls_imfac`` with a default value of 0.1, and
   ``imax`` by ``-tao_nls_imax`` with a default value of 100. When using
   the ``gltr`` method to solve the system of equations, an estimate of
   the minimum eigenvalue :math:`\lambda_1` of the Hessian matrix is
   available. This value is used to initialize the perturbation to
   :math:`\rho_{k+1} = \max\left\{\rho_{k+1}, -\lambda_1\right\}` in
   this case.

#. If :math:`\rho_k` is nonzero and a problem was detected with either
   the direction or Krylov subspace method, the perturbation is
   increased to

   .. math:: \rho_{k+1} = \min\left\{\text{pmax}, \max\left\{\text{pgfac} * \rho_k, \text{pmgfac} * \|g(x_k)\|\right\}\right\},

   where :math:`g(x_k)` is the gradient of the objective function and
   ``pgfac`` is set with the command line argument ``-tao_nls_pgfac``
   with a default value of 10, ``pmgfac`` by ``-tao_nls_pmgfac`` with a
   default value of 0.1, and ``pmax`` by ``-tao_nls_pmax`` with a
   default value of 100.

#. If :math:`\rho_k` is nonzero and no problems were detected with
   either the direction or Krylov subspace method, the perturbation is
   decreased to

   .. math:: \rho_{k+1} = \min\left\{\text{psfac} * \rho_k, \text{pmsfac} * \|g(x_k)\|\right\},

   where :math:`g(x_k)` is the gradient of the objective function,
   ``psfac`` is set with the command line argument ``-tao_nls_psfac``
   with a default value of 0.4, and ``pmsfac`` is set by
   ``-tao_nls_pmsfac`` with a default value of 0.1. Moreover, if
   :math:`\rho_{k+1} < \text{pmin}`, then :math:`\rho_{k+1} = 0`, where
   ``pmin`` is set with the command line argument ``-tao_nls_pmin`` and
   has a default value of :math:`10^{-12}`.

Near a local minimizer to the unconstrained optimization problem, the
Hessian matrix will be positive-semidefinite; the perturbation will
shrink toward zero, and one would eventually observe a superlinear
convergence rate.

When using ``nash``, ``stcg``, or ``gltr`` to solve the linear systems
of equation, a trust-region radius needs to be initialized and updated.
This trust-region radius simultaneously limits the size of the step
computed and reduces the number of iterations of the conjugate gradient
method. The method for initializing the trust-region radius is set with
the command line argument
``-tao_nls_init_type <constant,direction,interpolation>``;
``interpolation``, which chooses an initial value based on the
interpolation scheme found in :cite:`CGT`, is the default.
This scheme performs a number of function and gradient evaluations to
determine a radius such that the reduction predicted by the quadratic
model along the gradient direction coincides with the actual reduction
in the nonlinear function. The iterate obtaining the best objective
function value is used as the starting point for the main line search
algorithm. The ``constant`` method initializes the trust-region radius
by using the value specified with the ``-tao_trust0 <real>`` command
line argument, where the default value is 100. The ``direction``
technique solves the first quadratic optimization problem by using a
standard conjugate gradient method and initializes the trust region to
:math:`\|s_0\|`.

The method for updating the trust-region radius is set with the command
line argument ``-tao_nls_update_type <step,reduction,interpolation>``;
``step`` is the default. The ``step`` method updates the trust-region
radius based on the value of :math:`\tau_k`. In particular,

.. math::

   \Delta_{k+1} = \left\{\begin{array}{ll}
   \omega_1 \text{min}(\Delta_k, \|d_k\|) & \text{if } \tau_k \in [0, \nu_1) \\
   \omega_2 \text{min}(\Delta_k, \|d_k\|) & \text{if } \tau_k \in [\nu_1, \nu_2) \\
   \omega_3 \Delta_k & \text{if } \tau_k \in [\nu_2, \nu_3) \\
   \text{max}(\Delta_k, \omega_4 \|d_k\|) & \text{if } \tau_k \in [\nu_3, \nu_4) \\
   \text{max}(\Delta_k, \omega_5 \|d_k\|) & \text{if } \tau_k \in [\nu_4, \infty),
   \end{array}
   \right.

where
:math:`0 < \omega_1 < \omega_2 < \omega_3 = 1 < \omega_4 < \omega_5` and
:math:`0 < \nu_1 < \nu_2 < \nu_3 < \nu_4` are constants. The
``reduction`` method computes the ratio of the actual reduction in the
objective function to the reduction predicted by the quadratic model for
the full step,
:math:`\kappa_k = \frac{f(x_k) - f(x_k + d_k)}{q(x_k) - q(x_k + d_k)}`,
where :math:`q_k` is the quadratic model. The radius is then updated as

.. math::

   \Delta_{k+1} = \left\{\begin{array}{ll}
   \alpha_1 \text{min}(\Delta_k, \|d_k\|) & \text{if } \kappa_k \in (-\infty, \eta_1) \\
   \alpha_2 \text{min}(\Delta_k, \|d_k\|) & \text{if } \kappa_k \in [\eta_1, \eta_2) \\
   \alpha_3 \Delta_k & \text{if } \kappa_k \in [\eta_2, \eta_3) \\
   \text{max}(\Delta_k, \alpha_4 \|d_k\|) & \text{if } \kappa_k \in [\eta_3, \eta_4) \\
   \text{max}(\Delta_k, \alpha_5 \|d_k\|) & \text{if } \kappa_k \in [\eta_4, \infty),
   \end{array}
   \right.

where
:math:`0 < \alpha_1 < \alpha_2 < \alpha_3 = 1 < \alpha_4 < \alpha_5` and
:math:`0 < \eta_1 < \eta_2 < \eta_3 < \eta_4` are constants. The
``interpolation`` method uses the same interpolation mechanism as in the
initialization to compute a new value for the trust-region radius.

This algorithm will be deprecated in the next version and replaced by
the Bounded Newton Line Search (BNLS) algorithm that can solve both
bound constrained and unconstrained problems.

Newton Trust-Region Method (NTR)
""""""""""""""""""""""""""""""""

The Newton trust-region method solves the constrained quadratic
programming problem

.. math::

   \begin{array}{ll}
   \min_d  & \frac{1}{2}d^T H_k d  + g_k^T d \\
   \text{subject to} & \|d\| \leq \Delta_k
   \end{array}

to obtain a direction :math:`d_k`, where :math:`H_k` is the Hessian of
the objective function at :math:`x_k`, :math:`g_k` is the gradient of
the objective function at :math:`x_k`, and :math:`\Delta_k` is the
trust-region radius. If :math:`x_k + d_k` sufficiently reduces the
nonlinear objective function, then the step is accepted, and the
trust-region radius is updated. However, if :math:`x_k + d_k` does not
sufficiently reduce the nonlinear objective function, then the step is
rejected, the trust-region radius is reduced, and the quadratic program
is re-solved by using the updated trust-region radius. The Newton
trust-region method can be set by using the TAO solver ``tao_ntr``. The
options available for this solver are listed in
:numref:`table_ntroptions`. For the best efficiency, function and
gradient evaluations should be performed separately when using this
algorithm.

   .. table:: Summary of ``ntr`` options
      :name: table_ntroptions

      +---------------------------+----------------+------------------+----------------------+
      | Name                      | Value          | Default          | Description          |
      +===========================+================+==================+======================+
      | ``-tao_ntr_ksp_type``     | nash, stcg,    | stcg             | Type of Krylov       |
      |                           | gltr           |                  | subspace             |
      |                           |                |                  | method to use        |
      |                           |                |                  | when solving         |
      |                           |                |                  | linear system        |
      +---------------------------+----------------+------------------+----------------------+
      | ``-tao_ntr_pc_type``      | none, jacobi,  | lmvm             | Type of              |
      |                           | icc, ilu, lmvm |                  | preconditioner       |
      |                           |                |                  | to use when          |
      |                           |                |                  | solving linear       |
      |                           |                |                  | system               |
      +---------------------------+----------------+------------------+----------------------+
      | ``-tao_ntr_init_type``    | constant,      | interpolation    | Method used to       |
      |                           | direction,     |                  | initialize           |
      |                           | interpolation  |                  | trust-region         |
      |                           |                |                  | radius               |
      +---------------------------+----------------+------------------+----------------------+
      | ``-tao_ntr_mu1_i``        | real           | 0.35             | :math:`\mu_1`        |
      |                           |                |                  | in                   |
      |                           |                |                  | ``interpolation``    |
      |                           |                |                  | init                 |
      +---------------------------+----------------+------------------+----------------------+
      | ``-tao_ntr_mu2_i``        | real           | 0.50             | :math:`\mu_2`        |
      |                           |                |                  | in                   |
      |                           |                |                  | ``interpolation``    |
      |                           |                |                  | init                 |
      +---------------------------+----------------+------------------+----------------------+
      | ``-tao_ntr_gamma1_i``     | real           | 0.0625           | :math:`\gamma_1`     |
      |                           |                |                  | in                   |
      |                           |                |                  | ``interpolation``    |
      |                           |                |                  | init                 |
      +---------------------------+----------------+------------------+----------------------+
      | ``-tao_ntr_gamma2_i``     | real           | 0.50             | :math:`\gamma_2`     |
      |                           |                |                  | in                   |
      |                           |                |                  | ``interpolation``    |
      |                           |                |                  | init                 |
      +---------------------------+----------------+------------------+----------------------+
      | ``-tao_ntr_gamma3_i``     | real           | 2.00             | :math:`\gamma_3`     |
      |                           |                |                  | in                   |
      |                           |                |                  | ``interpolation``    |
      |                           |                |                  | init                 |
      +---------------------------+----------------+------------------+----------------------+
      | ``-tao_ntr_gamma4_i``     | real           | 5.00             | :math:`\gamma_4`     |
      |                           |                |                  | in                   |
      |                           |                |                  | ``interpolation``    |
      |                           |                |                  | init                 |
      +---------------------------+----------------+------------------+----------------------+
      | ``-tao_ntr_theta_i``      | real           | 0.25             | :math:`\theta`       |
      |                           |                |                  | in                   |
      |                           |                |                  | ``interpolation``    |
      |                           |                |                  | init                 |
      +---------------------------+----------------+------------------+----------------------+
      | ``-tao_ntr_update_type``  | reduction,     | reduction        | Method used to       |
      |                           | interpolation  |                  | update               |
      |                           |                |                  | trust-region         |
      |                           |                |                  | radius               |
      +---------------------------+----------------+------------------+----------------------+
      | ``-tao_ntr_eta1``         | real           | :                | :math:`\eta_1`       |
      |                           |                |                  | in ``reduction``     |
      |                           |                |                  | update               |
      +---------------------------+----------------+------------------+----------------------+
      | ``-tao_ntr_eta2``         | real           | 0.25             | :math:`\eta_2`       |
      |                           |                |                  | in ``reduction``     |
      |                           |                |                  | update               |
      +---------------------------+----------------+------------------+----------------------+
      | ``-tao_ntr_eta3``         | real           | 0.50             | :math:`\eta_3`       |
      |                           |                |                  | in ``reduction``     |
      |                           |                |                  | update               |
      +---------------------------+----------------+------------------+----------------------+
      | ``-tao_ntr_eta4``         | real           | 0.90             | :math:`\eta_4`       |
      |                           |                |                  | in ``reduction``     |
      |                           |                |                  | update               |
      +---------------------------+----------------+------------------+----------------------+
      | ``-tao_ntr_alpha1``       | real           | 0.25             | :math:`\alpha_1`     |
      |                           |                |                  | in ``reduction``     |
      |                           |                |                  | update               |
      +---------------------------+----------------+------------------+----------------------+
      | ``-tao_ntr_alpha2``       | real           | 0.50             | :math:`\alpha_2`     |
      |                           |                |                  | in ``reduction``     |
      |                           |                |                  | update               |
      +---------------------------+----------------+------------------+----------------------+
      | ``-tao_ntr_alpha3``       | real           | 1.00             | :math:`\alpha_3`     |
      |                           |                |                  | in ``reduction``     |
      |                           |                |                  | update               |
      +---------------------------+----------------+------------------+----------------------+
      | ``-tao_ntr_alpha4``       | real           | 2.00             | :math:`\alpha_4`     |
      |                           |                |                  | in ``reduction``     |
      |                           |                |                  | update               |
      +---------------------------+----------------+------------------+----------------------+
      | ``-tao_ntr_alpha5``       | real           | 4.00             | :math:`\alpha_5`     |
      |                           |                |                  | in ``reduction``     |
      |                           |                |                  | update               |
      +---------------------------+----------------+------------------+----------------------+
      | ``-tao_ntr_mu1``          | real           | 0.10             | :math:`\mu_1`        |
      |                           |                |                  | in                   |
      |                           |                |                  | ``interpolation``    |
      |                           |                |                  | update               |
      +---------------------------+----------------+------------------+----------------------+
      | ``-tao_ntr_mu2``          | real           | 0.50             | :math:`\mu_2`        |
      |                           |                |                  | in                   |
      |                           |                |                  | ``interpolation``    |
      |                           |                |                  | update               |
      +---------------------------+----------------+------------------+----------------------+
      | ``-tao_ntr_gamma1``       | real           | 0.25             | :math:`\gamma_1`     |
      |                           |                |                  | in                   |
      |                           |                |                  | ``interpolation``    |
      |                           |                |                  | update               |
      +---------------------------+----------------+------------------+----------------------+
      | ``-tao_ntr_gamma2``       | real           | 0.50             | :math:`\gamma_2`     |
      |                           |                |                  | in                   |
      |                           |                |                  | ``interpolation``    |
      |                           |                |                  | update               |
      +---------------------------+----------------+------------------+----------------------+
      | ``-tao_ntr_gamma3``       | real           | 2.00             | :math:`\gamma_3`     |
      |                           |                |                  | in                   |
      |                           |                |                  | ``interpolation``    |
      |                           |                |                  | update               |
      +---------------------------+----------------+------------------+----------------------+
      | ``-tao_ntr_gamma4``       | real           | 4.00             | :math:`\gamma_4`     |
      |                           |                |                  | in                   |
      |                           |                |                  | ``interpolation``    |
      |                           |                |                  | update               |
      +---------------------------+----------------+------------------+----------------------+
      | ``-tao_ntr_theta``        | real           | 0.05             | :math:`\theta`       |
      |                           |                |                  | in                   |
      |                           |                |                  | ``interpolation``    |
      |                           |                |                  | update               |
      +---------------------------+----------------+------------------+----------------------+

The quadratic optimization problem is approximately solved by applying
the Nash or Steihaug-Toint conjugate gradient methods or the generalized
Lanczos method to the symmetric system of equations
:math:`H_k d = -g_k`. The method used to solve the system of equations
is specified with the command line argument
``-tao_ntr_ksp_type <nash,stcg,gltr>``; ``stcg`` is the default. See the
PETSc manual for further information on changing the behavior of these
linear system solvers.

A good preconditioner reduces the number of iterations required to
compute the direction. For the Nash and Steihaug-Toint conjugate
gradient methods and generalized Lanczos method, this preconditioner
must be symmetric and positive definite. The available options are to
use no preconditioner, the absolute value of the diagonal of the Hessian
matrix, a limited-memory BFGS approximation to the Hessian matrix, or
one of the other preconditioners provided by the PETSc package. These
preconditioners are specified by the command line argument
``-tao_ntr_pc_type <none,jacobi,icc,ilu,lmvm>``, respectively. The
default is the ``lmvm`` preconditioner. See the PETSc manual for further
information on changing the behavior of the preconditioners.

The method for computing an initial trust-region radius is set with the
command line arguments
``-tao_ntr_init_type <constant,direction,interpolation>``;
``interpolation``, which chooses an initial value based on the
interpolation scheme found in :cite:`CGT`, is the default.
This scheme performs a number of function and gradient evaluations to
determine a radius such that the reduction predicted by the quadratic
model along the gradient direction coincides with the actual reduction
in the nonlinear function. The iterate obtaining the best objective
function value is used as the starting point for the main trust-region
algorithm. The ``constant`` method initializes the trust-region radius
by using the value specified with the ``-tao_trust0 <real>`` command
line argument, where the default value is 100. The ``direction``
technique solves the first quadratic optimization problem by using a
standard conjugate gradient method and initializes the trust region to
:math:`\|s_0\|`.

The method for updating the trust-region radius is set with the command
line arguments ``-tao_ntr_update_type <reduction,interpolation>``;
``reduction`` is the default. The ``reduction`` method computes the
ratio of the actual reduction in the objective function to the reduction
predicted by the quadratic model for the full step,
:math:`\kappa_k = \frac{f(x_k) - f(x_k + d_k)}{q(x_k) - q(x_k + d_k)}`,
where :math:`q_k` is the quadratic model. The radius is then updated as

.. math::

   \Delta_{k+1} = \left\{\begin{array}{ll}
   \alpha_1 \text{min}(\Delta_k, \|d_k\|) & \text{if } \kappa_k \in (-\infty, \eta_1) \\
   \alpha_2 \text{min}(\Delta_k, \|d_k\|) & \text{if } \kappa_k \in [\eta_1, \eta_2) \\
   \alpha_3 \Delta_k & \text{if } \kappa_k \in [\eta_2, \eta_3) \\
   \text{max}(\Delta_k, \alpha_4 \|d_k\|) & \text{if } \kappa_k \in [\eta_3, \eta_4) \\
   \text{max}(\Delta_k, \alpha_5 \|d_k\|) & \text{if } \kappa_k \in [\eta_4, \infty),
   \end{array}
   \right.

where
:math:`0 < \alpha_1 < \alpha_2 < \alpha_3 = 1 < \alpha_4 < \alpha_5` and
:math:`0 < \eta_1 < \eta_2 < \eta_3 < \eta_4` are constants. The
``interpolation`` method uses the same interpolation mechanism as in the
initialization to compute a new value for the trust-region radius.

This algorithm will be deprecated in the next version and replaced by
the Bounded Newton Trust Region (BNTR) algorithm that can solve both
bound constrained and unconstrained problems.

Newton Trust Region with Line Search (NTL)
""""""""""""""""""""""""""""""""""""""""""

NTL safeguards the trust-region globalization such that a line search
is used in the event that the step is initially rejected by the
predicted versus actual decrease comparison. If the line search fails to
find a viable step length for the Newton step, it falls back onto a
scaled gradient or a gradient descent step. The trust radius is then
modified based on the line search step length.

This algorithm will be deprecated in the next version and replaced by
the Bounded Newton Trust Region with Line Search (BNTL) algorithm that
can solve both bound constrained and unconstrained problems.

Limited-Memory Variable-Metric Method (LMVM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The limited-memory, variable-metric method (LMVM) computes a positive definite
approximation to the Hessian matrix from a limited number of previous
iterates and gradient evaluations. A direction is then obtained by
solving the system of equations

.. math:: H_k d_k = -\nabla f(x_k),

where :math:`H_k` is the Hessian approximation obtained by using the
BFGS update formula. The inverse of :math:`H_k` can readily be applied
to obtain the direction :math:`d_k`. Having obtained the direction, a
Moré-Thuente line search is applied to compute a step length,
:math:`\tau_k`, that approximately solves the one-dimensional
optimization problem

.. math:: \min_\tau f(x_k + \tau d_k).

The current iterate and Hessian approximation are updated, and the
process is repeated until the method converges. This algorithm is the
default unconstrained minimization solver and can be selected by using
the TAO solver ``tao_lmvm``. For best efficiency, function and gradient
evaluations should be performed simultaneously when using this
algorithm.

The primary factors determining the behavior of this algorithm are the
type of Hessian approximation used, the number of vectors stored for the
approximation and the initialization/scaling of the approximation. These
options can be configured using the ``-tao_lmvm_mat_lmvm`` prefix. For
further detail, we refer the reader to the ``MATLMVM`` matrix type
definitions in the PETSc Manual.

The LMVM algorithm also allows the user to define a custom initial
Hessian matrix :math:`H_{0,k}` through the interface function
``TaoLMVMSetH0()``. This user-provided initialization overrides any
other scalar or diagonal initialization inherent to the LMVM
approximation. The provided :math:`H_{0,k}` must be a PETSc ``Mat`` type
object that represents a positive-definite matrix. The approximation
prefers ``MatSolve()`` if the provided matrix has ``MATOP_SOLVE``
implemented. Otherwise, ``MatMult()`` is used in a KSP solve to perform
the inversion of the user-provided initial Hessian.

In applications where ``TaoSolve()`` on the LMVM algorithm is repeatedly
called to solve similar or related problems, ``-tao_lmvm_recycle`` flag
can be used to prevent resetting the LMVM approximation between
subsequent solutions. This recycling also avoids one extra function and
gradient evaluation, instead re-using the values already computed at the
end of the previous solution.

This algorithm will be deprecated in the next version and replaced by
the Bounded Quasi-Newton Line Search (BQNLS) algorithm that can solve
both bound constrained and unconstrained problems.

Nonlinear Conjugate Gradient Method (CG)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The nonlinear conjugate gradient method can be viewed as an extension of
the conjugate gradient method for solving symmetric, positive-definite
linear systems of equations. This algorithm requires only function and
gradient evaluations as well as a line search. The TAO implementation
uses a Moré-Thuente line search to obtain the step length. The nonlinear
conjugate gradient method can be selected by using the TAO solver
``tao_cg``. For the best efficiency, function and gradient evaluations
should be performed simultaneously when using this algorithm.

Five variations are currently supported by the TAO implementation: the
Fletcher-Reeves method, the Polak-Ribiére method, the Polak-Ribiére-Plus
method :cite:`NW99`, the Hestenes-Stiefel method, and the
Dai-Yuan method. These conjugate gradient methods can be specified by
using the command line argument ``-tao_cg_type <fr,pr,prp,hs,dy>``,
respectively. The default value is ``prp``.

The conjugate gradient method incorporates automatic restarts when
successive gradients are not sufficiently orthogonal. TAO measures the
orthogonality by dividing the inner product of the gradient at the
current point and the gradient at the previous point by the square of
the Euclidean norm of the gradient at the current point. When the
absolute value of this ratio is greater than :math:`\eta`, the algorithm
restarts using the gradient direction. The parameter :math:`\eta` can be
set by using the command line argument ``-tao_cg_eta <real>``; 0.1 is
the default value.

This algorithm will be deprecated in the next version and replaced by
the Bounded Nonlinear Conjugate Gradient (BNCG) algorithm that can solve
both bound constrained and unconstrained problems.

Nelder-Mead Simplex Method (NM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Nelder-Mead algorithm :cite:`nelder.mead:simplex` is a
direct search method for finding a local minimum of a function
:math:`f(x)`. This algorithm does not require any gradient or Hessian
information of :math:`f` and therefore has some expected advantages and
disadvantages compared to the other TAO solvers. The obvious advantage
is that it is easier to write an application when no derivatives need to
be calculated. The downside is that this algorithm can be slow to
converge or can even stagnate, and it performs poorly for large numbers
of variables.

This solver keeps a set of :math:`N+1` sorted vectors
:math:`{x_1,x_2,\ldots,x_{N+1}}` and their corresponding objective
function values :math:`f_1 \leq f_2 \leq \ldots \leq f_{N+1}`. At each
iteration, :math:`x_{N+1}` is removed from the set and replaced with

.. math:: x(\mu) = (1+\mu) \frac{1}{N} \sum_{i=1}^N x_i - \mu x_{N+1},

where :math:`\mu` can be one of
:math:`{\mu_0,2\mu_0,\frac{1}{2}\mu_0,-\frac{1}{2}\mu_0}` depending on
the values of each possible :math:`f(x(\mu))`.

The algorithm terminates when the residual :math:`f_{N+1} - f_1` becomes
sufficiently small. Because of the way new vectors can be added to the
sorted set, the minimum function value and/or the residual may not be
impacted at each iteration.

Two options can be set specifically for the Nelder-Mead algorithm:

``-tao_nm_lamda <value>``
   sets the initial set of vectors (:math:`x_0` plus ``value`` in each
   coordinate direction); the default value is :math:`1`.

``-tao_nm_mu <value>``
   sets the value of :math:`\mu_0`; the default is :math:`\mu_0=1`.

.. _sec_tao_bound:

Bound-Constrained Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bound-constrained optimization algorithms solve optimization problems of
the form

.. math::

   \begin{array}{ll} \displaystyle
   \min_{x} & f(x) \\
   \text{subject to} & l \leq x \leq u.
   \end{array}

These solvers use the bounds on the variables as well as objective
function, gradient, and possibly Hessian information.

For any unbounded variables, the bound value for the associated index
can be set to ``PETSC_INFINITY`` for the upper bound and
``PETSC_NINFINITY`` for the lower bound. If all bounds are set to
infinity, then the bounded algorithms are equivalent to their
unconstrained counterparts.

Before introducing specific methods, we will first define two projection
operations used by all bound constrained algorithms.

-  Gradient projection:

   .. math::

      \mathfrak{P}(g) = \left\{\begin{array}{ll}
      0 & \text{if} \; (x \leq l_i \land g_i > 0) \lor (x \geq u_i \land g_i < 0) \\
      g_i & \text{otherwise}
      \end{array}
      \right.

-  Bound projection:

   .. math::

      \mathfrak{B}(x) = \left\{\begin{array}{ll}
      l_i & \text{if} \; x_i < l_i \\
      u_i & \text{if} \; x_i > u_i \\
      x_i & \text{otherwise}
      \end{array}
      \right.

.. _sec_bnk:

Bounded Newton-Krylov Methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TAO features three bounded Newton-Krylov (BNK) class of algorithms,
separated by their globalization methods: projected line search (BNLS),
trust region (BNTR), and trust region with a projected line search
fall-back (BNTL). They are available via the TAO solvers ``TAOBNLS``,
``TAOBNTR`` and ``TAOBNTL``, respectively, or the ``-tao_type``
``bnls``/``bntr``/``bntl`` flag.

The BNK class of methods use an active-set approach to solve the
symmetric system of equations,

.. math:: H_k p_k = -g_k,

only for inactive variables in the interior of the bounds. The
active-set estimation is based on Bertsekas
:cite:`bertsekas:projected` with the following variable
index categories:

.. math::

   \begin{array}{rlll} \displaystyle
   \text{lower bounded}: & \mathcal{L}(x) & = & \{ i \; : \; x_i \leq l_i + \epsilon \; \land \; g(x)_i > 0 \}, \\
   \text{upper bounded}: & \mathcal{U}(x) & = & \{ i \; : \; x_i \geq u_i + \epsilon \; \land \; g(x)_i < 0 \}, \\
   \text{fixed}: & \mathcal{F}(x) & = & \{ i \; : \; l_i = u_i \}, \\
   \text{active-set}: & \mathcal{A}(x) & = & \{ \mathcal{L}(x) \; \bigcup \; \mathcal{U}(x) \; \bigcup \; \mathcal{F}(x) \}, \\
   \text{inactive-set}: & \mathcal{I}(x) & = & \{ 1,2,\ldots,n \} \; \backslash \; \mathcal{A}(x).
   \end{array}

At each iteration, the bound tolerance is estimated as
:math:`\epsilon_{k+1} = \text{min}(\epsilon_k, ||w_k||_2)` with
:math:`w_k = x_k - \mathfrak{B}(x_k - \beta D_k g_k)`, where the
diagonal matrix :math:`D_k` is an approximation of the Hessian inverse
:math:`H_k^{-1}`. The initial bound tolerance :math:`\epsilon_0` and the
step length :math:`\beta` have default values of :math:`0.001` and can
be adjusted using ``-tao_bnk_as_tol`` and ``-tao_bnk_as_step`` flags,
respectively. The active-set estimation can be disabled using the option
``-tao_bnk_as_type none``, in which case the algorithm simply uses the
current iterate with no bound tolerances to determine which variables
are actively bounded and which are free.

BNK algorithms invert the reduced Hessian using a Krylov iterative
method. Trust-region conjugate gradient methods (``KSPNASH``,
``KSPSTCG``, and ``KSPGLTR``) are required for the BNTR and BNTL
algorithms, and recommended for the BNLS algorithm. The preconditioner
type can be changed using the ``-tao_bnk_pc_type``
``none``/``ilu``/``icc``/``jacobi``/``lmvm``. The ``lmvm`` option, which
is also the default, preconditions the Krylov solution with a
``MATLMVM`` matrix. The remaining supported preconditioner types are
default PETSc types. If Jacobi is selected, the diagonal values are
safeguarded to be positive. ``icc`` and ``ilu`` options produce good
results for problems with dense Hessians. The LMVM and Jacobi
preconditioners are also used as the approximate inverse-Hessian in the
active-set estimation. If neither are available, or if the Hessian
matrix does not have ``MATOP_GET_DIAGONAL`` defined, then the active-set
estimation falls back onto using an identity matrix in place of
:math:`D_k` (this is equivalent to estimating the active-set using a
gradient descent step).

A special option is available to *accelerate* the convergence of the BNK
algorithms by taking a finite number of BNCG iterations at each Newton
iteration. By default, the number of BNCG iterations is set to zero and
the algorithms do not take any BNCG steps. This can be changed using the
option flag ``-tao_bnk_max_cg_its <i>``. While this reduces the number
of Newton iterations, in practice it simply trades off the Hessian
evaluations in the BNK solver for more function and gradient evaluations
in the BNCG solver. However, it may be useful for certain types of
problems where the Hessian evaluation is disproportionately more
expensive than the objective function or its gradient.

.. _sec_bnls:

Bounded Newton Line Search (BNLS)
"""""""""""""""""""""""""""""""""

BNLS safeguards the Newton step by falling back onto a BFGS, scaled
gradient, or gradient steps based on descent direction verifications.
For problems with indefinite Hessian matrices, the step direction is
calculated using a perturbed system of equations,

.. math:: (H_k + \rho_k I)p_k = -g_k,

where :math:`\rho_k` is a dynamically adjusted positive constant. The
step is globalized using a projected Moré-Thuente line search. If a
trust-region conjugate gradient method is used for the Hessian
inversion, the trust radius is modified based on the line search step
length.

.. _sec_bntr:

Bounded Newton Trust Region (BNTR)
""""""""""""""""""""""""""""""""""

BNTR globalizes the Newton step using a trust region method based on the
predicted versus actual reduction in the cost function. The trust radius
is increased only if the accepted step is at the trust region boundary.
The reduction check features a safeguard for numerical values below
machine epsilon, scaled by the latest function value, where the full
Newton step is accepted without modification.

.. _sec_bntl:

Bounded Newton Trust Region with Line Search (BNTL)
"""""""""""""""""""""""""""""""""""""""""""""""""""

BNTL safeguards the trust-region globalization such that a line search
is used in the event that the step is initially rejected by the
predicted versus actual decrease comparison. If the line search fails to
find a viable step length for the Newton step, it falls back onto a
scaled gradient or a gradient descent step. The trust radius is then
modified based on the line search step length.

.. _sec_bqnls:

Bounded Quasi-Newton Line Search (BQNLS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The BQNLS algorithm uses the BNLS infrastructure, but replaces the step
calculation with a direct inverse application of the approximate Hessian
based on quasi-Newton update formulas. No Krylov solver is used in the
solution, and therefore the quasi-Newton method chosen must guarantee a
positive-definite Hessian approximation. This algorithm is available via
``tao_type bqnls``.

.. _sec_bqnk:

Bounded Quasi-Newton-Krylov
^^^^^^^^^^^^^^^^^^^^^^^^^^^

BQNK algorithms use the BNK infrastructure, but replace the exact
Hessian with a quasi-Newton approximation. The matrix-free forward
product operation based on quasi-Newton update formulas are used in
conjunction with Krylov solvers to compute step directions. The
quasi-Newton inverse application is used to precondition the Krylov
solution, and typically helps converge to a step direction in
:math:`\mathcal{O}(10)` iterations. This approach is most useful with
quasi-Newton update types such as Symmetric Rank-1 that cannot strictly
guarantee positive-definiteness. The BNLS framework with Hessian
shifting, or the BNTR framework with trust region safeguards, can
successfully compensate for the Hessian approximation becoming
indefinite.

Similar to the full Newton-Krylov counterpart, BQNK algorithms come in
three forms separated by the globalization technique: line search
(BQNKLS), trust region (BQNKTR) and trust region w/ line search
fall-back (BQNKTL). These algorithms are available via
``tao_type <bqnkls, bqnktr, bqnktl>``.

.. _sec_bncg:

Bounded Nonlinear Conjugate Gradient (BNCG)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

BNCG extends the unconstrained nonlinear conjugate gradient algorithm to
bound constraints via gradient projections and a bounded Moré-Thuente
line search.

Like its unconstrained counterpart, BNCG offers gradient descent and a
variety of CG updates: Fletcher-Reeves, Polak-Ribiére,
Polak-Ribiére-Plus, Hestenes-Stiefel, Dai-Yuan, Hager-Zhang, Dai-Kou,
Kou-Dai, and the Self-Scaling Memoryless (SSML) BFGS, DFP, and Broyden
methods. These methods can be specified by using the command line
argument
``-tao_bncg_type <gd,fr,pr,prp,hs,dy,hz,dk,kd,ssml_bfgs,ssml_dfp,ssml_brdn>``,
respectively. The default value is ``ssml_bfgs``. We have scalar
preconditioning for these methods, and it is controlled by the flag
``tao_bncg_alpha``. To disable rescaling, use :math:`\alpha = -1.0`,
otherwise :math:`\alpha \in [0, 1]`. BNCG is available via the TAO
solver ``TAOBNCG`` or the ``-tao_type bncg`` flag.

Some individual methods also contain their own parameters. The
Hager-Zhang and Dou-Kai methods have a parameter that determines the
minimum amount of contribution the previous search direction gives to
the next search direction. The flags are ``-tao_bncg_hz_eta`` and
``-tao_bncg_dk_eta``, and by default are set to :math:`0.4` and
:math:`0.5` respectively. The Kou-Dai method has multiple parameters.
``-tao_bncg_zeta`` serves the same purpose as the previous two; set to
:math:`0.1` by default. There is also a parameter to scale the
contribution of :math:`y_k \equiv \nabla f(x_k) - \nabla f(x_{k-1})` in
the search direction update. It is controlled by ``-tao_bncg_xi``, and
is equal to :math:`1.0` by default. There are also times where we want
to maximize the descent as measured by :math:`\nabla f(x_k)^T d_k`, and
that may be done by using a negative value of :math:`\xi`; this achieves
better performance when not using the diagonal preconditioner described
next. This is enabled by default, and is controlled by
``-tao_bncg_neg_xi``. Finally, the Broyden method has its convex
combination parameter, set with ``-tao_bncg_theta``. We have this as 1.0
by default, i.e. it is by default the BFGS method. One can also
individually tweak the BFGS and DFP contributions using the
multiplicative constants ``-tao_bncg_scale``; both are set to :math:`1`
by default.

All methods can be scaled using the parameter ``-tao_bncg_alpha``, which
continuously varies in :math:`[0, 1]`. The default value is set
depending on the method from initial testing.

BNCG also offers a special type of method scaling. It employs Broyden
diagonal scaling as an option for its CG methods, turned on with the
flag ``-tao_bncg_diag_scaling``. Formulations for both the forward
(regular) and inverse Broyden methods are developed, controlled by the
flag ``-tao_bncg_mat_lmvm_forward``. It is set to True by default.
Whether one uses the forward or inverse formulations depends on the
method being used. For example, in our preliminary computations, the
forward formulation works better for the SSML_BFGS method, but the
inverse formulation works better for the Hestenes-Stiefel method. The
convex combination parameter for the Broyden scaling is controlled by
``-tao_bncg_mat_lmvm_theta``, and is 0 by default. We also employ
rescaling of the Broyden diagonal, which aids the linesearch immensely.
The rescaling parameter is controlled by ``-tao_bncg_mat_lmvm_alpha``,
and should be :math:`\in [0, 1]`. One can disable rescaling of the
Broyden diagonal entirely by setting
``-tao_bncg_mat_lmvm_sigma_hist 0``.

One can also supply their own preconditioner, serving as a Hessian
initialization to the above diagonal scaling. The appropriate user
function in the code is ``TaoBNCGSetH0(tao, H0)`` where ``H0`` is the
user-defined ``Mat`` object that serves as a preconditioner. For an
example of similar usage, see ``tao/tutorials/ex3.c``.

The active set estimation uses the Bertsekas-based method described in
:any:`sec_bnk`, which can be deactivated using
``-tao_bncg_as_type none``, in which case the algorithm will use the
current iterate to determine the bounded variables with no tolerances
and no look-ahead step. As in the BNK algorithm, the initial bound
tolerance and estimator step length used in the Bertsekas method can be
set via ``-tao_bncg_as_tol`` and ``-tao_bncg_as_step``, respectively.

In addition to automatic scaled gradient descent restarts under certain
local curvature conditions, we also employ restarts based on a check on
descent direction such that
:math:`\nabla f(x_k)^T d_k \in [-10^{11}, -10^{-9}]`. Furthermore, we
allow for a variety of alternative restart strategies, all disabled by
default. The ``-tao_bncg_unscaled_restart`` flag allows one to disable
rescaling of the gradient for gradient descent steps. The
``-tao_bncg_spaced_restart`` flag tells the solver to restart every
:math:`Mn` iterations, where :math:`n` is the problem dimension and
:math:`M` is a constant determined by ``-tao_bncg_min_restart_num`` and
is 6 by default. We also have dynamic restart strategies based on
checking if a function is locally quadratic; if so, go do a gradient
descent step. The flag is ``-tao_bncg_dynamic_restart``, disabled by
default since the CG solver usually does better in those cases anyway.
The minimum number of quadratic-like steps before a restart is set using
``-tao_bncg_min_quad`` and is 6 by default.

Generally Constrained Solvers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Constrained solvers solve optimization problems that incorporate either or both
equality and inequality constraints, and may optionally include bounds on
solution variables.

Alternating Direction Method of Multipliers (ADMM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The TAOADMM algorithm is intended to blend the decomposability
of dual ascent with the superior convergence properties of the method of
multipliers. :cite:`boyd` The algorithm solves problems in
the form

.. math::

   \begin{array}{ll}
   \displaystyle \min_{x} & f(x) + g(z) \\
   \text{subject to} & Ax + Bz = c
   \end{array}

where :math:`x \in \mathbb R^n`, :math:`z \in \mathbb R^m`,
:math:`A \in \mathbb R^{p \times n}`,
:math:`B \in \mathbb R^{p \times m}`, and :math:`c \in \mathbb R^p`.
Essentially, ADMM is a wrapper over two TAO solver, one for
:math:`f(x)`, and one for :math:`g(z)`. With method of multipliers, one
can form the augmented Lagrangian

.. math:: L_{\rho}(x,z,y) = f(x) + g(z) + y^T(Ax+Bz-c) + (\rho/2)||Ax+Bz-c||_2^2

Then, ADMM consists of the iterations

.. math:: x^{k+1} := \text{argmin}L_{\rho}(x,z^k,y^k)

.. math:: z^{k+1} := \text{argmin}L_{\rho}(x^{k+1},z,y^k)

.. math:: y^{k+1} := y^k + \rho(Ax^{k+1}+Bz^{k+1}-c)

In certain formulation of ADMM, solution of :math:`z^{k+1}` may have
closed-form solution. Currently ADMM provides one default implementation
for :math:`z^{k+1}`, which is soft-threshold. It can be used with either
``TaoADMMSetRegularizerType_ADMM()`` or
``-tao_admm_regularizer_type <regularizer_soft_thresh>``. User can also
pass spectral penalty value, :math:`\rho`, with either
``TaoADMMSetSpectralPenalty()`` or ``-tao_admm_spectral_penalty``.
Currently, user can use

-  ``TaoADMMSetMisfitObjectiveAndGradientRoutine()``

-  ``TaoADMMSetRegularizerObjectiveAndGradientRoutine()``

-  ``TaoADMMSetMisfitHessianRoutine()``

-  ``TaoADMMSetRegularizerHessianRoutine()``

Any other combination of routines is currently not supported. Hessian
matrices can either be constant or non-constant, of which fact can be
set via ``TaoADMMSetMisfitHessianChangeStatus()``, and
``TaoADMMSetRegularizerHessianChangeStatus()``. Also, it may appear in
certain cases where augmented Lagrangian’s Hessian may become nearly
singular depending on the :math:`\rho`, which may change in the case of
``-tao_admm_dual_update <update_adaptive>, <update_adaptive_relaxed>``.
This issue can be prevented by ``TaoADMMSetMinimumSpectralPenalty()``.

Augmented Lagrangian Method of Multipliers (ALMM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The TAOALMM method solves generally constrained problems of the form

.. math::

   \begin{array}{ll}
   \displaystyle \min_{x} & f(x) \\
   \text{subject to} & g(x) = 0\\
                     & h(x) \geq 0 \\
                     & l \leq x \leq u
   \end{array}

where :math:`g(x)` are equality constraints, :math:`h(x)` are inequality
constraints and :math:`l` and :math:`u` are lower and upper bounds on
the optimization variables, respectively.

TAOALMM converts the above general constrained problem into a sequence
of bound constrained problems at each outer iteration
:math:`k = 1,2,\dots`

.. math::

   \begin{array}{ll}
   \displaystyle \min_{x} & L(x, \lambda_k) \\
   \text{subject to} & l \leq x \leq u
   \end{array}

where :math:`L(x, \lambda_k)` is the augmented Lagrangian merit function
and :math:`\lambda_k` is the Lagrange multiplier estimates at outer
iteration :math:`k`.

TAOALMM offers two versions of the augmented Lagrangian formulation: the
canonical Hestenes-Powell augmented
Lagrangian :cite:`hestenes1969multiplier` :cite:`powell1969method`
with inequality constrained converted to equality constraints via slack
variables, and the slack-less Powell-Hestenes-Rockafellar
formulation :cite:`rockafellar1974augmented` that utilizes a
pointwise ``max()`` on the inequality constraints. For most
applications, the canonical Hestenes-Powell formulation is likely to
perform better. However, the PHR formulation may be desirable for
problems featuring very large numbers of inequality constraints as it
avoids inflating the dimension of the subproblem with slack variables.

The inner subproblem is solved using a nested bound-constrained
first-order TAO solver. By default, TAOALM uses a quasi-Newton-Krylov
trust-region method (TAOBQNKTR). Other first-order methods such as
TAOBNCG and TAOBQNLS are also appropriate, but a trust-region
globalization is strongly recommended for most applications.

Primal-Dual Interior-Point Method (PDIPM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The TAOPDIPM method (``-tao_type pdipm``) implements a primal-dual interior
point method for solving general nonlinear programming problems of the form

.. math::
   :label: eq_nlp_gen1

   \begin{array}{ll}
   \displaystyle \min_{x} & f(x) \\
   \text{subject to} & g(x) = 0 \\
                     & h(x) \geq 0 \\
                     & x^- \leq x \leq x^+
   \end{array}

Here, :math:`f(x)` is the nonlinear objective function, :math:`g(x)`,
:math:`h(x)` are the equality and inequality constraints, and
:math:`x^-` and :math:`x^+` are the lower and upper bounds on decision
variables :math:`x`.

PDIPM converts the inequality constraints to equalities using slack variables
:math:`z` and a log-barrier term, which transforms :eq:`eq_nlp_gen1` to

.. math::
   :label: eq_nlp_gen2

   \begin{aligned}
       \text{min}~&f(x) - \mu\sum_{i=1}^{nci}\ln z_i\\
       \text{s.t.}& \\
           &ce(x) = 0 \\
           &ci(x) - z = 0 \\
       \end{aligned}

Here, :math:`ce(x)` is set of equality constraints that include
:math:`g(x)` and fixed decision variables, i.e., :math:`x^- = x = x^+`.
Similarly, :math:`ci(x)` are inequality constraints including
:math:`h(x)` and lower/upper/box-constraints on :math:`x`. :math:`\mu`
is a parameter that is driven to zero as the optimization progresses.

The Lagrangian for :eq:`eq_nlp_gen2`) is

.. math::
   :label: eq_lagrangian

   L_{\mu}(x,\lambda_{ce},\lambda_{ci},z) = f(x) + \lambda_{ce}^Tce(x) - \lambda_{ci}^T(ci(x) - z) - \mu\sum_{i=1}^{nci}\ln z_i

where, :math:`\lambda_{ce}` and :math:`\lambda_{ci}` are the Lagrangian
multipliers for the equality and inequality constraints, respectively.

The first order KKT conditions for optimality are as follows

.. math::
   :label: eq_nlp_kkt

   \nabla L_{\mu}(x,\lambda_{ce},\lambda_{ci},z)    =
       \begin{bmatrix}
           \nabla f(x) + \nabla ce(x)^T\lambda_{ce} -  \nabla ci(x)^T \lambda_{ci} \\
           ce(x) \\
           ci(x) - z \\
           Z\Lambda_{ci}e - \mu e
       \end{bmatrix}
   = 0

:eq:`eq_nlp_kkt` is solved iteratively using Newton’s
method using PETSc’s SNES object. After each Newton iteration, a
line-search is performed to update :math:`x` and enforce
:math:`z,\lambda_{ci} \geq 0`. The barrier parameter :math:`\mu` is also
updated after each Newton iteration. The Newton update is obtained by
solving the second-order KKT system :math:`Hd = -\nabla L_{\mu}`.
Here,\ :math:`H` is the Hessian matrix of the KKT system. For
interior-point methods such as PDIPM, the Hessian matrix tends to be
ill-conditioned, thus necessitating the use of a direct solver. We
recommend using LU preconditioner ``-pc_type lu`` and using direct
linear solver packages such ``SuperLU_Dist`` or ``MUMPS``.

PDE-Constrained Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TAO solves PDE-constrained optimization problems of the form

.. math::

   \begin{array}{ll}
   \displaystyle \min_{u,v} & f(u,v) \\
   \text{subject to} & g(u,v) = 0,
   \end{array}

where the state variable :math:`u` is the solution to the discretized
partial differential equation defined by :math:`g` and parametrized by
the design variable :math:`v`, and :math:`f` is an objective function.
The Lagrange multipliers on the constraint are denoted by :math:`y`.
This method is set by using the linearly constrained augmented
Lagrangian TAO solver ``tao_lcl``.

We make two main assumptions when solving these problems: the objective
function and PDE constraints have been discretized so that we can treat
the optimization problem as finite dimensional and
:math:`\nabla_u g(u,v)` is invertible for all :math:`u` and :math:`v`.

.. _sec_lcl:

Linearly-Constrained Augmented Lagrangian Method (LCL)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Given the current iterate :math:`(u_k, v_k, y_k)`, the linearly
constrained augmented Lagrangian method approximately solves the
optimization problem

.. math::

   \begin{array}{ll}
   \displaystyle \min_{u,v} & \tilde{f}_k(u, v) \\
   \text{subject to} & A_k (u-u_k) + B_k (v-v_k) + g_k = 0,
   \end{array}

where :math:`A_k = \nabla_u g(u_k,v_k)`,
:math:`B_k = \nabla_v g(u_k,v_k)`, and :math:`g_k = g(u_k, v_k)` and

.. math:: \tilde{f}_k(u,v) = f(u,v) - g(u,v)^T y^k + \frac{\rho_k}{2} \| g(u,v) \|^2

is the augmented Lagrangian function. This optimization problem is
solved in two stages. The first computes the Newton direction and finds
a feasible point for the linear constraints. The second computes a
reduced-space direction that maintains feasibility with respect to the
linearized constraints and improves the augmented Lagrangian merit
function.

Newton Step
"""""""""""

The Newton direction is obtained by fixing the design variables at their
current value and solving the linearized constraint for the state
variables. In particular, we solve the system of equations

.. math:: A_k du = -g_k

to obtain a direction :math:`du`. We need a direction that provides
sufficient descent for the merit function

.. math:: \frac{1}{2} \|g(u,v)\|^2.

That is, we require :math:`g_k^T A_k du < 0`.

If the Newton direction is a descent direction, then we choose a penalty
parameter :math:`\rho_k` so that :math:`du` is also a sufficient descent
direction for the augmented Lagrangian merit function. We then find
:math:`\alpha` to approximately minimize the augmented Lagrangian merit
function along the Newton direction.

.. math:: \displaystyle \min_{\alpha \geq 0} \; \tilde{f}_k(u_k + \alpha du, v_k).

We can enforce either the sufficient decrease condition or the Wolfe
conditions during the search procedure. The new point,

.. math::

   \begin{array}{lcl}
   u_{k+\frac{1}{2}} & = & u_k + \alpha_k du \\
   v_{k+\frac{1}{2}} & = & v_k,
   \end{array}

satisfies the linear constraint

.. math:: A_k (u_{k+\frac{1}{2}} - u_k) + B_k (v_{k+\frac{1}{2}} - v_k) + \alpha_k g_k = 0.

If the Newton direction computed does not provide descent for the merit
function, then we can use the steepest descent direction
:math:`du = -A_k^T g_k` during the search procedure. However, the
implication that the intermediate point approximately satisfies the
linear constraint is no longer true.

Modified Reduced-Space Step
"""""""""""""""""""""""""""

We are now ready to compute a reduced-space step for the modified
optimization problem:

.. math::

   \begin{array}{ll}
   \displaystyle \min_{u,v} & \tilde{f}_k(u, v) \\
   \text{subject to} & A_k (u-u_k) + B_k (v-v_k) + \alpha_k g_k = 0.
   \end{array}

We begin with the change of variables

.. math::

   \begin{array}{ll}
   \displaystyle \min_{du,dv} & \tilde{f}_k(u_k+du, v_k+dv) \\
   \text{subject to} & A_k du + B_k dv + \alpha_k g_k = 0
   \end{array}

and make the substitution

.. math:: du = -A_k^{-1}(B_k dv + \alpha_k g_k).

Hence, the unconstrained optimization problem we need to solve is

.. math::

   \begin{array}{ll}
   \displaystyle \min_{dv} & \tilde{f}_k(u_k-A_k^{-1}(B_k dv + \alpha_k g_k), v_k+dv), \\
   \end{array}

which is equivalent to

.. math::

   \begin{array}{ll}
   \displaystyle \min_{dv} & \tilde{f}_k(u_{k+\frac{1}{2}} - A_k^{-1} B_k dv, v_{k+\frac{1}{2}}+dv). \\
   \end{array}

We apply one step of a limited-memory quasi-Newton method to this
problem. The direction is obtain by solving the quadratic problem

.. math::

   \begin{array}{ll}
   \displaystyle \min_{dv} & \frac{1}{2} dv^T \tilde{H}_k dv + \tilde{g}_{k+\frac{1}{2}}^T dv,
   \end{array}

where :math:`\tilde{H}_k` is the limited-memory quasi-Newton
approximation to the reduced Hessian matrix, a positive-definite matrix,
and :math:`\tilde{g}_{k+\frac{1}{2}}` is the reduced gradient.

.. math::

   \begin{array}{lcl}
   \tilde{g}_{k+\frac{1}{2}} & = & \nabla_v \tilde{f}_k(u_{k+\frac{1}{2}}, v_{k+\frac{1}{2}}) -
             \nabla_u \tilde{f}_k(u_{k+\frac{1}{2}}, v_{k+\frac{1}{2}}) A_k^{-1} B_k \\
          & = & d_{k+\frac{1}{2}} + c_{k+\frac{1}{2}} A_k^{-1} B_k
   \end{array}

The reduced gradient is obtained from one linearized adjoint solve

.. math:: y_{k+\frac{1}{2}} = A_k^{-T}c_{k+\frac{1}{2}}

and some linear algebra

.. math:: \tilde{g}_{k+\frac{1}{2}} = d_{k+\frac{1}{2}} + y_{k+\frac{1}{2}}^T B_k.

Because the Hessian approximation is positive definite and we know its
inverse, we obtain the direction

.. math:: dv = -H_k^{-1} \tilde{g}_{k+\frac{1}{2}}

and recover the full-space direction from one linearized forward solve,

.. math:: du = -A_k^{-1} B_k dv.

Having the full-space direction, which satisfies the linear constraint,
we now approximately minimize the augmented Lagrangian merit function
along the direction.

.. math::

   \begin{array}{lcl}
   \displaystyle \min_{\beta \geq 0} & \tilde{f_k}(u_{k+\frac{1}{2}} + \beta du, v_{k+\frac{1}{2}} + \beta dv)
   \end{array}

We enforce the Wolfe conditions during the search procedure. The new
point is

.. math::

   \begin{array}{lcl}
   u_{k+1} & = & u_{k+\frac{1}{2}} + \beta_k du \\
   v_{k+1} & = & v_{k+\frac{1}{2}} + \beta_k dv.
   \end{array}

The reduced gradient at the new point is computed from

.. math::

   \begin{array}{lcl}
   y_{k+1} & = & A_k^{-T}c_{k+1} \\
   \tilde{g}_{k+1} & = & d_{k+1} - y_{k+1}^T B_k,
   \end{array}

where :math:`c_{k+1} = \nabla_u \tilde{f}_k (u_{k+1},v_{k+1})` and
:math:`d_{k+1} = \nabla_v \tilde{f}_k (u_{k+1},v_{k+1})`. The
multipliers :math:`y_{k+1}` become the multipliers used in the next
iteration of the code. The quantities :math:`v_{k+\frac{1}{2}}`,
:math:`v_{k+1}`, :math:`\tilde{g}_{k+\frac{1}{2}}`, and
:math:`\tilde{g}_{k+1}` are used to update :math:`H_k` to obtain the
limited-memory quasi-Newton approximation to the reduced Hessian matrix
used in the next iteration of the code. The update is skipped if it
cannot be performed.

.. _sec_leastsquares:

Nonlinear Least-Squares
~~~~~~~~~~~~~~~~~~~~~~~

Given a function :math:`F: \mathbb R^n \to \mathbb R^m`, the nonlinear
least-squares problem minimizes

.. math::
   :label: eq_nlsf

   f(x)= \| F(x) \|_2^2 = \sum_{i=1}^m F_i(x)^2.

The nonlinear equations :math:`F` should be specified with the function
``TaoSetResidual()``.

.. _sec_pounders:

Bound-constrained Regularized Gauss-Newton (BRGN)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The TAOBRGN algorithms is a Gauss-Newton method is used to iteratively solve nonlinear least
squares problem with the iterations

.. math:: x_{k+1} = x_k - \alpha_k(J_k^T J_k)^{-1} J_k^T r(x_k)

where :math:`r(x)` is the least-squares residual vector,
:math:`J_k = \partial r(x_k)/\partial x` is the Jacobian of the
residual, and :math:`\alpha_k` is the step length parameter. In other
words, the Gauss-Newton method approximates the Hessian of the objective
as :math:`H_k \approx (J_k^T J_k)` and the gradient of the objective as
:math:`g_k \approx -J_k r(x_k)`. The least-squares Jacobian, :math:`J`,
should be provided to Tao using ``TaoSetJacobianResidual()`` routine.

The BRGN (``-tao_type brgn``) implementation adds a regularization term :math:`\beta(x)` such
that

.. math:: \min_{x} \; \frac{1}{2}||R(x)||_2^2 + \lambda\beta(x),

where :math:`\lambda` is the scalar weight of the regularizer. BRGN
provides two default implementations for :math:`\beta(x)`:

-  **L2-norm** - :math:`\beta(x) = \frac{1}{2}||x_k||_2^2`

-  **L2-norm Proximal Point** -
   :math:`\beta(x) = \frac{1}{2}||x_k - x_{k-1}||_2^2`

-  **L1-norm with Dictionary** -
   :math:`\beta(x) = ||Dx||_1 \approx \sum_{i} \sqrt{y_i^2 + \epsilon^2}-\epsilon`
   where :math:`y = Dx` and :math:`\epsilon` is the smooth approximation
   parameter.

The regularizer weight can be controlled with either
``TaoBRGNSetRegularizerWeight()`` or ``-tao_brgn_regularizer_weight``
command line option, while the smooth approximation parameter can be set
with either ``TaoBRGNSetL1SmoothEpsilon()`` or
``-tao_brgn_l1_smooth_epsilon``. For the L1-norm term, the user can
supply a dictionary matrix with ``TaoBRGNSetDictionaryMatrix()``. If no
dictionary is provided, the dictionary is assumed to be an identity
matrix and the regularizer reduces to a sparse solution term.

The regularization selection can be made using the command line option
``-tao_brgn_regularization_type <l2pure, l2prox, l1dict, user>`` where the ``user`` option allows
the user to define a custom :math:`\mathcal{C}2`-continuous
regularization term. This custom term can be defined by using the
interface functions:

-  ``TaoBRGNSetRegularizerObjectiveAndGradientRoutine()`` - Provide
   user-call back for evaluating the function value and gradient
   evaluation for the regularization term.

-  ``TaoBRGNSetRegularizerHessianRoutine()`` - Provide user call-back
   for evaluating the Hessian of the regularization term.

POUNDERS
^^^^^^^^

One algorithm for solving the least squares problem
(:eq:`eq_nlsf`) when the Jacobian of the residual vector
:math:`F` is unavailable is the model-based POUNDERS (Practical
Optimization Using No Derivatives for sums of Squares) algorithm
(``tao_pounders``). POUNDERS employs a derivative-free trust-region
framework as described in :cite:`Dfobook` in order to
converge to local minimizers. An example of this version of POUNDERS
applied to a practical least-squares problem can be found in
:cite:`UNEDF0`.

Derivative-Free Trust-Region Algorithm
""""""""""""""""""""""""""""""""""""""

In each iteration :math:`k`, the algorithm maintains a model
:math:`m_k(x)`, described below, of the nonlinear least squares function
:math:`f` centered about the current iterate :math:`x_k`.

If one assumes that the maximum number of function evaluations has not
been reached and that :math:`\|\nabla m_k(x_k)\|_2>`\ ``gtol``, the next
point :math:`x_+` to be evaluated is obtained by solving the
trust-region subproblem

.. math::
   :label: eq_poundersp

   \min\left\{
    m_k(x) :
    \|x-x_k\|_{p} \leq \Delta_k,
    \right \},

where :math:`\Delta_k` is the current trust-region radius. By default we
use a trust-region norm with :math:`p=\infty` and solve
(:eq:`eq_poundersp`) with the BLMVM method described in
:any:`sec_blmvm`. While the subproblem is a
bound-constrained quadratic program, it may not be convex and the BQPIP
and GPCG methods may not solve the subproblem. Therefore, a bounded
Newton-Krylov Method should be used; the default is the BNTR
algorithm.  Note: BNTR uses its own internal
trust region that may interfere with the infinity-norm trust region used
in the model problem (:eq:`eq_poundersp`).

The residual vector is then evaluated to obtain :math:`F(x_+)` and hence
:math:`f(x_+)`. The ratio of actual decrease to predicted decrease,

.. math:: \rho_k = \frac{f(x_k)-f(x_+)}{m_k(x_k)-m_k(x_+)},

as well as an indicator, ``valid``, on the model’s quality of
approximation on the trust region is then used to update the iterate,

.. math::

   x_{k+1} = \left\{\begin{array}{ll}
   x_+ & \text{if } \rho_k \geq \eta_1 \\
   x_+ & \text{if } 0<\rho_k <\eta_1  \text{ and \texttt{valid}=\texttt{true}}
   \\
   x_k & \text{else},
   \end{array}
   \right.

and trust-region radius,

.. math::

   \Delta_{k+1} = \left\{\begin{array}{ll}
    \text{min}(\gamma_1\Delta_k, \Delta_{\max}) & \text{if } \rho_k \geq
   \eta_1 \text{ and } \|x_+-x_k\|_p\geq \omega_1\Delta_k \\
   \gamma_0\Delta_k & \text{if } \rho_k < \eta_1 \text{ and
   \texttt{valid}=\texttt{true}} \\
   \Delta_k &  \text{else,}
   \end{array}
   \right.

where :math:`0 < \eta_1 < 1`, :math:`0 < \gamma_0 < 1 < \gamma_1`,
:math:`0<\omega_1<1`, and :math:`\Delta_{\max}` are constants.

If :math:`\rho_k\leq 0` and ``valid`` is ``false``, the iterate and
trust-region radius remain unchanged after the above updates, and the
algorithm tests whether the direction :math:`x_+-x_k` improves the
model. If not, the algorithm performs an additional evaluation to obtain
:math:`F(x_k+d_k)`, where :math:`d_k` is a model-improving direction.

The iteration counter is then updated, and the next model :math:`m_{k}`
is obtained as described next.

Forming the Trust-Region Model
""""""""""""""""""""""""""""""

In each iteration, POUNDERS uses a subset of the available evaluated
residual vectors :math:`\{ F(y_1), F(y_2), \cdots \}` to form an
interpolatory quadratic model of each residual component. The :math:`m`
quadratic models

.. math::
   :label: eq_models

   q_k^{(i)}(x) =
    F_i(x_k) + (x-x_k)^T g_k^{(i)} + \frac{1}{2} (x-x_k)^T H_k^{(i)} (x-x_k),
    \qquad i = 1, \ldots, m

thus satisfy the interpolation conditions

.. math:: q_k^{(i)}(y_j) = F_i(y_j), \qquad i=1, \ldots, m; \, j=1,\ldots , l_k

on a common interpolation set :math:`\{y_1, \cdots , y_{l_k}\}` of size
:math:`l_k\in[n+1,`\ ``npmax``\ :math:`]`.

The gradients and Hessians of the models in
(`:eq:eq_models`) are then used to construct the main
model,

.. math::
  :label: eq_newton2

   m_k(x) = f(x_k) +
   2(x-x_k)^T \sum_{i=1}^{m} F_i(x_k) g_k^{(i)} +
   (x-x_k)^T \sum_{i=1}^{m}
   \left( g_k^{(i)} \left(g_k^{(i)}\right)^T +  F_i(x_k)
   H_k^{(i)}\right) (x-x_k).

The process of forming these models also computes the indicator
``valid`` of the model’s local quality.

Parameters
""""""""""

POUNDERS supports the following parameters that can be set from the
command line or PETSc options file:

``-tao_pounders_delta <delta>``
   The initial trust-region radius (:math:`>0`, real). This is used to
   determine the size of the initial neighborhood within which the
   algorithm should look.

``-tao_pounders_npmax <npmax>``
   The maximum number of interpolation points used (:math:`n+2\leq`
   ``npmax`` :math:`\leq 0.5(n+1)(n+2)`). This input is made available
   to advanced users. We recommend the default value
   (``npmax``\ :math:`=2n+1`) be used by others.

``-tao_pounders_gqt``
   Use the gqt algorithm to solve the
   subproblem (:eq:`eq_poundersp`) (uses :math:`p=2`)
   instead of BQPIP.

``-pounders_subsolver``
   If the default BQPIP algorithm is used to solve the
   subproblem (:eq:`eq_poundersp`), the parameters of
   the subproblem solver can be accessed using the command line options
   prefix ``-pounders_subsolver_``. For example,

   ::

             -pounders_subsolver_tao_gatol 1.0e-5

   sets the gradient tolerance of the subproblem solver to
   :math:`10^{-5}`.

Additionally, the user provides an initial solution vector, a vector for
storing the separable objective function, and a routine for evaluating
the residual vector :math:`F`. These are described in detail in
:any:`sec_fghj` and
:any:`sec_evalsof`. Here we remark that because gradient
information is not available for scaling purposes, it can be useful to
ensure that the problem is reasonably well scaled. A simple way to do so
is to rescale the decision variables :math:`x` so that their typical
values are expected to lie within the unit hypercube :math:`[0,1]^n`.

Convergence Notes
"""""""""""""""""

Because the gradient function is not provided to POUNDERS, the norm of
the gradient of the objective function is not available. Therefore, for
convergence criteria, this norm is approximated by the norm of the model
gradient and used only when the model gradient is deemed to be a
reasonable approximation of the gradient of the objective. In practice,
the typical grounds for termination for expensive derivative-free
problems is the maximum number of function evaluations allowed.

.. _sec_complementarity:

Complementarity
~~~~~~~~~~~~~~~

Mixed complementarity problems, or box-constrained variational
inequalities, are related to nonlinear systems of equations. They are
defined by a continuously differentiable function,
:math:`F:\mathbb R^n \to \mathbb R^n`, and bounds,
:math:`\ell \in \{\mathbb R\cup \{-\infty\}\}^n` and
:math:`u \in \{\mathbb R\cup \{\infty\}\}^n`, on the variables such that
:math:`\ell \leq u`. Given this information,
:math:`\mathbf{x}^* \in [\ell,u]` is a solution to
MCP(:math:`F`, :math:`\ell`, :math:`u`) if for each
:math:`i \in \{1, \ldots, n\}` we have at least one of the following:

.. math::

   \begin{aligned}
   \begin{array}{ll}
   F_i(x^*) \geq 0 & \text{if } x^*_i = \ell_i \\
   F_i(x^*) = 0 & \text{if } \ell_i < x^*_i < u_i \\
   F_i(x^*) \leq 0 & \text{if } x^*_i = u_i.
   \end{array}\end{aligned}

Note that when :math:`\ell = \{-\infty\}^n` and
:math:`u = \{\infty\}^n`, we have a nonlinear system of equations, and
:math:`\ell = \{0\}^n` and :math:`u = \{\infty\}^n` correspond to the
nonlinear complementarity problem :cite:`cottle:nonlinear`.

Simple complementarity conditions arise from the first-order optimality
conditions from optimization
:cite:`karush:minima` :cite:`kuhn.tucker:nonlinear`. In the simple
bound-constrained optimization case, these conditions correspond to
MCP(:math:`\nabla f`, :math:`\ell`, :math:`u`), where
:math:`f: \mathbb R^n \to \mathbb R` is the objective function. In a
one-dimensional setting these conditions are intuitive. If the solution
is at the lower bound, then the function must be increasing and
:math:`\nabla f \geq 0`. If the solution is at the upper bound, then the
function must be decreasing and :math:`\nabla f \leq 0`. If the solution
is strictly between the bounds, we must be at a stationary point and
:math:`\nabla f = 0`. Other complementarity problems arise in economics
and engineering :cite:`ferris.pang:engineering`, game theory
:cite:`nash:equilibrium`, and finance
:cite:`huang.pang:option`.

Evaluation routines for :math:`F` and its Jacobian must be supplied
prior to solving the application. The bounds, :math:`[\ell,u]`, on the
variables must also be provided. If no starting point is supplied, a
default starting point of all zeros is used.

Semismooth Methods
^^^^^^^^^^^^^^^^^^

TAO has two implementations of semismooth algorithms
:cite:`munson.facchinei.ea:semismooth` :cite:`deluca.facchinei.ea:semismooth`
:cite:`facchinei.fischer.ea:semismooth` for solving mixed complementarity
problems. Both are based on a reformulation of the mixed complementarity
problem as a nonsmooth system of equations using the Fischer-Burmeister
function :cite:`fischer:special`. A nonsmooth Newton method
is applied to the reformulated system to calculate a solution. The
theoretical properties of such methods are detailed in the
aforementioned references.

The Fischer-Burmeister function, :math:`\phi:\mathbb R^2 \to \mathbb R`,
is defined as

.. math::

   \begin{aligned}
   \phi(a,b) := \sqrt{a^2 + b^2} - a - b.\end{aligned}

This function has the following key property,

.. math::

   \begin{aligned}
   \begin{array}{lcr}
           \phi(a,b) = 0 & \Leftrightarrow & a \geq 0,\; b \geq 0,\; ab = 0,
   \end{array}\end{aligned}

used when reformulating the mixed complementarity problem as the system
of equations :math:`\Phi(x) = 0`, where
:math:`\Phi:\mathbb R^n \to \mathbb R^n`. The reformulation is defined
componentwise as

.. math::

   \begin{aligned}
   \Phi_i(x) := \left\{ \begin{array}{ll}
      \phi(x_i - l_i, F_i(x)) & \text{if } -\infty < l_i < u_i = \infty, \\
      -\phi(u_i-x_i, -F_i(x)) & \text{if } -\infty = l_i < u_i < \infty, \\
      \phi(x_i - l_i, \phi(u_i - x_i, - F_i(x))) & \text{if } -\infty < l_i < u_i < \infty, \\
      -F_i(x) & \text{if } -\infty = l_i < u_i = \infty, \\
      l_i - x_i & \text{if } -\infty < l_i = u_i < \infty.
      \end{array} \right.\end{aligned}

We note that :math:`\Phi` is not differentiable everywhere but satisfies
a semismoothness property
:cite:`mifflin:semismooth` :cite:`qi:convergence` :cite:`qi.sun:nonsmooth`.
Furthermore, the natural merit function,
:math:`\Psi(x) := \frac{1}{2} \| \Phi(x) \|_2^2`, is continuously
differentiable.

The two semismooth TAO solvers both solve the system :math:`\Phi(x) = 0`
by applying a nonsmooth Newton method with a line search. We calculate a
direction, :math:`d^k`, by solving the system
:math:`H^kd^k = -\Phi(x^k)`, where :math:`H^k` is an element of the
:math:`B`-subdifferential :cite:`qi.sun:nonsmooth` of
:math:`\Phi` at :math:`x^k`. If the direction calculated does not
satisfy a suitable descent condition, then we use the negative gradient
of the merit function, :math:`-\nabla \Psi(x^k)`, as the search
direction. A standard Armijo search
:cite:`armijo:minimization` is used to find the new
iteration. Nonmonotone searches
:cite:`grippo.lampariello.ea:nonmonotone` are also available
by setting appropriate runtime options. See
:any:`sec_taolinesearch` for further details.

The first semismooth algorithm available in TAO is not guaranteed to
remain feasible with respect to the bounds, :math:`[\ell, u]`, and is
termed an infeasible semismooth method. This method can be specified by
using the ``tao_ssils`` solver. In this case, the descent test used is
that

.. math::

   \begin{aligned}
   \nabla \Psi(x^k)^Td^k \leq -\delta\| d^k \|^\rho.\end{aligned}

Both :math:`\delta > 0` and :math:`\rho > 2` can be modified by using
the runtime options ``-tao_ssils_delta <delta>`` and
``-tao_ssils_rho <rho>``, respectively. By default,
:math:`\delta = 10^{-10}` and :math:`\rho = 2.1`.

An alternative is to remain feasible with respect to the bounds by using
a projected Armijo line search. This method can be specified by using
the ``tao_ssfls`` solver. The descent test used is the same as above
where the direction in this case corresponds to the first part of the
piecewise linear arc searched by the projected line search. Both
:math:`\delta > 0` and :math:`\rho > 2` can be modified by using the
runtime options ``-tao_ssfls_delta <delta>`` and
``-tao_ssfls_rho <rho>`` respectively. By default,
:math:`\delta = 10^{-10}` and :math:`\rho = 2.1`.

The recommended algorithm is the infeasible semismooth method,
``tao_ssils``, because of its strong global and local convergence
properties. However, if it is known that :math:`F` is not defined
outside of the box, :math:`[\ell,u]`, perhaps because of the presence of
:math:`\log` functions, the feasibility-enforcing version of the
algorithm, ``tao_ssfls``, is a reasonable alternative.

Active-Set Methods
^^^^^^^^^^^^^^^^^^

TAO also contained two active-set semismooth methods for solving
complementarity problems.  These methods solve a reduced system
constructed by block elimination of active constraints.  The
subdifferential in these cases enables this block elimination.

The first active-set semismooth algorithm available in TAO is not guaranteed to
remain feasible with respect to the bounds, :math:`[\ell, u]`, and is
termed an infeasible active-set semismooth method. This method can be
specified by using the ``tao_asils`` solver.

An alternative is to remain feasible with respect to the bounds by using
a projected Armijo line search. This method can be specified by using
the ``tao_asfls`` solver.

Quadratic Solvers
~~~~~~~~~~~~~~~~~

Quadratic solvers solve optimization problems of the form

.. math::

   \begin{array}{ll}
   \displaystyle \min_{x} & \frac{1}{2}x^T Q x + c^T x \\
   \text{subject to} & l \geq x \geq u
   \end{array}

where the gradient and the Hessian of the objective are both constant.

Gradient Projection Conjugate Gradient Method (GPCG)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The GPCG :cite:`more-toraldo` algorithm is much like the
TRON algorithm, discussed in Section :any:`sec_tron`, except that
it assumes that the objective function is quadratic and convex.
Therefore, it evaluates the function, gradient, and Hessian only once.
Since the objective function is quadratic, the algorithm does not use a
trust region. All the options that apply to TRON except for trust-region
options also apply to GPCG. It can be set by using the TAO solver
``tao_gpcg`` or via the optio flag ``-tao_type gpcg``.

.. _sec_bqpip:

Interior-Point Newton’s Method (BQPIP)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The BQPIP algorithm is an interior-point method for bound constrained
quadratic optimization. It can be set by using the TAO solver of
``tao_bqpip`` or via the option flag ``-tao_type bgpip``. Since it
assumes the objective function is quadratic, it evaluates the function,
gradient, and Hessian only once. This method also requires the solution
of systems of linear equations, whose solver can be accessed and
modified with the command ``TaoGetKSP()``.

Legacy and Contributed Solvers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bundle Method for Regularized Risk Minimization (BMRM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

BMRM is a numerical approach to optimizing an
unconstrained objective in the form of
:math:`f(x) + 0.5 * \lambda \| x \|^2`. Here :math:`f` is a convex
function that is finite on the whole space. :math:`\lambda` is a
positive weight parameter, and :math:`\| x \|` is the Euclidean norm of
:math:`x`. The algorithm only requires a routine which, given an
:math:`x`, returns the value of :math:`f(x)` and the gradient of
:math:`f` at :math:`x`.

Orthant-Wise Limited-memory Quasi-Newton (OWLQN)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

OWLQN :cite:`owlqn` is a numerical approach to optimizing
an unconstrained objective in the form of
:math:`f(x) + \lambda \|x\|_1`. Here f is a convex and differentiable
function, :math:`\lambda` is a positive weight parameter, and
:math:`\| x \|_1` is the :math:`\ell_1` norm of :math:`x`:
:math:`\sum_i |x_i|`. The algorithm only requires evaluating the value
of :math:`f` and its gradient.

.. _sec_tron:

Trust-Region Newton Method (TRON)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The TRON :cite:`lin_c3` algorithm is an active-set method
that uses a combination of gradient projections and a preconditioned
conjugate gradient method to minimize an objective function. Each
iteration of the TRON algorithm requires function, gradient, and Hessian
evaluations. In each iteration, the algorithm first applies several
conjugate gradient iterations. After these iterates, the TRON solver
momentarily ignores the variables that equal one of its bounds and
applies a preconditioned conjugate gradient method to a quadratic model
of the remaining set of *free* variables.

The TRON algorithm solves a reduced linear system defined by the rows
and columns corresponding to the variables that lie between the upper
and lower bounds. The TRON algorithm applies a trust region to the
conjugate gradients to ensure convergence. The initial trust-region
radius can be set by using the command
``TaoSetInitialTrustRegionRadius()``, and the current trust region size
can be found by using the command ``TaoGetCurrentTrustRegionRadius()``.
The initial trust region can significantly alter the rate of convergence
for the algorithm and should be tuned and adjusted for optimal
performance.

This algorithm will be deprecated in the next version in favor of the
Bounded Newton Trust Region (BNTR) algorithm.

.. _sec_blmvm:

Bound-constrained Limited-Memory Variable-Metric Method (BLMVM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

BLMVM is a limited-memory, variable-metric method and is the
bound-constrained variant of the LMVM method for unconstrained
optimization. It uses projected gradients to approximate the Hessian,
eliminating the need for Hessian evaluations. The method can be set by
using the TAO solver ``tao_blmvm``. For more details, please see the
LMVM section in the unconstrained algorithms as well as the LMVM matrix
documentation in the PETSc manual.

This algorithm will be deprecated in the next version in favor of the
Bounded Quasi-Newton Line Search (BQNLS) algorithm.

Advanced Options
----------------

This section discusses options and routines that apply to most TAO
solvers and problem classes. In particular, we focus on linear solvers,
convergence tests, and line searches.

.. _sec_taolinearsolvers:

Linear Solvers
~~~~~~~~~~~~~~

One of the most computationally intensive phases of many optimization
algorithms involves the solution of linear systems of equations. The
performance of the linear solver may be critical to an efficient
computation of the solution. Since linear equation solvers often have a
wide variety of options associated with them, TAO allows the user to
access the linear solver with the

.. code-block::

      TaoGetKSP(Tao, KSP *);

command. With access to the KSP object, users can customize it for their
application to achieve improved performance. Additional details on the
KSP options in PETSc can be found in the :doc:`/docs/manual/index`.

Monitors
~~~~~~~~

By default the TAO solvers run silently without displaying information
about the iterations. The user can initiate monitoring with the command

.. code-block::

      TaoSetMonitor(Tao, PetscErrorCode (*mon)(Tao,void*), void*);

The routine ``mon`` indicates a user-defined monitoring routine, and
``void*`` denotes an optional user-defined context for private data for
the monitor routine.

The routine set by ``TaoSetMonitor()`` is called once during each
iteration of the optimization solver. Hence, the user can employ this
routine for any application-specific computations that should be done
after the solution update.

.. _sec_taoconvergence:

Convergence Tests
~~~~~~~~~~~~~~~~~

Convergence of a solver can be defined in many ways. The methods TAO
uses by default are mentioned in :any:`sec_customize`.
These methods include absolute and relative convergence tolerances as
well as a maximum number of iterations of function evaluations. If these
choices are not sufficient, the user can specify a customized test

Users can set their own customized convergence tests of the form

.. code-block::

      PetscErrorCode  conv(Tao, void*);

The second argument is a pointer to a structure defined by the user.
Within this routine, the solver can be queried for the solution vector,
gradient vector, or other statistic at the current iteration through
routines such as ``TaoGetSolutionStatus()`` and ``TaoGetTolerances()``.

To use this convergence test within a TAO solver, one uses the command

.. code-block::

      TaoSetConvergenceTest(Tao, PetscErrorCode (*conv)(Tao,void*), void*);

The second argument of this command is the convergence routine, and the
final argument of the convergence test routine denotes an optional
user-defined context for private data. The convergence routine receives
the TAO solver and this private data structure. The termination flag can
be set by using the routine

.. code-block::

      TaoSetConvergedReason(Tao, TaoConvergedReason);

.. _sec_taolinesearch:

Line Searches
~~~~~~~~~~~~~

By using the command line option ``-tao_ls_type``. Available line
searches include Moré-Thuente :cite:`more:92`, Armijo, gpcg,
and unit.

The line search routines involve several parameters, which are set to
defaults that are reasonable for many applications. The user can
override the defaults by using the following options

-  ``-tao_ls_maxfev <max>``

-  ``-tao_ls_stepmin <min>``

-  ``-tao_ls_stepmax <max>``

-  ``-tao_ls_ftol <ftol>``

-  ``-tao_ls_gtol <gtol>``

-  ``-tao_ls_rtol <rtol>``

One should run a TAO program with the option ``-help`` for details.
Users may write their own customized line search codes by modeling them
after one of the defaults provided.

.. _sec_taorecyclehistory:

Recycling History
~~~~~~~~~~~~~~~~~

Some TAO algorithms can re-use information accumulated in the previous
``TaoSolve()`` call to hot-start the new solution. This can be enabled
using the ``-tao_recycle_history`` flag, or in code via the
``TaoSetRecycleHistory()`` interface.

For the nonlinear conjugate gradient solver (``TAOBNCG``), this option
re-uses the latest search direction from the previous ``TaoSolve()``
call to compute the initial search direction of a new ``TaoSolve()``. By
default, the feature is disabled and the algorithm sets the initial
direction as the negative gradient.

For the quasi-Newton family of methods (``TAOBQNLS``, ``TAOBQNKLS``,
``TAOBQNKTR``, ``TAOBQNKTL``), this option re-uses the accumulated
quasi-Newton Hessian approximation from the previous ``TaoSolve()``
call. By default, the feature is disabled and the algorithm will reset
the quasi-Newton approximation to the identity matrix at the beginning
of every new ``TaoSolve()``.

The option flag has no effect on other TAO solvers.

.. _sec_tao_addsolver:

Adding a Solver
---------------

One of the strengths of both TAO and PETSc is the ability to allow users
to extend the built-in solvers with new user-defined algorithms. It is
certainly possible to develop new optimization algorithms outside of TAO
framework, but Using TAO to implement a solver has many advantages,

#. TAO includes other optimization solvers with an identical interface,
   so application problems may conveniently switch solvers to compare
   their effectiveness.

#. TAO provides support for function evaluations and derivative
   information. It allows for the direct evaluation of this information
   by the application developer, contains limited support for finite
   difference approximations, and allows the uses of matrix-free
   methods. The solvers can obtain this function and derivative
   information through a simple interface while the details of its
   computation are handled within the toolkit.

#. TAO provides line searches, convergence tests, monitoring routines,
   and other tools that are helpful in an optimization algorithm. The
   availability of these tools means that the developers of the
   optimization solver do not have to write these utilities.

#. PETSc offers vectors, matrices, index sets, and linear solvers that
   can be used by the solver. These objects are standard mathematical
   constructions that have many different implementations. The objects
   may be distributed over multiple processors, restricted to a single
   processor, have a dense representation, use a sparse data structure,
   or vary in many other ways. TAO solvers do not need to know how these
   objects are represented or how the operations defined on them have
   been implemented. Instead, the solvers apply these operations through
   an abstract interface that leaves the details to PETSc and external
   libraries. This abstraction allows solvers to work seamlessly with a
   variety of data structures while allowing application developers to
   select data structures tailored for their purposes.

#. PETSc provides the user a convenient method for setting options at
   runtime, performance profiling, and debugging.

.. _header-file-1:

Header File
~~~~~~~~~~~

TAO solver implementation files must include the TAO implementation file
``taoimpl.h``:

.. code-block::

      #include "petsc/private/taoimpl.h"

This file contains data elements that are generally kept hidden from
application programmers, but may be necessary for solver implementations
to access.

TAO Interface with Solvers
~~~~~~~~~~~~~~~~~~~~~~~~~~

TAO solvers must be written in C or C++ and include several routines
with a particular calling sequence. Two of these routines are mandatory:
one that initializes the TAO structure with the appropriate information
and one that applies the algorithm to a problem instance. Additional
routines may be written to set options within the solver, view the
solver, setup appropriate data structures, and destroy these data
structures. In order to implement the conjugate gradient algorithm, for
example, the following structure is useful.

.. code-block::

   typedef struct{

     PetscReal beta;
     PetscReal eta;
     PetscInt  ngradtseps;
     PetscInt  nresetsteps;
     Vec X_old;
     Vec G_old;

   } TAO_CG;

This structure contains two parameters, two counters, and two work
vectors. Vectors for the solution and gradient are not needed here
because the TAO structure has pointers to them.

Solver Routine
^^^^^^^^^^^^^^

All TAO solvers have a routine that accepts a TAO structure and computes
a solution. TAO will call this routine when the application program uses
the routine ``TaoSolve()`` and will pass to the solver information about
the objective function and constraints, pointers to the variable vector
and gradient vector, and support for line searches, linear solvers, and
convergence monitoring. As an example, consider the following code that
solves an unconstrained minimization problem using the conjugate
gradient method.

.. code-block::

   PetscErrorCode TaoSolve_CG(Tao tao)
   {
     TAO_CG  *cg = (TAO_CG *) tao->data;
     Vec x = tao->solution;
     Vec g = tao->gradient;
     Vec s = tao->stepdirection;
     PetscInt     iter=0;
     PetscReal  gnormPrev,gdx,f,gnorm,steplength=0;
     TaoLineSearchConvergedReason lsflag=TAO_LINESEARCH_CONTINUE_ITERATING;
     TaoConvergedReason reason=TAO_CONTINUE_ITERATING;

     PetscFunctionBegin;

     CHKERRQ(TaoComputeObjectiveAndGradient(tao,x,&f,g));
     CHKERRQ(VecNorm(g,NORM_2,&gnorm));

     CHKERRQ(VecSet(s,0));

     cg->beta=0;
     gnormPrev = gnorm;

     /* Enter loop */
     while (1){

       /* Test for convergence */
       CHKERRQ(TaoMonitor(tao,iter,f,gnorm,0.0,step,&reason));
       if (reason!=TAO_CONTINUE_ITERATING) break;

       cg->beta=(gnorm*gnorm)/(gnormPrev*gnormPrev);
       CHKERRQ(VecScale(s,cg->beta));
       CHKERRQ(VecAXPY(s,-1.0,g));

       CHKERRQ(VecDot(s,g,&gdx));
       if (gdx>=0){     /* If not a descent direction, use gradient */
         CHKERRQ(VecCopy(g,s));
         CHKERRQ(VecScale(s,-1.0));
         gdx=-gnorm*gnorm;
       }

       /* Line Search */
       gnormPrev = gnorm;  step=1.0;
       CHKERRQ(TaoLineSearchSetInitialStepLength(tao->linesearch,1.0));
       CHKERRQ(TaoLineSearchApply(tao->linesearch,x,&f,g,s,&steplength,&lsflag));
       CHKERRQ(TaoAddLineSearchCounts(tao));
       CHKERRQ(VecNorm(g,NORM_2,&gnorm));
       iter++;
     }

     PetscFunctionReturn(0);
   }

The first line of this routine casts the second argument to a pointer to
a ``TAO_CG`` data structure. This structure contains pointers to three
vectors and a scalar that will be needed in the algorithm.

After declaring an initializing several variables, the solver lets TAO
evaluate the function and gradient at the current point in the using the
routine ``TaoComputeObjectiveAndGradient()``. Other routines may be used
to evaluate the Hessian matrix or evaluate constraints. TAO may obtain
this information using direct evaluation or other means, but these
details do not affect our implementation of the algorithm.

The norm of the gradient is a standard measure used by unconstrained
minimization solvers to define convergence. This quantity is always
nonnegative and equals zero at the solution. The solver will pass this
quantity, the current function value, the current iteration number, and
a measure of infeasibility to TAO with the routine

.. code-block::

      PetscErrorCode TaoMonitor(Tao tao, PetscInt iter, PetscReal f,
                     PetscReal res, PetscReal cnorm, PetscReal steplength,
                     TaoConvergedReason *reason);

Most optimization algorithms are iterative, and solvers should include
this command somewhere in each iteration. This routine records this
information, and applies any monitoring routines and convergence tests
set by default or the user. In this routine, the second argument is the
current iteration number, and the third argument is the current function
value. The fourth argument is a nonnegative error measure associated
with the distance between the current solution and the optimal solution.
Examples of this measure are the norm of the gradient or the square root
of a duality gap. The fifth argument is a nonnegative error that usually
represents a measure of the infeasibility such as the norm of the
constraints or violation of bounds. This number should be zero for
unconstrained solvers. The sixth argument is a nonnegative steplength,
or the multiple of the step direction added to the previous iterate. The
results of the convergence test are returned in the last argument. If
the termination reason is ``TAO_CONTINUE_ITERATING``, the algorithm
should continue.

After this monitoring routine, the solver computes a step direction
using the conjugate gradient algorithm and computations using Vec
objects. These methods include adding vectors together and computing an
inner product. A full list of these methods can be found in the manual
pages.

Nonlinear conjugate gradient algorithms also require a line search. TAO
provides several line searches and support for using them. The routine

.. code-block::

      TaoLineSearchApply(TaoLineSearch ls, Vec x, PetscReal *f, Vec g,
                             TaoVec *s, PetscReal *steplength,
                             TaoLineSearchConvergedReason *lsflag)

passes the current solution, gradient, and objective value to the line
search and returns a new solution, gradient, and objective value. More
details on line searches can be found in
:any:`sec_taolinesearch`. The details of the
line search applied are specified elsewhere, when the line search is
created.

TAO also includes support for linear solvers using PETSc KSP objects.
Although this algorithm does not require one, linear solvers are an
important part of many algorithms. Details on the use of these solvers
can be found in the PETSc users manual.

Creation Routine
^^^^^^^^^^^^^^^^

The TAO solver is initialized for a particular algorithm in a separate
routine. This routine sets default convergence tolerances, creates a
line search or linear solver if needed, and creates structures needed by
this solver. For example, the routine that creates the nonlinear
conjugate gradient algorithm shown above can be implemented as follows.

.. code-block::

   PETSC_EXTERN PetscErrorCode TaoCreate_CG(Tao tao)
   {
     TAO_CG *cg = (TAO_CG*)tao->data;
     const char *morethuente_type = TAOLINESEARCH_MT;

     PetscFunctionBegin;

     CHKERRQ(PetscNewLog(tao,&cg));
     tao->data = (void*)cg;
     cg->eta = 0.1;
     cg->delta_min = 1e-7;
     cg->delta_max = 100;
     cg->cg_type = CG_PolakRibierePlus;

     tao->max_it = 2000;
     tao->max_funcs = 4000;

     tao->ops->setup = TaoSetUp_CG;
     tao->ops->solve = TaoSolve_CG;
     tao->ops->view = TaoView_CG;
     tao->ops->setfromoptions = TaoSetFromOptions_CG;
     tao->ops->destroy = TaoDestroy_CG;

     CHKERRQ(TaoLineSearchCreate(((PetscObject)tao)->comm, &tao->linesearch));
     CHKERRQ(TaoLineSearchSetType(tao->linesearch, morethuente_type));
     CHKERRQ(TaoLineSearchUseTaoRoutines(tao->linesearch, tao));

     PetscFunctionReturn(0);
   }
   EXTERN_C_END

This routine declares some variables and then allocates memory for the
``TAO_CG`` data structure. Notice that the ``Tao`` object now has a
pointer to this data structure (``tao->data``) so it can be accessed by
the other functions written for this solver implementation.

This routine also sets some default parameters particular to the
conjugate gradient algorithm, sets default convergence tolerances, and
creates a particular line search. These defaults could be specified in
the routine that solves the problem, but specifying them here gives the
user the opportunity to modify these parameters either by using direct
calls setting parameters or by using options.

Finally, this solver passes to TAO the names of all the other routines
used by the solver.

Note that the lines ``EXTERN_C_BEGIN`` and ``EXTERN_C_END`` surround
this routine. These macros are required to preserve the name of this
function without any name-mangling from the C++ compiler (if used).

Destroy Routine
^^^^^^^^^^^^^^^

Another routine needed by most solvers destroys the data structures
created by earlier routines. For the nonlinear conjugate gradient method
discussed earlier, the following routine destroys the two work vectors
and the ``TAO_CG`` structure.

.. code-block::

   PetscErrorCode TaoDestroy_CG(TAO_SOLVER tao)
   {
     TAO_CG *cg = (TAO_CG *) tao->data;

     PetscFunctionBegin;

     CHKERRQ(VecDestroy(&cg->X_old));
     CHKERRQ(VecDestroy(&cg->G_old));

     PetscFree(tao->data);
     tao->data = NULL;

     PetscFunctionReturn(0);
   }

This routine is called from within the ``TaoDestroy()`` routine. Only
algorithm-specific data objects are destroyed in this routine; any
objects indexed by TAO (``tao->linesearch``, ``tao->ksp``,
``tao->gradient``, etc.) will be destroyed by TAO immediately after the
algorithm-specific destroy routine completes.

SetUp Routine
^^^^^^^^^^^^^

If the SetUp routine has been set by the initialization routine, TAO
will call it during the execution of ``TaoSolve()``. While this routine
is optional, it is often provided to allocate the gradient vector, work
vectors, and other data structures required by the solver. It should
have the following form.

.. code-block::

   PetscErrorCode TaoSetUp_CG(Tao tao)
   {
     TAO_CG *cg = (TAO_CG*)tao->data;
     PetscFunctionBegin;

     CHKERRQ(VecDuplicate(tao->solution,&tao->gradient));
     CHKERRQ(VecDuplicate(tao->solution,&tao->stepdirection));
     CHKERRQ(VecDuplicate(tao->solution,&cg->X_old));
     CHKERRQ(VecDuplicate(tao->solution,&cg->G_old));

     PetscFunctionReturn(0);
   }

SetFromOptions Routine
^^^^^^^^^^^^^^^^^^^^^^

The SetFromOptions routine should be used to check for any
algorithm-specific options set by the user and will be called when the
application makes a call to ``TaoSetFromOptions()``. It should have the
following form.

.. code-block::

   PetscErrorCode TaoSetFromOptions_CG(Tao tao, void *solver);
   {
     TAO_CG *cg = (TAO_CG*)solver;
     PetscFunctionBegin;
     CHKERRQ(PetscOptionsReal("-tao_cg_eta","restart tolerance","",cg->eta,&cg->eta,0));
     CHKERRQ(PetscOptionsReal("-tao_cg_delta_min","minimum delta value","",cg->delta_min,&cg->delta_min,0));
     CHKERRQ(PetscOptionsReal("-tao_cg_delta_max","maximum delta value","",cg->delta_max,&cg->delta_max,0));
     PetscFunctionReturn(0);
   }

View Routine
^^^^^^^^^^^^

The View routine should be used to output any algorithm-specific
information or statistics at the end of a solve. This routine will be
called when the application makes a call to ``TaoView()`` or when the
command line option ``-tao_view`` is used. It should have the following
form.

.. code-block::

   PetscErrorCode TaoView_CG(Tao tao, PetscViewer viewer)
   {
     TAO_CG *cg = (TAO_CG*)tao->data;

     PetscFunctionBegin;
     CHKERRQ(PetscViewerASCIIPushTab(viewer));
     CHKERRQ(PetscViewerASCIIPrintf(viewer,"Grad. steps: %d\n",cg->ngradsteps));
     CHKERRQ(PetscViewerASCIIPrintf(viewer,"Reset steps: %d\n",cg->nresetsteps));
     CHKERRQ(PetscViewerASCIIPopTab(viewer));
     PetscFunctionReturn(0);
   }

Registering the Solver
^^^^^^^^^^^^^^^^^^^^^^

Once a new solver is implemented, TAO needs to know the name of the
solver and what function to use to create the solver. To this end, one
can use the routine

.. code-block::

     TaoRegister(const char *name,
                     const char *path,
                     const char *cname,
                     PetscErrorCode (*create) (Tao));

where ``name`` is the name of the solver (i.e., ``tao_blmvm``), ``path``
is the path to the library containing the solver, ``cname`` is the name
of the routine that creates the solver (in our case, ``TaoCreate_CG``),
and ``create`` is a pointer to that creation routine. If one is using
dynamic loading, then the fourth argument will be ignored.

Once the solver has been registered, the new solver can be selected
either by using the ``TaoSetType()`` function or by using the
``-tao_type`` command line option.

.. rubric:: Footnotes

.. [#mpi] For more on MPI and PETSc, see :any:`sec-running`.

.. raw:: html

    <hr>

.. bibliography:: /petsc.bib
   :filter: docname in docnames
