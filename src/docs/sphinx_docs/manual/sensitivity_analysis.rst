.. _chapter_sa:

Performing sensitivity analysis
-------------------------------

The ``TS`` library provides a framework based on discrete adjoint models
for sensitivity analysis for ODEs and DAEs. The ODE/DAE solution process
(henceforth called the forward run) can be obtained by using either
explicit or implicit solvers in ``TS``, depending on the problem
properties. Currently supported method types are ``TSRK`` (Runge-Kutta)
explicit methods and ``TSTHETA`` implicit methods, which include
``TSBEULER`` and ``TSCN``.

Using the discrete adjoint methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Consider the ODE/DAE

.. math:: F(t,y,\dot{y},p) = 0, \quad y(t_0)=y_0(p) \quad t_0 \le t \le t_F

and the cost function(s)

.. math:: \Psi_i(y_0,p) = \Phi_i(y_F,p) + \int_{t_0}^{t_F} r_i(y(t),p,t)dt \quad i=1,...,n_\text{cost}.

The ``TSAdjoint`` routines of PETSc provide

.. math:: \frac{\partial \Psi_i}{\partial y_0} = \lambda_i

and

.. math:: \frac{\partial \Psi_i}{\partial p} = \mu_i + \lambda_i (\frac{\partial y_0}{\partial p}).

To perform the discrete adjoint sensitivity analysis one first sets up
the ``TS`` object for a regular forward run but with one extra function
call

::

   TSSetSaveTrajectory(TS ts),

then calls ``TSSolve()`` in the usual manner.

One must create two arrays of :math:`n_\text{cost}` vectors
:math:`\lambda` and\ :math:`\mu` (if there are no parameters :math:`p`
then one can use ``NULL`` for the :math:`\mu` array.) The
:math:`\lambda` vectors are the same dimension and parallel layout as
the solution vector for the ODE, the :math:`mu` vectors are of dimension
:math:`p`; when :math:`p` is small usually all its elements are on the
first MPI process, while the vectors have no entries on the other
processes. :math:`\lambda_i` and :math:`mu_i` should be initialized with
the values :math:`d\Phi_i/dy|_{t=t_F}` and :math:`d\Phi_i/dp|_{t=t_F}`
respectively. Then one calls

::

   TSSetCostGradients(TS ts,PetscInt numcost, Vec *lambda,Vec *mu);

If :math:`F()` is a function of :math:`p` one needs to also provide the
Jacobian :math:`-F_p` with

::

   TSSetRHSJacobianP(TS ts,Mat Amat,PetscErrorCode (*fp)(TS,PetscReal,Vec,Mat,void*),void *ctx)

The arguments for the function ``fp()`` are the timestep context,
current time, :math:`y`, and the (optional) user-provided context.

If there is an integral term in the cost function, i.e. :math:`r` is
nonzero, it can be transformed into another ODE that is augmented to the
original ODE. To evaluate the integral, one needs to create a child
``TS`` objective by calling

::

   TSCreateQuadratureTS(TS ts,PetscBool fwd,TS *quadts);

and provide the ODE RHS function (which evaluates the integrand
:math:`r`) with

::

   TSSetRHSFunction(TS quadts,Vec R,PetscErrorCode (*rf)(TS,PetscReal,Vec,Vec,void*),void *ctx)

Similar to the settings for the original ODE, Jacobians of the integrand
can be provided with

::

   TSSetRHSJacobian(TS quadts,Vec DRDU,Vec DRDU,PetscErrorCode (*drdyf)(TS,PetscReal,Vec,Vec*,void*),void *ctx)
   TSSetRHSJacobianP(TS quadts,Vec DRDU,Vec DRDU,PetscErrorCode (*drdyp)(TS,PetscReal,Vec,Vec*,void*),void *ctx)

where :math:`\mathrm{drdyf}= dr /dy`, :math:`\mathrm{drdpf} = dr /dp`.
Since the integral term is additive to the cost function, its gradient
information will be included in :math:`\lambda` and :math:`\mu`.

Lastly, one starts the backward run by calling

::

   TSAdjointSolve(TS ts).

One can obtain the value of the integral term by calling

::

   TSGetCostIntegral(TS ts,Vec *q).

or accessing directly the solution vector used by quadts.

The second argument of ``TSCreateQuadratureTS()`` allows one to choose
if the integral term is evaluated in the forward run (inside
``TSSolve()``) or in the backward run (inside ``TSAdjointSolve()``) when
``TSSetCostGradients()`` and ``TSSetCostIntegrand()`` are called before
``TSSolve()``. Note that this also allows for evaluating the integral
without having to use the adjoint solvers.

To provide a better understanding of the use of the adjoint solvers, we
introduce a simple example, corresponding to
`TS Power Grid Tutorial ex3adj <../../src/ts/tutorials/power_grid/ex3adj.c.html>`__.
The problem is to study dynamic security of power system when there are
credible contingencies such as short-circuits or loss of generators,
transmission lines, or loads. The dynamic security constraints are
incorporated as equality constraints in the form of discretized
differential equations and inequality constraints for bounds on the
trajectory. The governing ODE system is

.. math::

   \begin{aligned}
       \phi' &= &\omega_B (\omega - \omega_S)  \\
       2H/\omega_S \, \omega' & =& p_m - p_{max} sin(\phi) -D (\omega - \omega_S), \quad t_0 \leq t \leq t_F,\end{aligned}

where :math:`\phi` is the phase angle and :math:`\omega` is the
frequency.

The initial conditions at time :math:`t_0` are

.. math::

   \begin{aligned}
   \phi(t_0) &=& \arcsin \left( p_m / p_{max} \right), \\
   w(t_0) & =& 1.\end{aligned}

:math:`p_{max}` is a positive number when the system operates normally.
At an event such as fault incidence/removal, :math:`p_{max}` will change
to :math:`0` temporarily and back to the original value after the fault
is fixed. The objective is to maximize :math:`p_m` subject to the above
ODE constraints and :math:`\phi<\phi_S` during all times. To accommodate
the inequality constraint, we want to compute the sensitivity of the
cost function

.. math:: \Psi(p_m,\phi) = -p_m + c \int_{t_0}^{t_F} \left( \max(0, \phi - \phi_S ) \right)^2 dt

with respect to the parameter :math:`p_m`. :math:`numcost` is :math:`1`
since it is a scalar function.

For ODE solution, PETSc requires user-provided functions to evaluate the
system :math:`F(t,y,\dot{y},p)` (set by ``TSSetIFunction()`` ) and its
corresponding Jacobian :math:`F_y + \sigma F_{\dot y}` (set by
``TSSetIJacobian()``). Note that the solution state :math:`y` is
:math:`[ \phi \;  \omega ]^T` here. For sensitivity analysis, we need to
provide a routine to compute :math:`\mathrm{f}_p=[0 \; 1]^T` using
``TSASetRHSJacobianP()``, and three routines corresponding to the
integrand :math:`r=c \left( \max(0, \phi - \phi_S ) \right)^2`,
:math:`r_p = [0 \; 0]^T` and
:math:`r_y= [ 2 c \left( \max(0, \phi - \phi_S ) \right) \; 0]^T` using
``TSSetCostIntegrand()``.

In the adjoint run, :math:`\lambda` and :math:`\mu` are initialized as
:math:`[ 0 \;  0 ]^T` and :math:`[-1]` at the final time :math:`t_F`.
After ``TSAdjointSolve()``, the sensitivity of the cost function w.r.t.
initial conditions is given by the sensitivity variable :math:`\lambda`
(at time :math:`t_0`) directly. And the sensitivity of the cost function
w.r.t. the parameter :math:`p_m` can be computed (by users) as

.. math:: \frac{\mathrm{d} \Psi}{\mathrm{d} p_m} = \mu(t_0) + \lambda(t_0)  \frac{\mathrm{d} \left[ \phi(t_0) \; \omega(t_0) \right]^T}{\mathrm{d} p_m}  .

For explicit methods where one does not need to provide the Jacobian
:math:`F_u` for the forward solve one still does need it for the
backward solve and thus must call

::

   TSSetRHSJacobian(TS ts,Mat Amat, Mat Pmat,PetscErrorCode (*f)(TS,PetscReal,Vec,Mat,Mat,void*),void *fP);

Examples include:

-  a discrete adjoint sensitivity using explicit time stepping methods
   `TS Tutorial ex16adj <../../src/ts/tutorials/ex16adj.c.html>`__,

-  a discrete adjoint sensitivity using implicit time stepping methods
   `TS Tutorial ex20adj <../../src/ts/tutorials/ex20adj.c.html>`__,

-  an optimization using the discrete adjoint models of ERK
   `TS Tutorial ex16opt_ic <../../src/ts/tutorials/ex16opt_ic.c.html>`__
   and
   `TS Tutorial ex16opt_p` <../../src/ts/tutorials/ex16opt_p.c.html>`__,

-  an optimization using the discrete adjoint models of Theta methods
   for stiff DAEs
   `TS Tutorial ex20opt_ic <../../src/ts/tutorials/ex20opt_ic.c.html>`__
   and
   `TS Tutorial ex20opt_p <../../src/ts/tutorials/ex20opt_p.c.html>`__,

-  an ODE-constrained optimization using the discrete adjoint models of
   Theta methods for cost function with an integral term
   `TS Power Grid Tutorial ex3opt <../../src/ts/tutorials/power_grid/ex3opt.c.html>`__,

-  a discrete adjoint sensitivity using ``TSCN`` (Crank-Nicolson)
   methods for DAEs with discontinuities
   `TS Power Grid Stability Tutorial ex9busadj.c <../../src/ts/tutorials/power_grid/stability_9bus/ex9busadj.c.html>`__,

-  a DAE-constrained optimization using the discrete adjoint models of
   ``TSCN`` (Crank-Nicolson) methods for cost function with an integral
   term
   `TS Power Grid Tutorial ex9busopt.c <../../src/ts/tutorials/power_grid/stability_9bus/ex9busopt.c.html>`__,

-  a discrete adjoint sensitivity using ``TSCN`` methods for a PDE
   problem
   `TS Advection-Diffusion-Reaction Tutorial ex5adj <../../src/ts/tutorials/advection-diffusion-reaction/ex5adj.c.html>`__.

Checkpointing
~~~~~~~~~~~~~

The discrete adjoint model requires the states (and stage values in the
context of multistage timestepping methods) to evaluate the Jacobian
matrices during the adjoint (backward) run. By default, PETSc stores the
whole trajectory to disk as binary files, each of which contains the
information for a single time step including state, time, and stage
values (optional). One can also make PETSc store the trajectory to
memory with the option ``-ts_trajectory_type memory``. However, there
might not be sufficient memory capacity especially for large-scale
problems and long-time integration.

A so-called checkpointing scheme is needed to solve this problem. The
scheme stores checkpoints at selective time steps and recomputes the
missing information. The ``revolve`` library is used by PETSc
``TSTrajectory`` to generate an optimal checkpointing schedule that
minimizes the recomputations given a limited number of available
checkpoints. One can specify the number of available checkpoints with
the option
``-ts_trajectory_max_cps_ram [maximum number of checkpoints in RAM]``.
Note that one checkpoint corresponds to one time step.

The ``revolve`` library also provides an optimal multistage
checkpointing scheme that uses both RAM and disk for storage. This
scheme is automatically chosen if one uses both the option
``-ts_trajectory_max_cps_ram [maximum number of checkpoints in RAM]``
and the option
``-ts_trajectory_max_cps_disk [maximum number of checkpoints on disk]``.

Some other useful options are listed below.

-  ``-ts_trajectory_view`` prints the total number of recomputations,

-  ``-ts_monitor`` and ``-ts_adjoint_monitor`` allow users to monitor
   the progress of the adjoint work flow,

-  ``-ts_trajectory_type visualization`` may be used to save the whole
   trajectory for visualization. It stores the solution and the time,
   but no stage values. The binary files generated can be read into
   MATLAB via the script
   ``$PETSC_DIR/share/petsc/matlab/PetscReadBinaryTrajectory.m``.

Solving Steady-State Problems with Pseudo-Timestepping
------------------------------------------------------

**Simple Example:** ``TS`` provides a general code for performing pseudo
timestepping with a variable timestep at each physical node point. For
example, instead of directly attacking the steady-state problem

.. math:: G(u) = 0,

we can use pseudo-transient continuation by solving

.. math:: u_t = G(u).

Using time differencing

.. math:: u_t \doteq \frac{{u^{n+1}} - {u^{n}} }{dt^{n}}

with the backward Euler method, we obtain nonlinear equations at a
series of pseudo-timesteps

.. math:: \frac{1}{dt^n} B (u^{n+1} - u^{n} ) = G(u^{n+1}).

For this problem the user must provide :math:`G(u)`, the time steps
:math:`dt^{n}` and the left-hand-side matrix :math:`B` (or optionally,
if the timestep is position independent and :math:`B` is the identity
matrix, a scalar timestep), as well as optionally the Jacobian of
:math:`G(u)`.

More generally, this can be applied to implicit ODE and DAE for which
the transient form is

.. math:: F(u,\dot{u}) = 0.

For solving steady-state problems with pseudo-timestepping one proceeds
as follows.

-  Provide the function ``G(u)`` with the routine

   ::

       TSSetRHSFunction(TS ts,Vec r,PetscErrorCode (*f)(TS,PetscReal,Vec,Vec,void*),void *fP);

   The arguments to the function ``f()`` are the timestep context, the
   current time, the input for the function, the output for the function
   and the (optional) user-provided context variable ``fP``.

-  Provide the (approximate) Jacobian matrix of ``G(u)`` and a function
   to compute it at each Newton iteration. This is done with the command

   ::

      TSSetRHSJacobian(TS ts,Mat Amat, Mat Pmat,PetscErrorCode (*f)(TS,PetscReal,Vec,Mat,Mat,void*),void *fP);

   The arguments for the function ``f()`` are the timestep context, the
   current time, the location where the Jacobian is to be computed, the
   (approximate) Jacobian matrix, an alternative approximate Jacobian
   matrix used to construct the preconditioner, and the optional
   user-provided context, passed in as ``fP``. The user must provide the
   Jacobian as a matrix; thus, if using a matrix-free approach, one must
   create a ``MATSHELL`` matrix.

In addition, the user must provide a routine that computes the
pseudo-timestep. This is slightly different depending on if one is using
a constant timestep over the entire grid, or it varies with location.

-  For location-independent pseudo-timestepping, one uses the routine

   ::

      TSPseudoSetTimeStep(TS ts,PetscInt(*dt)(TS,PetscReal*,void*),void* dtctx);

   The function ``dt`` is a user-provided function that computes the
   next pseudo-timestep. As a default one can use
   ``TSPseudoTimeStepDefault(TS,PetscReal*,void*)`` for ``dt``. This
   routine updates the pseudo-timestep with one of two strategies: the
   default

   .. math:: dt^{n} = dt_{\mathrm{increment}}*dt^{n-1}*\frac{|| F(u^{n-1}) ||}{|| F(u^{n})||}

   or, the alternative,

   .. math:: dt^{n} = dt_{\mathrm{increment}}*dt^{0}*\frac{|| F(u^{0}) ||}{|| F(u^{n})||}

   which can be set with the call

   ::

      TSPseudoIncrementDtFromInitialDt(TS ts);

   or the option ``-ts_pseudo_increment_dt_from_initial_dt``. The value
   :math:`dt_{\mathrm{increment}}` is by default :math:`1.1`, but can be
   reset with the call

   ::

      TSPseudoSetTimeStepIncrement(TS ts,PetscReal inc);

   or the option ``-ts_pseudo_increment <inc>``.

-  For location-dependent pseudo-timestepping, the interface function
   has not yet been created.
