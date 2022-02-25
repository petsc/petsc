==============
PETSc Hands On
==============

PETSc comes with a large number of example codes to illustrate usage. Here, we highlight a few, key ones:

-  :ref:`Linear Poisson equation on a 2D grid <handson_example_1>`

   -  example of linear equation problem
   -  see also `src/ksp/ksp/tutorials <../../src/ksp/ksp/tutorials/index.html>`__

-  :ref:`Nonlinear ODE arising from a time-dependent one dimensional PDE <handson_example_2>`

   -  example of time-stepping problem
   -  see also `src/ts/tutorials <../../src/ts/tutorials/index.html>`__

-  :ref:`Nonlinear PDE on a structured grid <handson_example_3>`

   -  example of nonlinear PDE
   -  see also `src/snes/tutorials <../../src/snes/tutorials/index.html>`__

-  :ref:`Linear Stokes-type PDE on a structured grid <handson_example_4>`
-  :ref:`Nonlinear time dependent PDE on Unstructured Grid <handson_example_5>`

Several examples are also included that represent the
interoperability with other numerical software packages in the `xSDK
Toolkit <http://www.xsdk.info>`__. These packages can be
automatically installed by PETSc by :ref:`configuring <tut_install_config>` with
``--download-trilinos``,  ``--download-hypre``, and/or
``--download-superlu_dist``.

-  :ref:`Nonlinear PDE Example <handson_example_3>` using `Trilinos preconditioner ML <http://trilinos.org/packages/ml>`__
-  :ref:`Nonlinear PDE Example <handson_example_3>` using BoomerAMG from `HYPRE <https://computation.llnl.gov/projects/hypre-scalable-linear-solvers-multigrid-methods>`__
-  :ref:`Linear Equation Example <handson_example_1>` using direct solver `SuperLU_DIST <https://crd-legacy.lbl.gov/~xiaoye/SuperLU/>`__

.. _handson_example_1:

Example 1: Linear Poisson equation on a 2D grid
-----------------------------------------------

WHAT THIS EXAMPLE DEMONSTRATES:

-  Using command line options
-  Using Linear Solvers
-  Handling a simple structured grid

FURTHER DETAILS:

-  `Mathematical description of the problem <../../src/ksp/ksp/tutorials/ex50.c.html#line1>`__
-  `the source code <../../src/ksp/ksp/tutorials/ex50.c.html#line21>`__

DO THE FOLLOWING:

-  Compile ``src/ksp/ksp/tutorials/ex50.c``

   .. code-block:: console

       $ cd petsc/src/ksp/ksp/tutorials
       $ make ex50



-  Run a 1 processor example with a 3x3 mesh and view the matrix
   assembled

   .. code-block:: console

           $ mpiexec -n 1 ./ex50  -da_grid_x 4 -da_grid_y 4 -mat_view

   Expected output:

   .. literalinclude:: /../src/ksp/ksp/tutorials/output/ex50_tut_1.out
    :language: none


-  Run with a 120x120 mesh on 4 processors using superlu_dist and
   view the solver options used

   .. code-block:: console

           $ mpiexec -n 4 ./ex50  -da_grid_x 120 -da_grid_y 120 -pc_type lu -pc_factor_mat_solver_type superlu_dist -ksp_monitor -ksp_view

   Expected output:

   .. literalinclude:: /../src/ksp/ksp/tutorials/output/ex50_tut_2.out
    :language: none


-  Run with a 1025x1025 grid using multigrid solver on 4
   processors with 9 multigrid levels

   .. code-block:: console

           $ mpiexec -n 4 ./ex50 -da_grid_x 1025 -da_grid_y 1025 -pc_type mg -pc_mg_levels 9 -ksp_monitor

   Expected output:

   .. literalinclude:: /../src/ksp/ksp/tutorials/output/ex50_tut_3.out
    :language: none


.. _handson_example_2:

Example 2: Nonlinear ODE arising from a time-dependent one dimensional PDE
--------------------------------------------------------------------------

WHAT THIS EXAMPLE DEMONSTRATES:

-  Using command line options
-  Handling a simple structured grid
-  Using the ODE integrator
-  Using call-back functions

FURTHER DETAILS:

-  `Mathematical description of the problem <../../src/ts/tutorials/ex2.c.html#line13>`__
-  `the source
   code <../../src/ts/tutorials/ex2.c.html#line36>`__

DO THE FOLLOWING:

-  Compile ``src/ts/tutorials/ex2.c``

   .. code-block:: console

            $ cd petsc/src/ts/tutorials
            $ make ex2


-  Run a 1 processor example on the default grid with all the
   default solver options

   .. code-block:: console

           $ mpiexec -n 1 ./ex2 -ts_max_steps 10 -ts_monitor

   Expected output:

   .. literalinclude:: /../src/ts/tutorials/output/ex2_tut_1.out
    :language: none


-  Run with the same options on 4 processors plus monitor
   convergence of the nonlinear and linear solvers

   .. code-block:: console

           $ mpiexec -n 4 ./ex2 -ts_max_steps 10 -ts_monitor -snes_monitor -ksp_monitor

   Expected output:

   .. literalinclude:: /../src/ts/tutorials/output/ex2_tut_2.out
    :language: none


-  Run with the same options on 4 processors with 128 grid points

   .. code-block:: console

           $ mpiexec -n 16 ./ex2 -ts_max_steps 10 -ts_monitor -M 128

   Expected output:

   .. literalinclude:: /../src/ts/tutorials/output/ex2_tut_3.out
    :language: none


.. _handson_example_3:

Example 3: Nonlinear PDE on a structured grid
---------------------------------------------

WHAT THIS EXAMPLE DEMONSTRATES:

-  Handling a 2d structured grid
-  Using the nonlinear solvers
-  Changing the default linear solver

FURTHER DETAILS:

-  `Mathematical description of the problem <../../src/snes/tutorials/ex19.c.html#line19>`__
-  `main program source
   code <../../src/snes/tutorials/ex19.c.html#line94>`__
-  `physics source
   code <../../src/snes/tutorials/ex19.c.html#line246>`__

DO THE FOLLOWING:

-  Compile ``src/snes/tutorials/ex19.c``

   .. code-block:: console

            $ cd petsc/src/snes/tutorials/
            $ make ex19


-  Run a 4 processor example with 5 levels of grid refinement,
   monitor the convergence of the nonlinear and linear solver and
   examine the exact solver used

   .. code-block:: console

           $ mpiexec -n 4 ./ex19 -da_refine 5 -snes_monitor -ksp_monitor -snes_view

   Expected output:

   .. literalinclude:: /../src/snes/tutorials/output/ex19_tut_1.out
    :language: none


-  Run with the same options but use geometric multigrid as the
   linear solver

   .. code-block:: console

           $ mpiexec -n 4 ./ex19 -da_refine 5 -snes_monitor -ksp_monitor -snes_view -pc_type mg

   Expected output:

   .. literalinclude:: /../src/snes/tutorials/output/ex19_tut_2.out
    :language: none


   Note this requires many fewer iterations than the default
   solver

-  Run with the same options but use algebraic multigrid (hypre's
   BoomerAMG) as the linear solver

   .. code-block:: console

           $ mpiexec -n 4 ./ex19 -da_refine 5 -snes_monitor -ksp_monitor -snes_view -pc_type hypre

   Expected output:

   .. literalinclude:: /../src/snes/tutorials/output/ex19_tut_3.out
    :language: none


   Note this requires many fewer iterations than the default
   solver but requires more linear solver iterations than
   geometric multigrid.

-  Run with the same options but use the ML preconditioner from
   Trilinos

   .. code-block:: console

           $ mpiexec -n 4 ./ex19 -da_refine 5 -snes_monitor -ksp_monitor -snes_view -pc_type ml

   Expected output:

   .. literalinclude:: /../src/snes/tutorials/output/ex19_tut_8.out
    :language: none


-  Run on 1 processor with the default linear solver and profile
   the run

   .. code-block:: console

           $ mpiexec -n 1 ./ex19 -da_refine 5 -log_view

   Expected output:

   .. literalinclude:: /../src/snes/tutorials/output/ex19_tut_4.out
    :language: none



   Search for the line beginning with SNESSolve, the fourth column
   gives the time for the nonlinear solve.

-  Run on 1 processor with the geometric multigrid linear solver
   and profile the run

   .. code-block:: console

           $ mpiexec -n 1 ./ex19 -da_refine 5 -log_view -pc_type mg

   Expected output:

   .. literalinclude:: /../src/snes/tutorials/output/ex19_tut_5.out
    :language: none



   Compare the runtime for SNESSolve to the case with the default
   solver

-  Run on 4 processors with the default linear solver and profile
   the run

   .. code-block:: console

           $ mpiexec -n 4 ./ex19 -da_refine 5 -log_view

   Expected output:

   .. literalinclude:: /../src/snes/tutorials/output/ex19_tut_6.out
    :language: none


   Compare the runtime for ``SNESSolve`` to the 1 processor case with
   the default solver. What is the speedup?

-  Run on 4 processors with the geometric multigrid linear solver
   and profile the run

   .. code-block:: console

           $ mpiexec -n 4 ./ex19 -da_refine 5 -log_view -pc_type mg

   Expected output:

   .. literalinclude:: /../src/snes/tutorials/output/ex19_tut_7.out
    :language: none


   Compare the runtime for SNESSolve to the 1 processor case with
   multigrid. What is the speedup? Why is the speedup for
   multigrid lower than the speedup for the default solver?

.. _handson_example_4:

Example 4: Linear Stokes-type PDE on a structured grid
------------------------------------------------------

WHAT THIS EXAMPLE DEMONSTRATES:

-  Handling a 3d structured grid
-  Controlling linear solver options
-  Selecting composible preconditioners
-  Solving a Stokes problem
-  Adding your own problem specific visualization

FURTHER DETAILS:

-  `Mathematical description of the problem <../../src/ksp/ksp/tutorials/ex42.c.html>`__
-  `main program source code <../../src/ksp/ksp/tutorials/ex42.c.html#line2059>`__
-  `physics source code <../../src/ksp/ksp/tutorials/ex42.c.html#line819>`__

DO THE FOLLOWING:

-  Compile ``src/ksp/ksp/tutorials/ex42.c``

   .. code-block:: console

            $ cd petsc/src/ksp/ksp/tutorials
            $ make ex42


-  Solve with the default solver

   .. code-block:: console

           $ mpiexec -n 4 ./ex42  -stokes_ksp_monitor

   Expected output:

   .. literalinclude:: /../src/ksp/ksp/tutorials/output/ex42_tut_1.out
     :language: none


   Note the poor convergence for even a very small problem

-  Solve with a solver appropriate for Stoke's problems
   ``-stokes_pc_type fieldsplit -stokes_pc_fieldsplit_type schur``

   .. code-block:: console

           $ mpiexec -n 4 ./ex42  -stokes_ksp_monitor -stokes_pc_type fieldsplit -stokes_pc_fieldsplit_type schur

   Expected output:

   .. literalinclude:: /../src/ksp/ksp/tutorials/output/ex42_tut_2.out
    :language: none


-  Solve with a finer mesh

   .. code-block:: console

           $ mpiexec -n 4 ./ex42  -mx 20 -stokes_ksp_monitor  -stokes_pc_type fieldsplit -stokes_pc_fieldsplit_type schur

   Expected output:

   .. literalinclude:: /../src/ksp/ksp/tutorials/output/ex42_tut_3.out
    :language: none


   Repeat with

   ::

       -mx 40

   and/or more MPI ranks.

.. _handson_example_5:

Example 5: Nonlinear time dependent PDE on Unstructured Grid
------------------------------------------------------------

WHAT THIS EXAMPLE DEMONSTRATES:

-  Changing the default ODE integrator
-  Handling unstructured grids
-  Registering your own interchangeable physics and algorithm
   modules

FURTHER DETAILS:

-  `Mathematical description of the problem <../../src/ts/tutorials/ex11.c.html>`__
-  `main program source code <../../src/ts/tutorials/ex11.c.html#line1403>`__
-  `source code of physics modules <../../src/ts/tutorials/ex11.c.html#line186>`__

DO THE FOLLOWING:

-  Compile ``src/ts/tutorials/ex11.c``

   .. code-block:: console

            $ cd petsc/src/ts/tutorials
            $ make ex11


-  Run simple advection through a tiny hybrid mesh

   .. code-block:: console

           $ mpiexec -n 1 ./ex11 -f ${PETSC_DIR}/share/petsc/datafiles/meshes/sevenside.exo

   Expected output:

   .. literalinclude:: /../src/ts/tutorials/output/ex11_tut_1.out
    :language: none


-  Run simple advection through a small mesh with a Rosenbrock-W
   solver

   .. code-block:: console

           $ mpiexec -n 1 ./ex11 -f ${PETSC_DIR}/share/petsc/datafiles/meshes/sevenside.exo -ts_type rosw

   Expected output:

   .. literalinclude:: /../src/ts/tutorials/output/ex11_tut_2.out
    :language: none


-  Run simple advection through a larger quadrilateral mesh of an
   annulus with least squares reconstruction and no limiting,
   monitoring the error

   .. code-block:: console

           $ mpiexec -n 4 ./ex11 -f ${PETSC_DIR}/share/petsc/datafiles/meshes/annulus-20.exo -monitor Error -advect_sol_type bump -petscfv_type leastsquares -petsclimiter_type sin

   Expected output:

   .. literalinclude:: /../src/ts/tutorials/output/ex11_tut_3.out
    :language: none


   Compare turning to the error after turning off reconstruction.

-  Run shallow water on the larger mesh with least squares
   reconstruction and minmod limiting, monitoring water Height
   (integral is conserved) and Energy (not conserved)

   .. code-block:: console

           $ mpiexec -n 4 ./ex11 -f ${PETSC_DIR}/share/petsc/datafiles/meshes/annulus-20.exo -physics sw -monitor Height,Energy -petscfv_type leastsquares -petsclimiter_type minmod

   Expected output:

   .. literalinclude:: /../src/ts/tutorials/output/ex11_tut_4.out
    :language: none
