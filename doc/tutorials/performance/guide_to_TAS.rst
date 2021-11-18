==========================================================================
Guide to PETSc Quick Start Tutorial on Time, Accuracy, Speed Analysis(TAS)
==========================================================================
.. highlight:: none

Below is the guide to running TAS using ex13, which is a Poisson Problem in 2D and 3D with Finite Elements:

1. Compile ``ex13.c``

  a. This example source file, and the corresponding ``makefile`` are located in ``PETSC_DIR/src/snes/tutorials/``
  b. Compile with the command:

      .. code-block:: console

         $ make ex13

2. Run ex13 with the following command:

   .. code-block:: console

      $ mpiexec -n 2 ./ex13 -log_view :/home/<user name>/PETSC_DIR/lib/petsc/bin/ex_13_test.py:ascii_info_detail \
        -dm_distribute \
        -dm_plex_box_faces 8,8 \
        -potential_petscspace_degree 1 \
        -snes_convergence_estimate \
        -convest_num_refine 5

3. A log file in the above directory called ``ex_13_test.py`` should now be present.  This is also the same directory that contains the TAS python3 script ``petsc_tas_analysis.py``

4. Now run ``petsc_tas_analysis.py``:

   .. code-block:: console

      $ ./petsc_tas_analysis.py -f ex_13_test

5. You should see something similar to the following in your terminal window:

   ::

      ex_13Test
          *******************Data for ex_13Test***************************
                    Times : [0.007 0.019 0.045 0.136 0.49 ]

                Mean Time : [0.007 0.019 0.045 0.136 0.49 ]

              Times Range : [3.87e-06 4.20e-06 4.40e-06 4.00e-06 3.00e-06]

         Time Growth Rate : [2.84  2.361 3.029 3.591]

                    Flops : [1.834e+04 1.443e+05 8.816e+05 7.164e+06 6.256e+07]

               Mean Flops : [9.168e+03 7.215e+04 4.408e+05 3.582e+06 3.128e+07]

               Flop Range : [9.691e+03 7.644e+04 4.595e+05 3.677e+06 3.174e+07]

         Flop Growth Rate : [7.87  6.109 8.126 8.733]

                LU Factor : [7.157e-06 2.878e-05 7.875e-05 2.821e-04 1.050e-03]

           LU Factor Mean : [6.480e-06 2.694e-05 7.535e-05 2.702e-04 1.048e-03]

          LU Factor Range : [1.354e-06 3.668e-06 6.786e-06 2.368e-05 4.970e-06]

       LU Factor Growth Rate : []

      **********Data for Field 0************
                     dofs : [  12.   56.  240.  992. 4032.]

                   Errors : [0.203 0.053 0.013 0.003 0.001]

      Least Squares Data
      ==================
      Mesh Convergence
      Alpha: -0.9460821998182669
        0.3575225062079136
      convRate: 1.8921643996365338 of ex_13Test data

6. Finally the graphs will appear in the subdirectory ``graphs/``

For more detailed help in using TAS:
 1. See detailed user's guide
 2. On the command line use ``./petsc_tas_analysis.py -h``
