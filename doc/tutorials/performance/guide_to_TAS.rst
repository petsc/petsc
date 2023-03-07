====================================
Time, Accuracy, Speed Analysis (TAS)
====================================
.. highlight:: none

Below is the guide to running TAS using ex13, which is a Poisson Problem in 2D and 3D with Finite Elements:

1. Compile ``ex13.c``

  a. This file is located in ``PETSC_DIR/src/snes/tutorials/``
  b. If you do not know how to build a PETSc code here is an example makefile

      .. literalinclude:: makefile

2. Run ex13 with the following command:

   .. code-block::

      mpiexec -n 2 ./ex13 -log_view :/home/<user name>/PETSC_DIR/lib/petsc/bin/ex_13_test.py:ascii_info_detail \
        -dm_distribute \
        -dm_plex_box_faces 8,8 \
        -potential_petscspace_degree 1 \
        -snes_convergence_estimate \
        -convest_num_refine 5
3. A log file in the above directory called ``ex_13_test.py`` should now be present.  This is also the same directory that contains the TAS python3 script ``petsc_tas_analysis.py``
4. Now run ``petsc_tas_analysis.py``:

   .. code-block::

      ./petsc_tas_analysis.py -f ex_13_test
5. You should see something similar to the following in your terminal window:

    .. literalinclude:: exampleTASOutPut.txt

6. Finally the graphs will appear in the subdirectory ``graphs/``

For more detailed help in using TAS:
 1. See detailed user's guide
 2. On the command line use ``./petsc_tas_analysis.py -h``
