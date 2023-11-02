====================
Changes: Development
====================

..
   STYLE GUIDELINES:
   * Capitalize sentences
   * Use imperative, e.g., Add, Improve, Change, etc.
   * Don't use a period (.) at the end of entries
   * If multiple sentences are needed, use a period or semicolon to divide sentences, but not at the end of the final sentence

.. rubric:: General:

.. rubric:: Configure/Build:

- Add ``--download-blis-use-openmp=0`` to force ``download-blis`` to not build with OpenMP when ``with-openmp`` is provided
- Add ```PetscBLASSetNumThreads()`` and ``PetscBLASGetNumThreads()`` for controlling how many threads the BLAS routines use

.. rubric:: Sys:

.. rubric:: Event Logging:

.. rubric:: PetscViewer:

.. rubric:: PetscDraw:

.. rubric:: AO:

.. rubric:: IS:

.. rubric:: VecScatter / PetscSF:

- Add MPI-4.0 persistent neighborhood collectives support. Use -sf_neighbor_persistent along with -sf_type neighbor to enable it

.. rubric:: PF:

.. rubric:: Vec:

- Add ``VecGhostGetGhostIS()`` to get the ghost indices of a ghosted vector

.. rubric:: PetscSection:

.. rubric:: PetscPartitioner:

.. rubric:: Mat:

.. rubric:: MatCoarsen:

- Add ``MatCoarsenSetMaximumIterations()`` with corresponding option ``-mat_coarsen_max_it <4>``. The number of iteration of the coarsening method. Used for the HEM coarsener
- Add ``MatCoarsenSetThreshold()`` with corresponding option ``-mat_coarsen_threshold <-1>``. Threshold for filtering graph for HEM. Like GAMG < 0 means no filtering
- Change API for several PetscCD methods used internally in GAMG and MatCoarsen (eg, change ``PetscCDSetChuckSize()`` to ``PetscCDSetChunckSize()``), remove Mat argument from``PetscCDGetASMBlocks()``

.. rubric:: PC:

- Add ``PCGAMGSetLowMemoryFilter()`` with corresponding option ``-pc_gamg_low_memory_threshold_filter``. Use the system ``MatFilter`` graph/matrix filter, without a temporary copy of the graph, otherwise use method that can be faster

.. rubric:: KSP:

.. rubric:: SNES:

.. rubric:: SNESLineSearch:

.. rubric:: TS:

- Add support for custom predictor callbacks in the second-order generalized-alpha method using ``TSAlpha2SetPredictor()``
- Allow adaptivity to change time step size in first step of second-order generalized-alpha method.

.. rubric:: TAO:

.. rubric:: DM/DA:

.. rubric:: DMSwarm:

.. rubric:: DMPlex:

- Replace ``DMProjectCoordinates()`` with ``DMSetCoordinateDisc()``
- Add argument to ``DMPlexCreateCoordinateSpace()``

.. rubric:: FE/FV:

.. rubric:: DMNetwork:

.. rubric:: DMStag:

.. rubric:: DT:

.. rubric:: Fortran:
