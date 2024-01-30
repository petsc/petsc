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

- Add ``PetscBench`` an object class for managing benchmarks in PETSc

.. rubric:: Event Logging:

.. rubric:: PetscViewer:

- Change ``PetscViewerRestoreSubViewer()`` to no longer need a call to ``PetscViewerFlush()`` after it
- Introduce ``PetscOptionsRestoreViewer()`` that must be called after ``PetscOptionsGetViewer()`` and ``PetscOptionsGetViewers()``
  to ensure thread safety

.. rubric:: PetscDraw:

.. rubric:: AO:

.. rubric:: IS:

.. rubric:: VecScatter / PetscSF:

- Add MPI-4.0 persistent neighborhood collectives support. Use -sf_neighbor_persistent along with -sf_type neighbor to enable it

.. rubric:: PF:

.. rubric:: Vec:

- Add ``VecGhostGetGhostIS()`` to get the ghost indices of a ghosted vector
- Add ``-vec_mdot_use_gemv`` to let ``VecMDot()``, ``VecMTDot()``  use BLAS2 ``gemv()`` instead of custom unrolled kernel. Default is on
- Add ``-vec_maxpy_use_gemv`` to let ``VecMAXPY()`` use BLAS2 ``gemv()`` instead of custom unrolled kernel. Default is off
- ``VecReplaceArray()`` on the first Vec obtained from ``VecDuplicateVecs()`` with either of the two above \*_use_gemv options won't work anymore. If needed, turn them off or use ``VecDuplicateVec()`` instead
- ``VecScale()`` is now a logically collective operation
- Add ``VecISShift()`` to shift a part of the vector
- ``VecISSet()`` does no longer accept NULL as index set

.. rubric:: PetscSection:

- Add ``PetscSectionGetBlockStarts()`` and ``PetscSectionSetBlockStarts()``

.. rubric:: PetscPartitioner:

.. rubric:: Mat:

- Reset ``MATLMVM`` history vectors if size is changed
- Add specific support for ``MatMultHermitianTranspose()`` and ``MatMultHermitianTransposeAdd()`` in ``MATSHELL``, ``MATDENSE``, ``MATNEST``, and ``MATSCALAPACK``
- Add function ``MatProductGetAlgorithm()``
- ``MATTRANSPOSEVIRTUAL``, ``MATHERMITIANTRANSPOSEVIRTUAL``, ``MATNORMAL``, ``MATNORMALHERMITIAN``, and ``MATCOMPOSITE`` now derive from ``MATSHELL``. This implies a new behavior for those ``Mat``, as calling ``MatAssemblyBegin()``/``MatAssemblyEnd()`` destroys scalings and shifts for ``MATSHELL``, but it was not previously the case for other ``MatType``

.. rubric:: MatCoarsen:

- Add ``MatCoarsenSetMaximumIterations()`` with corresponding option ``-mat_coarsen_max_it <4>``. The number of iteration of the coarsening method. Used for the HEM coarsener
- Add ``MatCoarsenSetThreshold()`` with corresponding option ``-mat_coarsen_threshold <-1>``. Threshold for filtering graph for HEM. Like GAMG < 0 means no filtering
- Change API for several PetscCD methods used internally in ``PCGAMG`` and ``MatCoarsen`` (eg, change ``PetscCDSetChuckSize()`` to ``PetscCDSetChunckSize()``), remove ``Mat`` argument from``PetscCDGetASMBlocks()``

.. rubric:: PC:

- Add ``PCGAMGSetLowMemoryFilter()`` with corresponding option ``-pc_gamg_low_memory_threshold_filter``. Use the system ``MatFilter`` graph/matrix filter, without a temporary copy of the graph, otherwise use method that can be faster
- Add ``PCGAMGASMSetHEM()`` with corresponding option ``-pc_gamg_asm_hem_aggs N``. Use ASM smoother constructed from N applications of heavy edge matching
- ``PCMAT`` use ``MatSolve()`` if implemented by the matrix type
- Add ``PCLMVMSetUpdateVec()`` for the automatic update of the LMVM preconditioner inside a SNES solve
- Add ``PCGAMGSetInjectionIndex()`` with corresponding option ``-pc_gamg_injection_index i,j,k...``. Inject provided indices of fine grid operator as first coarse grid restriction (sort of p-multigrid for C1 elements)

.. rubric:: KSP:

.. rubric:: SNES:

- Add support for Quasi-Newton models in ``SNESNEWTONTR`` via ``SNESNewtonTRSetQNType``
- Add support for trust region norm customization in ``SNESNEWTONTR`` via ``SNESNewtonTRSetNormType``
- Remove default of ``KSPPREONLY`` and ``PCLU`` for ``SNESNASM`` subdomain solves: for ``SNESASPIN`` use ``-npc_sub_ksp_type preonly -npc_sub_pc_type lu``

.. rubric:: SNESLineSearch:

.. rubric:: TS:

- Add support for custom predictor callbacks in the second-order generalized-alpha method using ``TSAlpha2SetPredictor()``
- Allow adaptivity to change time step size in first step of second-order generalized-alpha method.
- Add ``TSSetPostEventStep()`` to control the first step after event
- Rename ``TSSetPostEventIntervalStep()`` to ``TSSetPostEventSecondStep()``, controlling the second step after event
- Rename option ``-ts_event_post_eventinterval_step`` to ``-ts_event_post_event_second_step``
- Change the (event) indicator functions type from ``PetscScalar[]`` to ``PetscReal[]`` in the user ``indicator()`` callback set by ``TSSetEventHandler()``

.. rubric:: TAO:

- Deprecate ``TaoCancelMonitors()`` (resp. ``-tao_cancelmonitors``) in favor of ``TaoMonitorCancel()`` (resp. ``-tao_monitor_cancel``)

.. rubric:: DM/DA:

- Add MPI reduction inside ``SNESComputeObjective_DMDA()``. No need to call reduction into local callback

.. rubric:: DMSwarm:

.. rubric:: DMPlex:

- Drop support for MED, i.e. remove ``DMPlexCreateMedFromFile()`` and ``--with-med``
- Change protototype of ``DMPlexSetSNESLocalFEM()``. Now it accepts a single context and a Boolean indicating to use the objective function callback
- Replace ``DMProjectCoordinates()`` with ``DMSetCoordinateDisc()``
- Add argument to ``DMPlexCreateCoordinateSpace()``
- Add ``DMPlexReorderSectionGetDefault()`` and ``DMPlexReorderSectionSetDefault()`` to allow point permutations when sections are built automatically
- Add ``DMPlexCoordMap`` and some default maps
- Add Boolean argument to ``DMPlexPartitionLabelCreateSF()`` to sort ranks
- Add ``DMClearAuxiliaryVec()`` to clear the auxiliary data

.. rubric:: FE/FV:

- Add Jacobian type argument to ``PetscFEIntegrateBdJacobian()``

.. rubric:: DMNetwork:

.. rubric:: DMStag:

- Add support for ``DMLocalToLocalBegin()`` and ``DMLocalToLocalEnd()``

.. rubric:: DT:

- Add ``PetscDSUpdateBoundaryLabels()``

.. rubric:: Fortran:
