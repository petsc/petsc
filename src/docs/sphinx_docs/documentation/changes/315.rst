====================
Changes: Development
====================

   .. rubric:: General:

   -  Add ``PetscKokkosInitializeCheck()``, which initializes Kokkos if it
      is not yet initialized
   -  Add support for ``-debug_terminal`` Terminal to use Apple's Terminal
      instead of xterm, allowing use of cut-paste
   -  Make Terminal the default device to display the debugger on Apple
      instead of xterm
   -  Add PetscHasExternalPackage() determining whether PETSc has been
      configured with the given external package such as "hdf5"

   .. rubric:: Configure/Build:

   -  On macOS, ``MACOS_FIREWALL=1`` or ``MACOS_FIREWALL_REFRESH=1`` can
      be passed to make to automatically add firewall rules preventing
      firewall popups during testing. See
      ``make -f gmakefile.test help`` for details
   -  ``./configure --with-macos-firewall-rules`` makes
      ``MACOS_FIREWALL=1`` the default
   -  Change ``--download-petsc4py`` to ``--with-petsc4py`` to have PETSc build
      and use petsc4py
   -  Add ``--download-mmg`` and ``--download-parmmg``, 3D unstructured mesh
      adaptation package (interaction with DMPlex not available yet)
   -  Improve detection of Git repositories when a ``--download-package``
      option is used

      -  Support ``ssh://*.git`` and ``https://*.git`` URLs without the
         additional ``git://`` prefix
      -  Local directories can be specified without the ``git://``
         prefix as well
      -  Any valid Git repository (including bare and with
         ``--separate-git-dir``) is now correctly detected

   -  ``--download-yaml`` or ``--with-yaml`` are no longer required for
      YAML support (but can still be used to avoid compiling source
      included with PETSc)

   .. rubric:: Sys:

   -  Add ``PETSCRANDOMCURAND`` to support CURAND random number generator
   -  Add ``PetscRandomGetValues()`` and ``PetscRandomGetValuesReal()`` to retrieve
      an array of random numbers
   -  Add ``PetscOptions`` argument to ``PetscOptionsInsertFileYAML()``
   -  Add ``PetscCalloc()`` to allocate zeroed memory
   -  Automatically detect YAML and JSON option files by extension or
      particular first line
   -  Update YAML options file processing to ignore keys starting with
      ``$``, Add some special processing
   -  Add ``PetscBagViewFromOptions()``
   -  Add ``PetscLogEventDeactivatePush()``, ``PetscLogEventDeactivatePop()``
   -  Add new option to ``-log_view`` to view nested event timing
      information as a flame graph

   .. rubric:: PetscViewer:

   -  ``PetscViewerAndFormat`` now allows a payload
   -  Change ``PetscViewerFlowControlStepMaster()``,
      ``PetscViewerFlowControlEndMaster()`` to
      ``PetscViewerFlowControlStepMain()``, ``PetscViewerFlowControlEndMain()``
   - HDF5: ``FILE_MODE_APPEND`` (= ``FILE_MODE_UPDATE``) now creates a new file if it does not exist yet
   - VU: ``PetscViewerVUSetMode()`` is now deprecated;
     please use standard ``PetscViewerFileSetMode()`` instead

   .. rubric:: PetscDraw:

   .. rubric:: AO:

   .. rubric:: IS:

   .. rubric:: VecScatter / PetscSF:

   -  ``VecScatter`` is now the same type as ``PetscSF``, in other words, we
      have ``typedef PetscSF VecScatter``
   -  Remove ``VecScatter`` types ``VECSCATTER{SEQ,MPI1,MPI3,MPI3NODE,SF}``. One
      can use all ``PetcSF`` types as ``VecScatter`` types
   -  Rename ``PetscLayoutsCreateSF()`` to ``PetscSFCreateFromLayouts()`` and
      move its declaration from ``petscis.h`` to ``petscsf.h``
   -  Deprecate ``MPIU_REPLACE``; Use ``MPI_REPLACE`` instead
   -  Deprecate ``PetscSFBcastAndOp`` variants; Use ``PetscSFBcast`` instead
   -  Deprecate ``PetscSFCreateEmbeddedSF``; Use ``PetscSFCreateEmbeddedRootSF``
      instead
   -  Add experimental NVIDIA NVSHMEM support; For details on how to use
      it, contact petsc-maint@mcs.anl.gov
   -  Add ``PetscSFCreateByMatchingIndices()`` to create SF by matching root
      and leaf indices

   .. rubric:: PF:

   .. rubric:: Vec:

   -  Change ``Vec{Get,Restore}Array{Read}Inplace`` to
      ``Vec{Get,Restore}Array{Read}AndMemType()`` and add an extra argument
      to also return the memory type of the array
   -  Remove vector type ``VECNODE``
   -  Add ``VecConcatenate()`` function for vertically concatenating an
      array of vectors into a single vector. Also returns an array of
      index sets to access the original components within the
      concatenated final vector

   .. rubric:: PetscSection:

   .. rubric:: PetscPartitioner:

   .. rubric:: Mat:

   -  Add ``MatSetPreallocationCOO`` and ``MatSetValuesCOO`` to preallocate and
      set values in a matrix using COO format. Currently efficiently
      implemented only for ``MATCUSPARSE``
   -  Add the option ``MAT_FORCE_DIAGONAL_ENTRIES`` for ``MatSetOption()``. It
      forces allocation of all diagonal entries
   -  Remove ``MAT_NEW_DIAGONALS`` from ``MatOption``
   -  Add ``UNKNOW_NONZERO_PATTERN`` as new value for ``MatStructure``. It
      indicates that the relationship is unknown, when set the AIJ
      matrices check if the two matrices have identical patterns and if
      so use the faster code
   -  Add ``MAT_FACTOR_QR``, ``MatQRFactor()``, ``MatQRFactorSymbolic()``, and
      ``MatQRFactorNumeric()`` for QR factorizations. Currently the only
      built-in implementation uses LAPACK on sequential dense matrices
   - Change option ``-mat_cusparse_transgen`` to ``-mat_form_explicit_transpose`` to hint PETSc to form an explicit transpose for repeated operations like MatMultTranspose. Currently implemented only for ``AIJCUSPARSE`` and ``AIJKOKKOS``
   - Add a ``MatOption`` ``MAT_FORM_EXPLICIT_TRANSPOSE``

   .. rubric:: PC:

   -  Add ``PCGAMGSetRankReductionFactors()``, provide an array,
      ``-pc_gamg_rank_reduction_factors factors``, tp specify factor by
      which to reduce active processors on coarse grids in ``PCGAMG`` that
      overrides default heuristics
   -  Change ``PCCompositeAddPC()`` to ``PCCompositeAddPCType()``, now
      ``PCCompositeAddPC()`` adds a specific ``PC`` object
   -  Add a Compatible Relaxation (CR) viewer ``PCMG`` with ``-pc_mg_adapt_cr``
   -  Experimental: Add support for assembling AIJ (CUSPARSE and KOKKOS)
      matrix on the Cuda device with ``MatSetValuesDevice()``,
      ``MatCUSPARSEGetDeviceMatWrite()``, and Kokkos with
      ``MatKokkosGetDeviceMatWrite``
   -  Add ``PCMGSetResidualTranspose()`` to support transposed linear solve
      using ``PCMG`` and ``PCGAMG``

   .. rubric:: KSP:

   -  Add ``-all_ksp_monitor`` which turns on monitoring for all KSP
      solvers regardless of their prefix. This is useful for monitoring
      solvers with inner solvers such as ``PCMG``, ``PCGAMG``, ``PCFIELDSPLIT``.
   -  Add support for monitor ``KSPPREONLY``. This is useful for monitoring
      solvers with inner solvers such as ``PCMG``, ``PCGAMG``, ``PCFIELDSPLIT``.
   -  Add ``KSPConvergedReasonViewSet()`` to set an ADDITIONAL function that
      is to be used at the end of the linear solver to display the
      convergence reason of the linear solver
   -  Add ``KSPConvergedReasonViewCancel()`` to remove all user-added
      converged reason view functions
   -  Add ``KSPGetConvergedReasonString()`` to retrieve a human readable
      string for ksp converged reason
   -  Change ``KSPReasonView()`` to ``KSPConvergenceReasonView()``
   -  Change ``KSPReasonViewFromOptions()`` to
      ``KSPConvergedReasonViewFromOptions()``
   -  Add ``KSPConvergedDefaultSetConvergedMaxits()`` to declare convergence
      when the maximum number of iterations is reached
   -  Fix many ``KSP`` implementations to actually perform the number of
      iterations requested
   -  Chebyshev uses ``MAT_SPD`` to default to CG for the eigen estimate
   -  Add ``KSPPIPECG2``, a pipelined solver that reduces the number of
      allreduces to one per two iterations and overlaps it with two PCs
      and SPMVs using non-blocking allreduce
   -  Add ``KSPConvergedRateView()`` and ``KSPComputeConvergenceRate()`` to
      check the convergence rate of a linear solve
   -  Add ``KSPSetUseExplicitTranspose()`` to explicitly transpose the
      system in ``KSPSolveTranspose()``
   -  Add ``KSPMonitorLGCreate()``, and remove ``KSPMonitorLGResidualNorm*()``
      and ``KSPMonitorLGTrueResidualNorm*()``
   -  Add ``KSPMonitorError()``, used by ``-ksp_monitor_error``
   -  Add arguments to ``KSPMonitorSetFromOptions()`` to allow line graphs
      to be configured
   -  Deprecate ``KSP{Set|Get}MatSolveBlockSize()``, use
      ``KSP{Set|Get}MatSolveBatchSize()`` instead
   -  Reduce default ``KSPView()`` ASCII output to a single subdomain's
      KSP/PC information for ``PCASM``, resp. ``PCBJacobi``. Use
      ``-ksp_view ::ascii_info_detail`` to output KSP/PC information for all
      subdomains

   .. rubric:: SNES:

   -  Add ``SNESConvergedCorrectPressure()``, which can be selected using
      ``-snes_convergence_test correct_pressure``
   -  Remove ``SNESMonitorLGCreate()`` and ``SNESMonitorLGResidualNorm()`` which
      are now handled by the default monitor
   -  Add ``SNESConvergedReasonViewSet()`` to set an ADDITIONAL function
      that is to be used at the end of the nonlinear solver to display
      the convergence reason of the nonlinear solver
   -  Add ``SNESConvergedReasonViewCancel()`` to remove all user-added
      converged reason view functions
   -  Add ``SNESGetConvergedReasonString()`` to retrieve a human readable
      string for snes converged reason
   -  Add ``SNESFASFullSetTotal()`` to use total residual restriction and
      total solution interpolation in the initial cycle of full FAS
      multigrid
   -  Deprecate ``-snes_nasm_sub_view``, use ``-snes_view ::ascii_info_detail`` instead


   .. rubric:: SNESLineSearch:

   .. rubric:: TS:

   -  Change to ``--download-sundials2`` to indicate the version of SUNDIALS
      PETSc downloads, which is very old and out-dated
   -  Add forward and adjoint sensitivity support for cases that involve
      parameterized mass matrices
   -  Add ``TSGetNumEvents()`` to retrieve the number of events
   -  Add ``-ts_monitor_cancel``
   -  Now ``-ts_view_solution`` respects the TS prefix
   -  Add ``TSSetMatStructure()`` to indicate the relationship between the
      nonzero structures of the I Jacobian and the RHS Jacobian
   -  Automatically set the ``MatStructure`` flag of TS to
      ``SAME_NONZERO_PATTERN`` if the RHS matrix is obtained with a
      ``MatDuplicate()`` from the I Jacobian

   .. rubric:: TAO:

   -  Add ``TaoSetRecycleFlag()`` and ``TaoGetRecycleFlag()`` interfaces to
      enable some Tao algorithms to re-use iterate information from the
      previous ``TaoSolve()`` call
   -  Add new Augmented Lagrangian Multiplier Method (``TAOALMM``) for
      solving optimization problems with general nonlinear constraints

   .. rubric:: DM/DA:

   -  Remove unneeded ``Vec`` argument from ``DMPatchZoom()``
   -  Change ``DMDACreatePatchIS()`` to collective operation and add an
      extra argument to indicate whether off processor values will be
      returned
   -  Add ``DMComputeError()``, which uses ``PetscDS`` information for the exact
      solution
   -  Add ``DMShellGetGLobalVector()``
   -  Add ``DMInterpolateSolution()`` for interpolating solutions between
      meshes in a potentially nonlinear way
   -  ``DMInterpolationSetUp()`` now can drop points outside the domain

   .. rubric:: DMSwarm:

   -  ``DMSwarmViewXDMF()`` can now use a full path for the filename
   -  Add ``DMSwarmSetPointCoordinatesRandom()``
   -  Add ``-dm_view_radius`` to set size of drawn particles

   .. rubric:: DMPlex:

   -  Using ``-petscpartitioner_simple_node_grid`` and
      ``-petscpartitioner_simple_process_grid``, the Simple partitioner can
      now make grid partitions
   -  Add ``DMGet/SetFieldAvoidTensor()`` to allow fields to exclude tensor
      cells in their definition
   -  Remove regular refinement and marking from ``DMPlexCreateDoublet()``
   -  Add high order FEM interpolation to ``DMInterpolationEvaluate()``

   .. rubric:: FE/FV:

   -  Add ``PetscDualSpaceTransformHessian()``,
      ``PetscDualSpacePushforwardHessian()``, and
      ``PetscFEPushforwardHessian()``
   -  Now ``PetscFEGetCellTabulation()`` and ``PetscFEGetFaceTabulation()`` ask
      for the number of derivatives
   -  Add ``PetscDualSpaceLagrangeGet/SetUseMoments()`` and
      ``PetscDualSpaceLagrangeGet/SetMomentOrder()`` to allow a moment
      integral for P0

   .. rubric:: DMNetwork:

   -  Add ``DMNetworkAddSubnetwork()`` for network of subnetworks
   -  Add ``DMNetworkAdd/GetSharedVertices()``, ``DMNetworkIsSharedVertex()``
   -  Remove ``DMNetworkSetEdgeList()``,
      ``DMNetworkSet/GetComponentNumVariables()``,
      ``DMNetworkSet/Add/GetNumVariables()``,
      ``DMNetworkGetComponentKeyOffset()``, ``DMNetworkGetVariableOffset()``,
      ``DMNetworkGetVariableGlobalOffset()``
   -  Change the prototypes for ``DMNetworkAdd/GetComponent()``
   -  Rename ``DMNetworkSet/GetSizes()`` to ``DMNetworkSet/GetNumSubNetworks()``
   -  Rename ``DMNetworkGetComponentVariableOffset()`` to
      ``DMNetworkGetLocalVecOffset()``,
      ``DMNetworkGetComponentVariableGlobalOffset()`` to
      ``DMNetworkGetGlobalVecOffset()``
   -  Rename ``DMNetworkGetSubnetworkInfo()`` to ``DMNetworkGetSubnetwork()``

   .. rubric:: DT:

   -  ``PetscDSCopyBoundary()`` now takes a list of fields for which
      boundary copying is done
   -  Add ``PetscDSGet/SetJetDegree()``, and ``-dm_ds_jet_degree`` is needed to
      enable it under a DM
   -  Add ``PetscWeakForm`` class to manage function pointers for problem
      assembly

   .. rubric:: Fortran:

   -  Add configure option ``--with-mpi-f90module-visibility``
      [default=``1``]. With ``0``, ``mpi.mod`` will not be visible in use code
      (via ``petscsys.mod``) - so ``mpi_f08`` can now be used
   -  Add ``PetscDLAddr()`` to get name for a symbol
