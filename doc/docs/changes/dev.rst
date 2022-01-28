====================
Changes: Development
====================

..
   STYLE GUIDELINES:
   * Capitalize sentences
   * Use imperative, e.g., Add, Improve, Change, etc.
   * Don't use a period (.) at the end of entries
   * If multiple sentences are needed, use a period or semicolon to divide sentences, but not at the end of the final sentence
   * Use full function names, for ease of searching and so that man pages links are generated

.. rubric:: General:

- PETSc now requires a C99 compliant C compiler in all cases. Previously C99 was only required when building PETSc, but this now extends to public interfaces and header-files
- PETSc now requires a C++11 compliant C++ compiler. Note this requirement is only enforced if C++ is used; it is acceptable to have a compiler that does not support C++11 if you only ever build C source
- PETSc now requires at least Microsoft Visual Studio 2015 when using the Microsoft Visual C/C++ Compiler

.. rubric:: Configure/Build:

- Change minimum value of ``--with-cxx-dialect`` argument from "03" to "11"
- C++ dialect will now also be inferred from compiler flags, although users will be warned that they should let PETSc auto-detect the flag when setting the dialect this way
- Change C++ dialect flag option to be consistent with compiler flags;  ``--with-cxx-dialect=gnu++14`` means you want ``-std=gnu++14``, no more, no less
- Fix for requesting no C++ dialect flag via ``--with-cxx-dialect=0``. Previously ``configure`` would bail out immediately without running the tests and therefore wouldn't set any of the capability defines. ``configure`` now runs all tests, just doesn't add the flag in the end
- Fix a number of corner-cases when handling C++ dialect detection
- Remove deprecated ``PETSC_VERSION_PATCH`` so as to not have confusion with patch releases where the subminor version changes
- Change ``PETSC_HAVE_MKL`` to ``PETSC_HAVE_MKL_LIBS``
- Add ``PETSC_HAVE_MKL_INCLUDES``
- Enable HYPRE GPU for 64bit indices build (using HYPRE's mixed-int configuration)

.. rubric:: Sys:

- Add ``MPI_Comm_get_name()`` and ``MPI_Comm_set_name()`` to MPIUNI
- Remove ``petsccublas.h`` and ``petschipblas.h``
- Remove ``-petsc_use_default_null_stream`` and ``-[cuda|hip]_synchronize`` options
- Remove ``PetscCUDASynchronize`` and ``PetscHIPSynchronize``. Their operation is now managed by ``PetscDeviceContext`` via its ``PetscStreamType`` attribute
- Remove ``PetscCUDAInitialize()``, ``PetscCUDAInitializeCheck()``, ``PetscHIPInitialize()``, and ``PetscHIPInitializeCheck()``. Their function is now handled by ``PetscDeviceInitialize()`` and ``PetscDeviceInitialized()``
- Remove ``PetscCUBLASInitializeHandle()``, ``PetscCUSOLVERDnInitializeHandle()``, ``PetscHIPBLASInitializeHandle()``, and ``PetscHIPSOLVERInitializeHandle()``. Their function is now handled implicitly by ``PetscDeviceContext``
- Remove ``petsc_gputimer_begin`` and ``petsc_gputimer_begin``
- Add ``-device_enable``, ``-device_select`` and ``-device_view`` startup-options to control coarse-grained device initialization strategy
- Replace ``-[cuda|hip]_device`` with split options ``-device_enable_[cuda|hip]`` and ``-device_select_[cuda|hip]`` to enable fine-grained control of device selection and initialization strategy
- Replace ``-[cuda|hip]_view`` with ``-device_view_[cuda|hip]``
- Add ``PetscDeviceInitType`` to enumerate PETSc device initialization strategies
- Add ``PetscDeviceInitialize()`` to eagerly initialize a ``PetscDeviceType``, and ``PetscDeviceInitialized()`` to query the corresponding initialization state
- Change ``PetscDeviceCreate()`` to also accept a ``PetscInt devid``, to create a ``PetscDevice`` for a specific device
- Add ``PetscDeviceView()``
- Move ``PetscInt64_FMT`` and ``MPIU_INT64`` definitions to ``petscsystypes.h``
- Add ``PetscBLASInt_FMT``, ``PETSC_MPI_COMM_FMT``, and ``PETSC_MPI_WIN_FMT`` format specifiers
- Add ``petscmacros.h`` header to house common PETSc preprocessor macros
- Add ``PetscUnreachable()`` to indicate unreachable code section to compiler
- Add ``PetscHasAttribute()`` macro to query for existence of an ``__attribute__`` specifier
- Add ``PetscCommGetComm()`` and ``PetscCommRestoreComm()`` to allow reuse of MPI communicator with external packages, as some MPI implementations have  broken ``MPI_Comm_free()``
- Add ``PetscExpand()``, ``PetscConcat()``, ``PetscCompl()``, and ``PetscExpandToNothing()``
- Add ``PETSC_CONSTEXPR_14``, ``PETSC_NULLPTR``, and ``PETSC_NODISCARD``
- Add ``PetscSizeT`` as a language-agnostic equivalent of ``size_t`` from ``<stddef.h>``
- Add ``PetscCount`` as a signed datatype for counts, equivalent to ``ptrdiff_t`` from ``<stddef.h>``.
- Deprecate ``SETERRQ1()`` - ``SETERRQ9()`` in favor of ``SETERRQ()`` which is now variadic
- Deprecate ``PetscInfo1()`` - ``PetscInfo9()`` in favor of ``PetscInfo()`` which is now variadic
- Deprecate ``PETSC_INLINE``, ``inline`` is a standard keyword since C99 and C++11
- Remove ``PETSC_C_RESTRICT``, ``restrict`` is a standard keyword since C99
- Change ``SETERRMPI()`` to be variadic
- Change ``SETERRABORT()`` to be variadic

.. rubric:: PetscViewer:

- Add  ``PetscViewerHDF5SetDefaultTimestepping()`` and ``PetscViewerHDF5SetDefaultTimestepping()`` to deal with HDF5 files missing the timestepping attribute

.. rubric:: PetscDraw:

.. rubric:: AO:

.. rubric:: IS:

-  ``ISLocalToGlobalMappingCreateSF()``: allow passing ``start = PETSC_DECIDE``
-  Add ``ISGeneralSetIndicesFromMask()``

.. rubric:: VecScatter / PetscSF:

- Add MPI-4.0 large count support. With an MPI-4.0 compliant MPI implementation and 64-bit indices, one can now pass over 2 billion elements in a single message in either VecScatter or PetscSF
- Add ``PetscSFFetchAndOpWithMemTypeBegin()``, which is similar to ``PetscSFFetchAndOpBegin()``, but with explicit memory types

.. rubric:: PF:

.. rubric:: Vec:

-  Change ``VecTaggerComputeBoxes()`` and ``VecTaggerComputeIS()`` to return a boolean whose value is true if the list was created
-  Add ``-vec_bind_below`` option for specifying size threshold below which GPU is not used for ``Vec`` operations
-  Add ``VecSetBindingPropagates()``
-  Add ``VecGetBindingPropagates()``
-  For CUDA and ViennaCL and HIP GPU vectors, ``VecCreate()`` no longer allocates the array on CPU eagerly, it is only allocated if it is needed
-  ``VecGetArrayAndMemType()`` and ``VecGetArrayReadAndMemType()`` now always return a device pointer (copying the data to the device if needed) for the standard CUDA, HIP, and CUDA/HIP Kokkos vectors. Previously, they did so only when the device had the latest data
-  Add ``VecGetArrayWriteAndMemType()`` and  ``VecRestoreArrayWriteAndMemType()``, which are similar to the ``VecGetArrayReadAndMemType()`` family, but only write to the vector on device

.. rubric:: PetscSection:

.. rubric:: PetscPartitioner:

.. rubric:: Mat:

-  Add ``-mat_bind_below`` option for specifying size threshold below which GPU is not used for ``Mat`` operations
-  Add ``MatSetBindingPropagates()``
-  Add ``MatGetBindingPropagates()``
-  Add ``MatSeqAIJGetArrayWrite()`` and ``MatSeqAIJRestoreArrayWrite()`` to get write-access to the value array of ``MatSeqAIJ`` on CPU
-  Add ``MatCUSPARSESetUseCPUSolve()`` Use CPU solve with cuSparse for LU factorization that are on the CPU
-  Change ``MatCreateIS()`` behavior when NULL is passed for the mappings. Now a NULL map implies matching local and global spaces
-  Add ``MatMultHermitianTransposeEqual()`` and ``MatMultHermitianTransposeAddEqual()``
-  Add support of ``MatSetValuesCOO()`` and ``MatSetPreallocationCOO()`` for matrix type AIJKOKKOS. Additionally, for AIJKOKKOS, they support negative indices and remote entries
-  Add ``MatSetPreallocationCOOLocal()`` to set preallocation for matrices using a coordinate format of the entries with local indices
- Change ``MatStructures`` enumeration to avoid spaces and match capitalization of other enumerations
-  Change size argument of ``MatSetPreallocationCOO()`` to ``PetscCount``
-  Add ``MATORDERINGMETISND`` use METIS for nested dissection ordering of ``MatSeqAIJ``, with options ``nseps``, ``niter``, ``ufactor`` and ``pfactor`` under the common prefix ``-mat_ordering_metisnd_``
-  Change options ``-matproduct_<product_type>_via`` to ``-mat_product_algorithm``

.. rubric:: PC:

- Add MG option ``-pc_mg_galerkin_mat_product_algorithm [cusparse|hypre]`` and ``PCMGGalerkinSetMatProductAlgorithm()`` to use cuSparse or hypre's SpGEMM for Galerkin products in hypre

.. rubric:: KSP:

-  Outer most ``KSPSolve()`` will error if KSP_DIVERGED_ITS and ```KSPSetErrorIfNotConverged()`` is used
-  Add ``KSPQMRCGS`` to support qmrcgstab with right preconditioning

.. rubric:: SNES:

-  Add ``SNESNewtonTRDCGetRhoFlag()``, ``SNESNewtonTRDCSetPreCheck()``, ``SNESNewtonTRDCGetPreCheck()``, ``SNESNewtonTRDCSetPostCheck()``, ``SNESNewtonTRDCGetPostCheck()``

.. rubric:: SNESLineSearch:

.. rubric:: TS:

- Add ``TSSundialsSetUseDense()`` and options database option ``-ts_sundials_use_dense`` to use a dense linear solver (serial only) within CVODE, instead of the default iterative solve
- Change timestepper type ``TSDISCGRAD`` to include additional conservation terms based on formulation from [Gonzalez 1996] for Hamiltonian systems:
  - Add ``TSDiscGradIsGonzalez()`` to check flag for whether to use additional conservative terms in discrete gradient formulation
  - Add ``TSDiscGradUseGonzalez()`` to set discrete gradient formulation with or without additional conservative terms.  Without flag, the discrete gradients timestepper is just backwards euler
- Add ``TSRemoveTrajectory`` to destroy and remove the internal TSTrajectory object from TS

.. rubric:: TAO:

.. rubric:: DM/DA:

-  Add ``DMLabelGetNonEmptyStratumValuesIS()``, similar to ``DMLabelGetValueIS()`` but counts only nonempty strata
-  Add ``DMLabelCompare()`` for ``DMLabel`` comparison
-  Add ``DMCompareLabels()`` comparing ``DMLabel``\s of two ``DM``\s
-  ``DMCopyLabels()`` now takes DMCopyLabelsMode argument determining duplicity handling
-  Add ``-dm_bind_below`` option for specifying size threshold below which GPU is not used for ``Vec`` and ``Mat`` objects associated with a DM
-  Add ``DMCreateMassMatrixLumped()`` to support explicit timestepping, also add ``DMTSCreateRHSMassMatrix()``, ``DMTSCreateRHSMassMatrixLumped()``, and ``DMTSDestroyRHSMassMatrix()``
- Promote ``DMGetFirstLabelEntry()`` to public API and rename

.. rubric:: DMSwarm:

.. rubric:: DMPlex:

- Add ``DMExtrude()`` which now the default extrusion
- Change ``DMPlexExtrude()`` to use DMPlexTransform underneath
- Add ``DMGetNaturalSF()`` and ``DMSetNaturalSF()``
- Change ``-dm_plex_csr_via_mat`` to ``-dm_plex_csr_alg`` which takes a DMPlexCSRAlgorithm name
- Add public API for metric-based mesh adaptation:
    - Move ``DMPlexMetricCtx`` from public to private and give it to ``DMPlex``
    - Add ``DMPlexMetricSetFromOptions()`` to assign values to ``DMPlexMetricCtx``
    - Add ``DMPlexMetricSetIsotropic()`` for declaring whether a metric is isotropic
    - Add ``DMPlexMetricIsIsotropic()`` for determining whether a metric is isotropic
    - Add ``DMPlexMetricSetUniform()`` for declaring whether a metric is uniform
    - Add ``DMPlexMetricIsUniform()`` for determining whether a metric is uniform
    - Add ``DMPlexMetricSetRestrictAnisotropyFirst()`` for declaring whether anisotropy should be restricted before normalization
    - Add ``DMPlexMetricRestrictAnisotropyFirst()`` for determining whether anisotropy should be restricted before normalization
    - Add ``DMPlexMetricSetMinimumMagnitude()`` for specifying the minimum tolerated metric magnitude
    - Add ``DMPlexMetricGetMinimumMagnitude()`` for retrieving the minimum tolerated metric magnitude
    - Add ``DMPlexMetricSetMaximumMagnitude()`` for specifying the maximum tolerated metric magnitude
    - Add ``DMPlexMetricGetMaximumMagnitude()`` for retrieving the maximum tolerated metric magnitude
    - Add ``DMPlexMetricSetMaximumAnisotropy()`` for specifying the maximum tolerated metric anisostropy
    - Add ``DMPlexMetricGetMaximumAnisotropy()`` for retrieving the maximum tolerated metric anisotropy
    - Add ``DMPlexMetricSetTargetComplexity()`` for specifying the target metric complexity
    - Add ``DMPlexMetricGetTargetComplexity()`` for retrieving the target metric complexity
    - Add ``DMPlexMetricSetNormalizationOrder()`` for specifying the order of L-p normalization
    - Add ``DMPlexMetricGetNormalizationOrder()`` for retrieving the order of L-p normalization
    - Change ``DMPlexMetricCtx`` so that it is only instantiated when one of the above routines are called
    - Change ``DMPlexMetricEnforceSPD()`` to have more arguments:
        - control whether anisotropy is restricted
        - output the modified metric, rather than modifying the input
        - output the determinant
    - Change ``DMPlexMetricNormalize()`` to have another argument, for controlling whether anisotropy is restricted
- Change ``DMAdaptor`` so that its ``-adaptor_refinement_h_min/h_max/a_max/p`` command line arguments become ``-dm_plex_metric_h_min/h_max/a_max/p``
- Add 2D and 3D mesh adaptation interface to Mmg and 3D mesh adaptation interface to ParMmg. Mmg/ParMmg specific changes:
    - Change ``DMPlexBuildFromCellListParallel()`` to have another argument, for the connectivity
    - Change ``DMPlexCreateFromCellListParallelPetsc()`` to have another argument, for the connectivity
    - Add ``DMPlexMetricSetVerbosity()`` for setting the verbosity of the metric-based mesh adaptation package
    - Add ``DMPlexMetricGetVerbosity()`` for getting the verbosity of the metric-based mesh adaptation package
    - Add ``DMPlexMetricSetNoInsertion()`` to turn off node insertion and deletion for (Par)Mmg
    - Add ``DMPlexMetricNoInsertion()`` to determine whether node insertion and deletion are turned off for (Par)Mmg
    - Add ``DMPlexMetricSetNoSwapping()`` to turn off facet swapping for (Par)Mmg
    - Add ``DMPlexMetricNoSwapping()`` to determine whether facet swapping is turned off for (Par)Mmg
    - Add ``DMPlexMetricSetNoMovement()`` to turn off node movement for (Par)Mmg
    - Add ``DMPlexMetricNoMovement()`` to determine whether node movement is turned off for (Par)Mmg
    - Add ``DMPlexMetricSetGradationFactor()`` to set the metric gradation factor
    - Add ``DMPlexMetricGetGradationFactor()`` to get the metric gradation factor
    - Add ``DMPlexMetricSetNumIterations()`` to set the number of ParMmg adaptation iterations
    - Add ``DMPlexMetricGetNumIterations()`` to get the number of ParMmg adaptation iterations
- Change ``DMPlexCoordinatesLoad()`` to take a ``PetscSF`` as argument
- Change ``DMPlexLabelsLoad()`` to take the ``PetscSF`` argument and load in parallel
- Change ``DMPlexCreateFromFile()`` to take the mesh name as argument
- Change ``DMAdaptMetric`` so that it takes an additional argument for cell tags
- Change ``DMTransformAdaptLabel`` so that it takes an additional argument for cell tags
- Change ``DMGenerateRegister`` so that it registers routines that take an additional argument for cell tags
- Change ``DMPlexFindVertices()`` to take ``Vec`` and ``IS`` arguments instead of arrays
- Add ``DMPlexTSComputeRHSFunctionFEM()`` to support explicit timestepping

.. rubric:: FE/FV:

- Deprecate ``PetscSpacePolynomialGetSymmetric()`` and ``PetscSpacePolynomialSetSymmetric()``: symmetric polynomials were never supported and support is no longer planned
- Remove ``PetscSpacePolynomialType`` enum and associated array of strings ``PetscSpacePolynomialTypes``: other polynomial spaces are now handled by other implementations of ``PetscSpace``
- Add ``PETSCSPACEPTRIMMED`` that implements trimmed polynomial spaces (also known as the spaces in Nedelec face / edge elements of the first kind)
- Replace ``PetscDSGet/SetHybrid()`` with ``PetscDSGet/SetCohesive()``
- Add ``PetscDSIsCohesive()``, ``PetscDSGetNumCohesive()``, and ``PetscDSGetFieldOffsetCohesive()``
- Add argument to ``PetscFEIntegrateHybridJacobian()`` to indicate the face for the integration

.. rubric:: DMNetwork:

-  ``DMNetworkAddComponent()`` now requires a valid component key for each call

.. rubric:: DMStag:

.. rubric:: DT:

- Add ``PetscDTPTrimmedEvalJet()`` to evaluate a stable basis for trimmed polynomials, and ``PetscDTPTrimmedSize()`` for the size of that space
- Add ``PetscDSGetRHSResidual()`` and ``PetscDSSetRHSResidual()`` to support explicit timestepping

.. rubric:: Fortran:
