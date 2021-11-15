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

- C++ dialect will now also be inferred from compiler flags, although users will be warned that they should let PETSc auto-detect the flag when setting the dialect this way
- Change C++ dialect flag option to be consistent with compiler flags;  ``--with-cxx-dialect=gnu++14`` means you want ``-std=gnu++14``, no more, no less
- Fix for requesting no C++ dialect flag via ``--with-cxx-dialect=0``. Previously ``configure`` would bail out immediately without running the tests and therefore wouldn't set any of the capability defines. ``configure`` now runs all tests, just doesn't add the flag in the end
- Fix a number of corner-cases when handling C++ dialect detection
- Remove deprecated PETSC_VERSION_PATCH so as to not have confusion with patch releases where the subminor version changes

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
- Add ``PetscHasAttribute()`` macro to query for existence of an ``__attribute__`` specifier

.. rubric:: PetscViewer:

- Add  ``PetscViewerHDF5SetDefaultTimestepping`` and ``PetscViewerHDF5SetDefaultTimestepping`` to deal with HDF5 files missing the timestepping attribute

.. rubric:: PetscDraw:

.. rubric:: AO:

.. rubric:: IS:

-  ``ISLocalToGlobalMappingCreateSF()``: allow passing ``start = PETSC_DECIDE``

.. rubric:: VecScatter / PetscSF:

.. rubric:: PF:

.. rubric:: Vec:

-  Change ``VecTaggerComputeBoxes()`` and ``VecTaggerComputeIS()`` to return a boolean whose value is true if the list was created
-  Add ``-vec_bind_below`` option for specifying size threshold below which GPU is not used for ``Vec`` operations
-  Add ``VecSetBindingPropagates()``
-  Add ``VecGetBindingPropagates()``
-  For CUDA and ViennaCL and HIP GPU vectors, ``VecCreate()`` no longer allocates the array on CPU eagerly, it is only allocated if it is needed

.. rubric:: PetscSection:

.. rubric:: PetscPartitioner:

.. rubric:: Mat:

-  Add ``-mat_bind_below`` option for specifying size threshold below which GPU is not used for ``Mat`` operations
-  Add ``MatSetBindingPropagates()``
-  Add ``MatGetBindingPropagates()``

.. rubric:: PC:

.. rubric:: KSP:

-  Outer most ``KSPSolve()`` will error if KSP_DIVERGED_ITS and ```KSPSetErrorIfNotConverged()`` is used

.. rubric:: SNES:

.. rubric:: SNESLineSearch:

.. rubric:: TS:

- Add ``TSSundialsSetUseDense()`` and options database option ``-ts_sundials_use_dense`` to use a dense linear solver (serial only) within CVODE, instead of the default iterative solve.

.. rubric:: TAO:

.. rubric:: DM/DA:

-  Add ``DMLabelGetNonEmptyStratumValuesIS()``, similar to ``DMLabelGetValueIS()`` but counts only nonempty strata
-  Add ``DMLabelCompare()`` for ``DMLabel`` comparison
-  Add ``DMCompareLabels()`` comparing ``DMLabel``\s of two ``DM``\s
-  ``DMCopyLabels()`` now takes DMCopyLabelsMode argument determining duplicity handling
-  Add ``-dm_bind_below`` option for specifying size threshold below which GPU is not used for ``Vec`` and ``Mat`` objects associated with a DM

.. rubric:: DMSwarm:

.. rubric:: DMPlex:

- Add ``DMExtrude()`` which now the default extrusion
- Change ``DMPlexExtrude()`` to use DMPlexTransform underneath
- Move ``DMPlexMetricCtx`` from public to private and give it to ``DMPlex``
- Add ``DMPlexMetricSetFromOptions()`` to assign values to ``DMPlexMetricCtx``
- Add ``DMPlexMetricSetIsotropic()`` for declaring whether a metric is isotropic
- Add ``DMPlexMetricIsIsotropic()`` for determining whether a metric is isotropic
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
- Change ``DMPlexMetricEnforceSPD()`` to have another argument, for controlling whether anisotropy is restricted
- Change ``DMPlexMetricNormalize()`` to have another argument, for controlling whether anisotropy is restricted
- Change ``DMAdaptor`` so that its ``-adaptor_refinement_h_min/h_max/a_max/p`` command line arguments become ``-dm_plex_metric_h_min/h_max/a_max/p``
- Add ``DMGetNaturalSF()`` and ``DMSetNaturalSF()``

.. rubric:: FE/FV:

.. rubric:: DMNetwork:

.. rubric:: DMStag:

.. rubric:: DT:

- Add ``PetscDTPTrimmedEvalJet()`` to evaluate a stable basis for trimmed polynomials, and ``PetscDTPTrimmedSize()`` for the size of that space

.. rubric:: Fortran:
