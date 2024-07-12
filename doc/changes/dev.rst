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

- Add ``--with-openmp-kernels``

.. rubric:: Sys:

- Add ``PetscPragmaUseOMPKernels``
- Deprecate ``PetscOptionsRestoreViewer()`` in favor of ``PetscViewerDestroy()``
- Deprecate ``PetscOptionsGetViewer()``, and ``PetscOptionsGetViewers()`` in favor of ``PetscOptionsCreateViewer()`` and ``PetscOptionsCreateViewers()``
- Deprecate ``PetscOptionsPushGetViewerOff()``, ``PetscOptionsPopGetViewerOff()``, and ``PetscOptionsGetViewerOff()`` in favor of
  ``PetscOptionsPushCreateViewerOff()``, ``PetscOptionsPopCreateViewerOff()``, and ``PetscOptionsGetCreateViewerOff()``
- Add ``PetscObjectContainerCompose()``, and ``PetscObjectContainerQuery()``
- Add ``size_t`` argument to ``PetscMPIErrorString()``
- Add ``PetscCallExternalAbort()`` for calling external library functions from functions not returning ``PetscErrorCode``

.. rubric:: Event Logging:

.. rubric:: PetscViewer:

- Add ``PetscViewerASCIIStdoutSetFileUnit()``
- Add ``PetscShmgetAllocateArrayScalar()``, ``PetscShmgetDeallocateArrayScalar()``, ``PetscShmgetAllocateArrayInt()``, and ``PetscShmgetDeallocateArrayInt()`` for Fortran

.. rubric:: PetscDraw:

.. rubric:: AO:

.. rubric:: IS:

.. rubric:: VecScatter / PetscSF:

.. rubric:: PF:

.. rubric:: Vec:

- The ``IS`` passed to ``VecISAXPY()``, ``VecISCopy()``. ``VecISSet()``, and ``VecISShift()`` must have the same communicator of the vectors used
- Make ``VecLock`` API active in optimized mode
- ``VecNestSetSubVec()`` and ``VecNestSetSubVecs()`` now take references to input vectors rather than creating duplicates
- Deprecate ``VecSetInf()`` with ``VecFlag()``

.. rubric:: PetscSection:

.. rubric:: PetscPartitioner:

.. rubric:: Mat:

-  Change ``MatProductSetFill()`` to support ``PETSC_DETERMINE`` and ``PETSC_CURRENT``. ``MatMatMult()`` and its friends and relations now accept
   ``PETSC_DETERMINE`` and ``PETSC_CURRENT`` in the ``fill`` argument. ``PETSC_DEFAULT`` is deprecated for those functions
- Change the default ``MatType`` of the output ``Mat`` of ``MatSchurComplementComputeExplicitOperator()`` to be ``MATDENSE``. It may be changed from the command line, e.g., ``-fieldsplit_1_explicit_operator_mat_type aij``

.. rubric:: MatCoarsen:

.. rubric:: PC:

- Add support in ``PCFieldSplitSetFields()`` including with ``-pc_fieldsplit_%d_fields fields`` for ``MATNEST``,  making it possible to
  utilize multiple levels of ``PCFIELDSPLIT`` with ``MATNEST`` from the command line
- Add ``PCCompositeSpecialSetAlphaMat()`` API to use a matrix other than the identity in
  preconditioners based on an alternating direction iteration, e.g., setting :math:`M` for
  :math:`P = (A + alpha M) M^{-1} (alpha M + B)`
- Reuse the result of :math:`T = A_{00}^-1 A_{01}` in ``PCApply_FieldSplit_Schur`` with ``-pc_fieldsplit_schur_fact_type full``

- Change the option database keys for coarsening for ``PCGAMG`` to use the prefix ``-pc_gamg_``, for example ``-pc_gamg_mat_coarsen_type``

.. rubric:: KSP:

- Add support for ``PETSC_DETERMINE`` as an argument to ``KSPSetTolerances()`` to set the parameter back to its initial value when the object's type was set
- Deprecate ``PETSC_DEFAULT`` in favor of ``PETSC_CURRENT`` for  ``KSPSetTolerances()``

.. rubric:: SNES:

- Add support for ``PETSC_DETERMINE`` as an argument to ``SNESSetTolerances()`` to set the parameter back to its initial value when the object's type was set
- Deprecate ``PETSC_DEFAULT`` in favor of ``PETSC_CURRENT`` for  ``SNESSetTolerances()``
- Add ``DMAdaptorMonitor()``, ``DMAdaptorMonitorSet()``,  ``DMAdaptorMonitorCancel()``, ``DMAdaptorMonitorSetFromOptions()``
- Add ``DMAdaptorMonitorSize()``, ``DMAdaptorMonitorError()``, ``DMAdaptorMonitorErrorDraw()``, ``DMAdaptorMonitorErrorDrawLGCreate()``, ``DMAdaptorMonitorErrorDrawLG()``
- Add ``DMAdaptorMonitorRegister()``, ``DMAdaptorMonitorRegisterAll()``, ``DMAdaptorMonitorRegisterDestroy()``
- Add ``DMAdaptorGetCriterion()`` and ``DMAdaptorSetCriterion()``
- Add ``DMAdaptorSetOptionsPrefix()``
- Add Newton's method with arc length continuation: ``SNESNEWTONAL`` with ``SNESNewtonALSetFunction()``, ``SNESNewtonALGetFunction()``, ``SNESNewtonALComputeFunction()``, ``SNESNewtonALGetLoadParameter()``, and ``SNESNewtonALSetCorrectionType()``
- Add ``SNESNewtonTRSetTolerances()`` and ``SNESNewtonTRSetUpdateParameters()`` to programmatically set trust region parameters
- Deprecate ``SNESSetTrustRegionTolerance()`` in favor of ``SNESNewtonTRSetTolerances()``

.. rubric:: SNESLineSearch:

.. rubric:: TS:

- Add Rosenbrock-W methods from :cite:`rang2015improved` with :math:`B_{PR}` stability: ``TSROSWR34PRW``, ``TSROSWR3PRL2``, ``TSROSWRODASPR``, and ``TSROSWRODASPR2``
- Add support for ``PETSC_DETERMINE`` as an argument to ``TSSetTolerances()`` to set the parameter back to its initial value when the object's type was set
- Deprecate ``PETSC_DEFAULT`` in favor of ``PETSC_CURRENT`` for  ``TSSetTolerances()``
- Add support for ``PETSC_DETERMINE`` as an argument to ``TSSetMaxSteps()`` and ``TSSetMaxTime()``
- Deprecate ``PETSC_DEFAULT`` in favor of ``PETSC_CURRENT`` for ``TSAdaptSetSafety()``
- Deprecate ``PETSC_DEFAULT`` in favor of ``PETSC_CURRENT`` for ``TSAdaptSetClip()``
- Deprecate ``PETSC_DEFAULT`` in favor of ``PETSC_CURRENT`` for ``TSAdaptSetStepLimits()``
- Add  ``TSGetStepResize()``

.. rubric:: TAO:

- Add support for ``PETSC_DETERMINE`` as an argument to ``TaoSetTolerances()`` and ``TaoSetConstraintTolerances()`` to set the parameter back to its initial value when the object's type was set
- Deprecate ``PETSC_DEFAULT`` in favor of ``PETSC_CURRENT`` for  ``TaoSetTolerances()`` and ``TaoSetConstraintTolerances()``

.. rubric:: DM/DA:

- Add ``DMGetSparseLocalize()`` and ``DMSetSparseLocalize()``
- Add ``DMGeomModelRegister()``, ``DMGeomModelRegisterAll()``, ``DMGeomModelRegisterDestroy()``, ``DMSnapToGeomModel()``, ``DMSetSnapToGeomModel()`` to support registering geometric models
- Add ``DMGetOutputSequenceLength()``

.. rubric:: DMSwarm:

.. rubric:: DMPlex:

- Add ``DMLabelGetValueBounds()``
- Add ``DMPlexOrientLabel()``
- Add an argument to ``DMPlexLabelCohesiveComplete()`` in order to change behavior at surface boundary
- Remove ``DMPlexSnapToGeomModel()``
- Add refinement argument to ``DMPlexCreateHexCylinderMesh()``
- Now ``DMPlexComputeBdIntegral()`` takes one function per field
- Add ``DMPlexCreateEdgeNumbering()``
- Add ``DMPlexComputeL2FluxDiffVec()`` and ``DMPlexComputeL2FluxDiffVecLocal()``
- Add ``DMAdaptorSetType()``, ``DMAdaptorGetType()``, ``DMAdaptorRegister()``, ``DMAdaptorRegisterAll()``, ``DMAdaptorRegisterDestroy()``
- Add ``DMAdaptorGetMixedSetupFunction()`` and ``DMAdaptorSetMixedSetupFunction()``
- Add ``DMPlexCreateCellNumbering()``
- Add ``-dm_plex_box_label`` to add "Face Sets" label with current "box" conventions
- Add "Face Sets" label to simplex meshes using current "box" conventions

.. rubric:: FE/FV:

.. rubric:: DMNetwork:

.. rubric:: DMStag:

.. rubric:: DT:

- Add ``PetscDSSetIntegrationParameters()`` and ``PetscDSSetCellParameters()``

.. rubric:: Fortran:

- Add ``PETSC_NULL_ENUM`` to be used instead of ``PETSC_NULL_INTEGER`` when a pointer to an ``enum`` is expected in a PETSc function call
- Add ``PETSC_NULL_INTEGER_ARRAY``, ``PETSC_NULL_SCALAR_ARRAY``, and ``PETSC_NULL_REAL_ARRAY`` for use instead of
  ``PETSC_NULL_INTEGER``, ``PETSC_NULL_SCALAR``,  and ``PETSC_NULL_REAL`` when an array is expected in a PETSc function call
- Add automatically generated interface definitions for most PETSc functions to detect illegal usage at compile time
- Add ``PetscObjectIsNull()`` for users to check if a PETSc object is ``NULL``
- Change the PETSc Fortran API so that non-array values, ``v``, passed to PETSc routines expecting arrays must be cast with ``[v]`` in the calling sequence
