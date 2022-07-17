====================
Changes: Development
====================

Changes you should make for main and version 3.18 so that it is portable to previous versions of PETSc

- Remove the error handling from uses of  ``PetscOptionsBegin()``, ``PetscOptionsEnd()``, ``PetscObjectOptionsBegin()``, ``PetscOptionsHead()``,  and ``PetscOptionsTail()``
- Remove the error handling from uses of ``PetscDrawCollectiveBegin()`` and ``PetscDrawCollectiveEnd()``
- Remove the error handling from uses of ``MatPreallocateInitialize()`` and ``MatPreallocateFinalize()``
- Replace ``MatUpdateMPIAIJWithArrays()`` with ``MatUpdateMPIAIJWithArray()``

Changes you can make for main and version 3.18 so that is not portable to previous versions of PETSc. This will remove all deprecation warnings when you build.
In addition to the changes above

- Change  ``PetscOptionsHead()`` and ``PetscOptionsTail()`` to  ``PetscOptionsHeadBegin()`` and ``PetscOptionsHeadEnd()``
- Change ``MatPreallocateInitialize()`` and ``MatPreallocateFinalize()`` to ``MatPreallocateBegin()`` and ``MatPreallocateEnd()``
- Change uses of `MatGetOption()` with `MAT_SYMMETRIC`, `MAT_STRUCTURALLY_SYMMETRIC`, `MAT_HERMITIAN`,  `MAT_SPD` to calls to `MatIsSymmetric()`, `MatIsSymmetricKnown()` etc.
- Whenever you call `MatSetOption()` with one of the above options and it is intended to stay with the matrix through calls to `MatSetValues()` etc add a call
  to `MatSetOption()` with `MAT_SYMMETRY_ETERNAL` etc

..
   STYLE GUIDELINES:
   * Capitalize sentences
   * Use imperative, e.g., Add, Improve, Change, etc.
   * Don't use a period (.) at the end of entries
   * If multiple sentences are needed, use a period or semicolon to divide sentences, but not at the end of the final sentence

.. rubric:: General:

- Change ``PetscOptionsBegin()``, ``PetscOptionsEnd()``, and ``PetscObjectOptionsBegin()`` to not return an error code
- Change ``PetscOptionsHead()``, ``PetscOptionsTail()``, to ``PetscOptionsHeadBegin()`` and ``PetscOptionsHeadEnd()`` and to not return an error code
- Add ``PETSC_ATTRIBUTE_FORMAT()`` to enable compile-time ``printf()``-style format specifier checking and apply it any PETSc functions taking a format string
- Deprecate the use of ``%D`` for printing ``PetscInt`` in favor of ``%" PetscInt_FMT "``. Compilers may now emit warnings when using ``%D`` as a result of applying ``PETSC_ATTRIBUTE_FORMAT``. Users that need to support older versions of PETSc may do one of two things:

  #. **Recommended** Insert the following code block *after* all PETSc header-file inclusions

     ::

        #if !defined(PetscInt_FMT)
        #  if defined(PETSC_USE_64BIT_INDICES)
        #    if !defined(PetscInt64_FMT)
        #      if defined(PETSC_HAVE_STDINT_H) && defined(PETSC_HAVE_INTTYPES_H) && defined(PETSC_HAVE_MPI_INT64_T)
        #        include <inttypes.h>
        #        define PetscInt64_FMT PRId64
        #      elif (PETSC_SIZEOF_LONG_LONG == 8)
        #        define PetscInt64_FMT "lld"
        #      elif defined(PETSC_HAVE___INT64)
        #        define PetscInt64_FMT "ld"
        #      else
        #        error "cannot determine PetscInt64 type"
        #      endif
        #    endif
        #    define PetscInt_FMT PetscInt64_FMT
        #  else
        #    define PetscInt_FMT "d"
        #  endif
        #endif


     This will ensure that the appropriate format specifiers are defined regardless of PETSc version.

  #. **Not Recommended** Compilers warnings can be permanently suppressed by defining ``PETSC_SKIP_ATTRIBUTE_FORMAT`` prior to all PETSc header-file inclusions

.. rubric:: Configure/Build:

- Remove python2 support, python-3.4+ is now required

.. rubric:: Sys:

-  Change -log_view to no longer print out the amount of memory associated with different types of objects. That data was often incorrect
-  Change ``PetscCall()`` from Fortran so that ``call PetscFunction(args,ierr);CHKERRQ(ierr);`` can be replaced with ``PetscCall(PetscFunction(args,ierr))``
-  Add ``PetscCallA()`` from Fortran so that ``call PetscFunction(args,ierr);CHKERRA(ierr);`` can be replaced with ``PetscCallA(PetscFunction(args,ierr))``
-  Add ``PetscCallMPI()`` and ``PetscCallMPIA()`` that may be used to call MPI functions from Fortran
-  Change the ``PetscCheck()`` and ``PetscAssert()`` macros to behave like function calls by wrapping in ``do { } while (0)``. Previously these macros expanded to ``if (...) SETERRQ(...)``, which meant they could be chained with subsequent conditionals.
-  Change ``PetscStackCallStandard()`` to ``PetscCallExternal()``
-  Change ``PetscStackCall()`` to ``PetscStackCallExternalVoid()``
-  Change ``PetscStackCallXXX()`` to ``PetscCallXXX()``
-  Add ``PetscCallBack()' for calling all PETSc callbacks (usually to user code) to replace the use of ``PetscStackPush()`` and ``PetscStackPop``

.. rubric:: PetscViewer:

.. rubric:: PetscDraw:

- Add ``PetscDrawSPGetDimension()``
-  Change ``PetscDrawCollectiveBegin()`` and ``PetscDrawCollectiveEnd()`` to not return an error code. Users can remove the error code checking for
   these functions and it will work correctly for all versions of PETSc

.. rubric:: AO:

.. rubric:: IS:

- Add ``ISShift()``

.. rubric:: VecScatter / PetscSF:

.. rubric:: PF:

.. rubric:: Vec:

- Add ``VecSetPreallocationCOO()``, ``VecSetValuesCOO()`` and ``VecSetPreallocationCOOLocal()`` to support vector assembly with coordinates

.. rubric:: PetscSection:

- Add ``PetscSectionCreateSubdomainSection()``

.. rubric:: PetscPartitioner:

.. rubric:: Mat:

- Change ``MatPreallocateInitialize()`` and ``MatPreallocateFinalize()`` to ``MatPreallocateBegin()`` and ``MatPreallocateEnd()`` and to not return an error code
- Change ``MatDenseGetSubMatrix()`` to be able to retrieve only selected contiguous rows instead of all rows
- Add ``MatSetOptionsPrefixFactor()`` and ``MatAppendOptionsPrefixFactor()`` to allow controlling the options prefix used by factors created from this matrix
- Change ``MatSetOptionsPrefix()`` to no longer affect the options prefix used by factors created from this matrix
- Change matrix factor options called from within `KSP`/`PC` to always inherit the options prefix from the `KSP`/`PC`, not the options prefix in the originating matrix
- Add ``MatIsStructurallySymmetricKnown()`` and ``MatIsSPDKnown()``
- Change ``MatGetOption()`` to no longer produce results for ``MAT_STRUCTURALLY_SYMMETRIC``, ``MAT_SYMMETRIC``, ``MAT_SPD``, and ``MAT_HERMITIAN``
- Add ``MatCreateGraph()`` to create a scalar matrix for use in graph algorithms
- Add ``MatFilter()`` to remove values with an absolute value equal to or below a give threshold
- Add an option -mat_factor_bind_factorization <host, device> to control where to do matrix factorization. Currently only supported with SEQAIJCUSPARSE matrices.
- Add ``MatUpdateMPIAIJWithArray()`` and deprecate ``MatUpdateMPIAIJWithArrays()``
- Change the coordinate array parameters in ``MatSetPreallocationCOO`` from const to non-const
- Add enforcement of the previously unenforced rule that ``MAT_REUSE_MATRIX`` with ``MatTranspose()`` can only be used after a call to ``MatTranspose()`` with ``MAT_INITIAL_MATRIX``. Add ``MatTransposeSetPrecursor()`` to allow using ``MAT_REUSE_MATRIX`` with ``MatTranspose()`` without the initial call to ``MatTranspose()``.
- Add ``MatTransposeSymbolic()``

.. rubric:: MatCoarsen:

- Add ``MISK`` coarsening type. Distance-k maximal independent set (MIS) C-F coarsening with a greedy, MIS based aggregation algorithm

.. rubric:: PC:

- Add PC type of mpi which can be used in conjunction with -mpi_linear_solver_server to use MPI parallelism to solve a system created on a single MPI rank

.. rubric:: KSP:

- Deprecate ``KSPHPDDMGetDeflationSpace()`` (resp. ``KSPHPDDMSetDeflationSpace()``) in favor of ``KSPHPDDMGetDeflationMat()`` (resp. ``KSPHPDDMSetDeflationMat()``)
- Add ``KSPNONE`` as alias for ``KSPPREONLY``

.. rubric:: SNES:

- Add ``DMDASNESSetFunctionLocalVec()``, ``DMDASNESSetJacobianLocalVec()`` and ``DMDASNESSetObjectiveLocalVec()``, and associate types ``DMDASNESFunctionVec``, ``DMDASNESJacobianVec`` and ``DMDASNESObjectiveVec``,
  which accept Vec parameters instead of void pointers in contrast to versions without the Vec suffix
- Add ``SNESLINESEARCHNONE`` as alias for ``SNESLINESEARCHBASIC``

.. rubric:: SNESLineSearch:

.. rubric:: TS:

- Add ``TSSetTimeSpan()``, ``TSGetTimeSpan()`` and ``TSGetTimeSpanSolutions()`` to support time span
- Add ``DMTSGetIFunctionLocal()``, ``DMTSGetIJacobianLocal()``, and ``DMTSGetRHSFunctionLocal()``

.. rubric:: TAO:

.. rubric:: DM/DA:

- Add ``DMDAMapMatStencilToGlobal()`` to map MatStencils to global indices
- Add ``DMGetCellCoordinateDM()``, ``DMSetCellCoordinateDM()``, ``DMGetCellCoordinateSection()``, ``DMSetCellCoordinateSection()``, ``DMGetCellCoordinates()``, ``DMSetCellCoordinates()``, ``DMGetCellCoordinatesLocalSetup()``, ``DMGetCellCoordinatesLocal()``, ``DMGetCellCoordinatesLocalNoncollective()``, ``DMSetCellCoordinatesLocal()``
- Add ``DMFieldCreateDSWithDG()`` to allow multiple representations of a given field
- Add ``DMProjectFieldLabel()``

.. rubric:: DMSwarm:

- Add ``DMSwarmGetCoordinateFunction()``, ``DMSwarmSetCoordinateFunction()``, ``DMSwarmGetVelocityFunction()``, ``DMSwarmSetVelocityFunction()`` to allow flexible layout of particles

.. rubric:: DMPlex:

- Add ``DMLabelPropagateBegin()``, ``DMLabelPropagatePush()``, and ``DMLabelPropagateEnd()``
- Add ``DMPlexPointQueue`` and API
- Add label value argument to ``DMPlexLabelCohesiveComplete()`` and ``DMPlexCreateHybridMesh()``
- Change ``DMPlexCheckPointSF()`` to take optional ``PetscSF`` parameter
- Add ``DMPlexCheck()``
- Add ``DMPlexMetricDeterminantCreate()`` for creating determinant fields for Riemannian metrics
- Change ``DMPlexMetricEnforceSPD()``:
    - pass determinant Vec, rather than its address
    - pass output metric, rather than its address
- Change ``DMPlexMetricNormalize()``:
    - pass output metric, rather than its address
    - pass determinant Vec, rather than its address
- Change ``DMPlexMetricAverage()``, ``DMPlexMetricAverage2()`` and ``DMPlexMetricAverage3()`` to pass output metric, rather than its address
- Change ``DMPlexMetricIntersection()``, ``DMPlexMetricIntersection2()`` and ``DMPlexMetricIntersection3()`` to pass output metric, rather than its address
- Add capability to specify whether the DMPlex should be reordered by default:
    - add ``DMPlexReorderDefaultFlag``
    - add ``DMPlexReorderGetDefault()`` and ``DMPlexReorderSetDefault()`` to get and set this flag
- Add ``DMPlexCreateOverlapLabelFromLabels()`` for more customized overlap
- Add ``DMPlexSetOverlap()`` to promote an internal interface
- Add ``DMGetCellCoordinateDM()``, ``DMSetCellCoordinateDM()``, ``DMGetCellCoordinateSection()``, ``DMSetCellCoordinateSection()``, ``DMGetCellCoordinates()``, ``DMSetCellCoordinates()``, ``DMGetCellCoordinatesLocalSetUp()``, ``DMGetCellCoordinatesLocal()``, ``DMGetCellCoordinatesLocalNoncollective()``, and ``DMSetCellCoordinatesLocal()`` to provide an independent discontinuous representation of coordinates
- Change ``DMGetPeriodicity()`` and ``DMSetPeriodicity()`` to get rid of the flag and boundary type. Since we have an independent representation, we can tell if periodicity was imposed, and boundary types were never used, so they can be inferred from the given L. We also add Lstart to allow tori that do not start at 0.
- Add ``DMPlexGetCellCoordinates()`` and ``DMPlexRestoreCellCoordinates()`` for clean interface for periodicity

.. rubric:: FE/FV:

- Add ``PetscFECreateFromSpaces()`` to build similar space from pieces

.. rubric:: DMNetwork:

.. rubric:: DMStag:

.. rubric:: DT:

- Add probability distributions ``PetscPDFGaussian3D()``, ``PetscPDFSampleGaussian3D()``, ``PetscPDFConstant2D()``, ``PetscCDFConstant2D()``, ``PetscPDFSampleConstant2D()``, ``PetscPDFConstant3D()``, ``PetscCDFConstant3D()``, ``PetscPDFSampleConstant3D()``

.. rubric:: Fortran:
