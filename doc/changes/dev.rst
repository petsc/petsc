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

- Add ``PetscCtxDestroyFn`` as the prototype for all context destroy functions. It is ``PetscErrorCode ()(void **)``. Previously some context destructor
  setters took ``PetscErrorCode ()(void *)``. But these would not work directly with PETSc objects as contexts and having two different
  context destructor models added unneeded complexity to the library. This change is not backward compatible
- Deprecate ``PetscContainerSetUserDestroy()`` with ``PetscContainerSetCtxDestroy()``, updating will require a small change in calling code
- Deprecate ``PetscContainerCtxDestroyDefault`` with ``PetscCtxDestroyDefault()``
- Add ``PetscIntViewNumColumns()``, ``PetscScalarViewNumColumns()``, and ``PetscRealViewNumColumns()``

.. rubric:: Configure/Build:

- Update ``--download-pastix`` to use CMake build, with additional dependency on LAPACKE and CBLAS, can use for ex. MKL  with ``--with-blaslapack-dir=${MKLROOT}``, or Netlib LAPACK with ``--download-netlib-lapack --with-netlib-lapack-c-bindings``
- Add option ``--with-library-name-suffix=<suffix>``

.. rubric:: Sys:

- Add ``PetscCIntCast()``
- Add ``PetscObjectHasFunction()`` to query for the presence of a composed method
- Add ``PetscSortedCheckDupsCount()`` and ``PetscFindCount()``

.. rubric:: Event Logging:

.. rubric:: PetscViewer:

.. rubric:: PetscDraw:

- Add ``PetscDrawHGAddWeightedValue()``

.. rubric:: AO:

.. rubric:: IS:

- Add ``ISGetCompressOutput()`` and ``ISSetCompressOutput()``

.. rubric:: VecScatter / PetscSF:

.. rubric:: PF:

.. rubric:: Vec:

- Add ``PetscKDTree``, an implementation of K-d trees for efficient nearest-neighbor point searches. Includes ``PetscKDTreeCreate()``, ``PetscKDTreeDestroy()``, ``PetscKDTreeView()``, and then ``PetscKDTreeQueryPointsNearestNeighbor()`` for actually doing the nearest-neighbor query

.. rubric:: PetscSection:

.. rubric:: PetscPartitioner:

.. rubric:: Mat:

- Add ``MatCopyHashToXAIJ()`` which allows assembling an XAIJ matrix in hash table form into another XAIJ matrix
- Add ``MatResetHash()`` which allows resetting an XAIJ matrix to use a hash table

.. rubric:: MatCoarsen:

.. rubric:: PC:

- Add `PCHYPREGetCFMarkers()` to extract Coarse/Fine splittings created by BoomerAMG from `PCHYPRE`, similar to `PCGetInterpolations()` and `PCGetCoarseOperators()`

.. rubric:: KSP:

.. rubric:: SNES:

- Add ``DMPlexSetSNESVariableBounds()``

.. rubric:: SNESLineSearch:

- Add ``SNESLINESEARCHBISECTION`` as new SNES line search type, performing a bisection line search on the directional derivative

.. rubric:: TS:

.. rubric:: TAO:

.. rubric:: DM/DA:

- Deprecate ``DMGetSection()`` and ``DMSetSection()`` for existing ``DMGetLocalSection()`` and ``DMSetLocalSection()``

.. rubric:: DMSwarm:

- Add ``DMSwarmSortRestorePointsPerCell()``
- Change ``DMSwarmVectorGetField()`` and add ``DMSwarmVectorDefineFields()`` to handle multiple fields
- Add ``DMSwarmComputeMoments()``
- Add ``DMSwarmCellDMCreate()``, ``DMSwarmCellDMDestroy()``, ``DMSwarmCellDMView()``, ``DMSwarmCellDMGetDM()``, ``DMSwarmCellDMGetFields()``, ``DMSwarmCellDMGetCoordinateFields()``, ``DMSwarmCellDMGetCellID()``, ``DMSwarmCellDMGetSort()``, ``DMSwarmCellDMSetSort()``, and ``DMSwarmCellDMGetBlockSize()``
- Add ``DMSwarmAddCellDM()``, ``DMSwarmSetCellDMActive()``, and ``DMSwarmGetCellDMActive()``
- Add ``DMSwarmCreateGlobalVectorFromFields()``, ``DMSwarmDestroyGlobalVectorFromFields()``, ``DMSwarmCreateLocalVectorFromFields()``, and ``DMSwarmDestroyLocalVectorFromFields()``
- Add ``DMSwarmSortDestroy()``
- Add ``DMSwarmRemapType``, ``DMSwarmRemap()``, and ``DMSwarmDuplicate()``
- Add ``DMSwarmGetType()``
- Add ``DMSwarmGetCellDMByName()`` and ``DMSwarmGetCellDMNames()``

.. rubric:: DMPlex:

- Add ``DMPlexTransformGetMatchStrata()`` and ``DMPlexTransformSetMatchStrata()``
- Deprecate ``DMPlexSetGlobalToNaturalSF()`` and ``DMPlexGetGlobalToNaturalSF()`` for existing ``DMSetNaturalSF()`` and ``DMGetNaturalSF()``
- Add ``-dm_plex_box_label_bd`` to setup isoperiodicity when using ``-dm_plex_box_label_bd``
- Change ``PetscViewerCGNSGetSolutionTime()`` to no longer error if "TimeValues" array isn't found in CGNS file
- Add ``PetscViewerCGNSGetSolutionIteration()``
- Add ``DMPlexGetInterpolatePreferTensor()`` and ``DMPlexSetInterpolatePreferTensor()``
- Add ``PetscCallEGADS()``

.. rubric:: FE/FV:

.. rubric:: DMNetwork:

.. rubric:: DMStag:

.. rubric:: DT:

- Add ``PetscDSGetLowerBound()``, ``PetscDSSetLowerBound()``, ``PetscDSGetUpperBound()``, ``PetscDSSetUpperBound()``, ``PetscDSCopyBounds()``

.. rubric:: Fortran:
