====================
Changes: Development
====================

Changes you should make for main and version 3.18 so that it is portable to previous versions of PETSc

- Remove the error handling from uses of  ``PetscOptionsBegin()``, ``PetscOptionsEnd()``, ``PetscObjectOptionsBegin()``, ``PetscOptionsHead()``,  and ``PetscOptionsTail()``
- Remove the error handling from uses of ``PetscDrawCollectiveBegin()`` and ``PetscDrawCollectiveEnd()``
- Remove the error handling from uses of ``MatPreallocateInitialize()`` and ``MatPreallocateFinalize()``

Changes you should make for main and version 3.18 so that is not portable to previous versions of PETSc. This will remove all deprecation warnings when you build.
In addition to the changes above

- Change  ``PetscOptionsHead()`` and ``PetscOptionsTail()`` to  ``PetscOptionsHeadBegin()`` and ``PetscOptionsHeadEnd()``
- Change ``MatPreallocateInitialize()`` and ``MatPreallocateFinalize()`` to ``MatPreallocateBegin()`` and ``MatPreallocateEnd()``

..
   STYLE GUIDELINES:
   * Capitalize sentences
   * Use imperative, e.g., Add, Improve, Change, etc.
   * Don't use a period (.) at the end of entries
   * If multiple sentences are needed, use a period or semicolon to divide sentences, but not at the end of the final sentence

.. rubric:: General:

- Change ``PetscOptionsBegin()``, ``PetscOptionsEnd()``, and ``PetscObjectOptionsBegin()`` to not return an error code
- Change ``PetscOptionsHead()``, ``PetscOptionsTail()``, to ``PetscOptionsHeadBegin()`` and ``PetscOptionsHeadEnd()`` and to not return an error code

.. rubric:: Configure/Build:

.. rubric:: Sys:

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

.. rubric:: PetscSection:

- Add ``PetscSectionCreateSubdomainSection()``

.. rubric:: PetscPartitioner:

.. rubric:: Mat:

- Change ``MatPreallocateInitialize()`` and ``MatPreallocateFinalize()`` to ``MatPreallocateBegin()`` and ``MatPreallocateEnd()`` and to not return an error code

.. rubric:: PC:

.. rubric:: KSP:

.. rubric:: SNES:

.. rubric:: SNESLineSearch:

.. rubric:: TS:

- Add ``TSSetTimeSpan()``, ``TSGetTimeSpan()`` and ``TSGetTimeSpanSolutions()`` to support time span
- Add ``DMTSGetIFunctionLocal()``, ``DMTSGetIJacobianLocal()``, and ``DMTSGetRHSFunctionLocal()``

.. rubric:: TAO:

.. rubric:: DM/DA:

.. rubric:: DMSwarm:

- Add ``DMSwarmGetCoordinateFunction()``, ``DMSwarmSetCoordinateFunction()``, ``DMSwarmGetVelocityFunction()``, ``DMSwarmSetVelocityFunction()`` to allow flexible layout of particles

.. rubric:: DMPlex:

.. rubric:: FE/FV:

.. rubric:: DMNetwork:

.. rubric:: DMStag:

.. rubric:: DT:

- Add probability distributions ``PetscPDFGaussian3D()``, ``PetscPDFSampleGaussian3D()``, ``PetscPDFConstant2D()``, ``PetscCDFConstant2D()``, ``PetscPDFSampleConstant2D()``, ``PetscPDFConstant3D()``, ``PetscCDFConstant3D()``, ``PetscPDFSampleConstant3D()``

.. rubric:: Fortran:
