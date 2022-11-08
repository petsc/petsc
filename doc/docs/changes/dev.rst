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

.. rubric:: Sys:

.. rubric:: Event Logging:

.. rubric:: PetscViewer:

.. rubric:: PetscDraw:

.. rubric:: AO:

.. rubric:: IS:

.. rubric:: VecScatter / PetscSF:

.. rubric:: PF:

.. rubric:: Vec:

.. rubric:: PetscSection:

.. rubric:: PetscPartitioner:

.. rubric:: Mat:

- Add ``MatEliminateZeros()``

.. rubric:: MatCoarsen:

.. rubric:: PC:

.. rubric:: KSP:

- Add ``KSPMonitorDynamicToleranceCreate()`` and ``KSPMonitorDynamicToleranceSetCoefficient()``
- Change ``-sub_ksp_dynamic_tolerance_param`` to ``-sub_ksp_dynamic_tolerance``

.. rubric:: SNES:

.. rubric:: SNESLineSearch:

.. rubric:: TS:

.. rubric:: TAO:

.. rubric:: DM/DA:

.. rubric:: DMSwarm:

.. rubric:: DMPlex:

- Add ``DMPlexGetOrientedCone()`` and ``DMPlexRestoreOrientedCone()`` to return both cone and orientation together
- Add ``DMPlexTransformGetChart()``, ``DMPlexTransformGetCellType()``, ``DMPlexTransformGetDepth()``, ``DMPlexTransformGetDepthStratum()``, ``DMPlexTransformGetConeSize()`` to enable ephemeral meshes
- Remove ``DMPlexAddConeSize()``

.. rubric:: FE/FV:

.. rubric:: DMNetwork:

.. rubric:: DMStag:

.. rubric:: DT:

.. rubric:: Fortran:
