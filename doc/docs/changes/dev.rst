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

.. rubric:: PC:

.. rubric:: KSP:

.. rubric:: SNES:

.. rubric:: SNESLineSearch:

.. rubric:: TS:

.. rubric:: TAO:

.. rubric:: DM/DA:

-  Add ``DMLabelGetNonEmptyStratumValuesIS()``, similar to ``DMLabelGetValueIS()`` but counts only nonempty strata
-  Add ``DMLabelCompare()`` for ``DMLabel`` comparison
-  Add ``DMCompareLabels()`` comparing ``DMLabel``s of two ``DM``s
-  ``DMCopyLabels()`` now takes DMCopyLabelsMode argument determining duplicity handling

.. rubric:: DMSwarm:

.. rubric:: DMPlex:

- Add ``DMExtrude()`` which now the default extrusion
- Change ``DMPlexExtrude()`` to use DMPlexTransform underneath

.. rubric:: FE/FV:

.. rubric:: DMNetwork:

.. rubric:: DMStag:

.. rubric:: DT:

.. rubric:: Fortran:
