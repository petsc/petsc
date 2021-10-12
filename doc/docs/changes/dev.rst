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
-  ``ISLocalToGlobalMappingCreateSF()``: allow passing ``start = PETSC_DECIDE``

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

.. rubric:: FE/FV:

.. rubric:: DMNetwork:

.. rubric:: DMStag:

.. rubric:: DT:

- Add ``PetscDTPTrimmedEvalJet()`` to evaluate a stable basis for trimmed polynomials, and ``PetscDTPTrimmedSize()`` for the size of that space

.. rubric:: Fortran:
