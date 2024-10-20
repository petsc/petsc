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

- Update ``--download-pastix`` to use CMake build, with additional dependency on LAPACKE and CBLAS, can use for ex. MKL  with ``--with-blaslapack-dir=${MKLROOT}``, or Netlib LAPACK with ``--download-netlib-lapack --with-netlib-lapack-c-bindings``

.. rubric:: Sys:

- Add ``PetscCIntCast()``
- Add ``PetscObjectHasFunction()`` to query for the presence of a composed method

.. rubric:: Event Logging:

.. rubric:: PetscViewer:

.. rubric:: PetscDraw:

.. rubric:: AO:

.. rubric:: IS:

- Add ``ISGetCompressOutput()`` and ``ISSetCompressOutput()``

.. rubric:: VecScatter / PetscSF:

.. rubric:: PF:

.. rubric:: Vec:

.. rubric:: PetscSection:

.. rubric:: PetscPartitioner:

.. rubric:: Mat:

.. rubric:: MatCoarsen:

.. rubric:: PC:

.. rubric:: KSP:

.. rubric:: SNES:

.. rubric:: SNESLineSearch:

.. rubric:: TS:

.. rubric:: TAO:

.. rubric:: DM/DA:

.. rubric:: DMSwarm:

.. rubric:: DMPlex:

- Add ``DMPlexTransformGetMatchStrata()`` and ``DMPlexTransformSetMatchStrata()``
- Deprecate ``DMPlexSetGlobalToNaturalSF()`` and ``DMPlexGetGlobalToNaturalSF()`` for existing ``DMSetNaturalSF()`` and ``DMGetNaturalSF()``

.. rubric:: FE/FV:

.. rubric:: DMNetwork:

.. rubric:: DMStag:

.. rubric:: DT:

.. rubric:: Fortran:
