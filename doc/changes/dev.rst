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

.. rubric:: MatCoarsen:

.. rubric:: PC:

- Add support in ``PCFieldSplitSetFields()`` including with ``-pc_fieldsplit_%d_fields fields`` for ``MATNEST``,  making it possible to
  utilize multiple levels of ``PCFIELDSPLIT`` with ``MATNEST`` from the command line

.. rubric:: KSP:

.. rubric:: SNES:

.. rubric:: SNESLineSearch:

.. rubric:: TS:

- Add Rosenbrock-W methods from :cite:`rang2015improved` with :math:`B_{PR}` stability: ``TSROSWR34PRW``, ``TSROSWR3PRL2``, ``TSROSWRODASPR``, and ``TSROSWRODASPR2``

.. rubric:: TAO:

.. rubric:: DM/DA:

- Add ``DMGetSparseLocalize()`` and ``DMSetSparseLocalize()``

.. rubric:: DMSwarm:

.. rubric:: DMPlex:

- Add ``DMLabelGetValueBounds()``
- Add ``DMPlexOrientLabel()``
- Add an argument to ``DMPlexLabelCohesiveComplete()`` in order to change behavior at surface boundary

.. rubric:: FE/FV:

.. rubric:: DMNetwork:

.. rubric:: DMStag:

.. rubric:: DT:

.. rubric:: Fortran:
