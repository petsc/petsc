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

- Add ``PetscDeviceContextGetStreamHandle()`` to return a handle to the stream the current device context is using
- Add ``PetscStrcmpAny()`` to compare against multiple non-empty strings

.. rubric:: Event Logging:

.. rubric:: PetscViewer:

.. rubric:: PetscDraw:

.. rubric:: AO:

.. rubric:: IS:

.. rubric:: VecScatter / PetscSF:

.. rubric:: PF:

.. rubric:: Vec:

- Add ``VecErrorWeightedNorms()`` to unify weighted local truncation error norms used in ``TS``

.. rubric:: PetscSection:

.. rubric:: PetscPartitioner:

.. rubric:: Mat:

- Add ``MatCreateDenseFromVecType()``
- Add support for calling ``MatDuplicate()`` on a matirx preallocated via ``MatSetPreallocationCOO()``, and then ``MatSetValuesCOO()`` on the new matrix
- Remove ``MATSOLVERSPARSEELEMENTAL`` since it is no longer functional
- Add MATSELLCUDA. It supports fast ``MatMult()``, ``MatMultTranspose()`` and ``MatMultAdd()`` on GPUs
- Add support for ``MAT_FACTOR_LU`` and ``MAT_FACTOR_CHOLESKY`` with ``MATSOLVERMUMPS`` for ``MATNEST``
- ``MatGetFactor()`` can now return ``NULL`` for some combinations of matrices and solvers types. This is to support those combinations that can only be inspected at runtime (i.e. MatNest with AIJ blocks vs MatNest with SHELL blocks).
- Remove ``MatSetValuesDevice()``, ``MatCUSPARSEGetDeviceMatWrite()``, ``MatKokkosGetDeviceMatWrite``
- Add ``MatDenseCUDASetPreallocation()`` and ``MatDenseHIPSetPreallocation()``
- Add support for KOKKOS in ``MATH2OPUS``
- Add ``-pc_precision single`` option for use with ``MATSOLVERSUPERLU_DIST``

.. rubric:: MatCoarsen:

.. rubric:: PC:

.. rubric:: KSP:

- Add ``KSPSetMinimumIterations()`` and ``KSPGetMinimumIterations()``

.. rubric:: SNES:

- Add a convenient, developer-level ``SNESConverged()`` function that runs the convergence test and updates the internal converged reason.
- Swap the order of monitor and convergence test. Now monitors are always called after a convergence test.

.. rubric:: SNESLineSearch:

.. rubric:: TS:

- Remove ``TSErrorWeightedNormInfinity()``, ``TSErrorWeightedNorm2()``, ``TSErrorWeightedENormInfinity()``, ``TSErrorWeightedENorm2()`` since the same functionality can be obtained with ``VecErrorWeightedNorms()``

.. rubric:: TAO:

.. rubric:: DM/DA:

.. rubric:: DMSwarm:

.. rubric:: DMPlex:

.. rubric:: FE/FV:

.. rubric:: DMNetwork:

- Add ``DMNetworkViewSetShowRanks()``, ``DMNetworkViewSetViewRanks()``, ``DMNetworkViewSetShowGlobal()``, ``DMNetworkViewSetShowVertices()``, ``DMNetworkViewSetShowNumbering()``

- Add ``-dmnetwork_view_all_ranks`` ``-dmnetwork_view_rank_range`` ``-dmnetwork_view_no_vertices`` ``-dmnetwork_view_no_numbering`` for viewing DMNetworks with the Matplotlib viewer

.. rubric:: DMStag:

.. rubric:: DT:

.. rubric:: Fortran:

- Add ``PetscCheck()`` and ``PetscCheckA()`` for Fortran
