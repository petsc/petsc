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

- Add perfstubs package, see https://petsc.org/release/docs/manual/profiling/#using-tau for more information on usage

.. rubric:: Configure/Build:

- Remove unused preprocessor variables ``PETSC_HAVE_VPRINTF_CHAR``, ``PETSC_HAVE_VFPRINTF_CHAR``, ``PETSC_STAT_MACROS_BROKEN``, ``PETSC_HAVE_FORTRAN_GETARG``, ``PETSC_uid_t``, ``PETSC_gid_t``, ``PETSC_HAVE_PTHREAD_BARRIER_T``, ``PETSC_HAVE_SCHED_CPU_SET_T``, ``PETSC_HAVE_SYS_SYSCTL_H``, ``PETSC_HAVE_SYS_SYSINFO_H``, and ``PETSC_HAVE_SYSINFO_3ARG``
- Deprecate ``--with-gcov`` configure option in favor of ``--with-coverage``
- Add ``--with-coverage-exec`` configure option to specify the coverage-collection tool to be used e.g. ``gcov`` or ``/path/to/llvm-cov-15``
- Add ``--with-strict-petscerrorcode`` configure option to enable compile-time checking for correct usage of ``PetscErrorCode``, see below

.. rubric:: Sys:

- Change ``PetscOptionsMonitorDefault()`` to also take in the option source, and ``PetscOptionsMonitorSet()`` to take the new monitor function
- Deprecate ``PetscTable`` and related functions. Previous users of ``PetscTable`` are encouraged to use the more performant ``PetscHMapI`` instead, though they should note that this requires additional steps and limitations:

  #. ``#include <petscctable.h>`` must be swapped for ``#include <petsc/private/hashmapi.h>``. This of course requires that you have access to the private PETSc headers.
  #. While most of the old ``PetscTable`` routines have direct analogues in ``PetscHMapI``, ``PetscAddCount()`` does not. All uses of this routine should be replaced with the following snippet:

     ::

        // PetscHMapI hash_table;
        // PetscInt   key;

        PetscHashIter it;
        PetscBool     missing;

        PetscCall(PetscHMapIPut(hash_table, key, &it, &missing));
        if (missing) {
          PetscInt size;

          PetscCall(PetscHMapIGetSize(hash_table, &size));
          PetscCall(PetscHMapIIterSet(hash_table, it, size));
        }


  Furthermore, users should note that ``PetscHMapI`` is based on -- and directly ``#include`` s -- ``${PETSC_DIR}/include/petsc/private/khash/khash.h``. This file contains external source code that is licensed under the MIT license, which is separate from the PETSc license.

- Remove undocumented ``PETSC_MPI_WIN_FMT`` and ``PETSC_MPI_COMM_FMT``. Users should cast both ``MPI_Comm`` and ``MPI_Win`` to ``PETSC_INTPTR_T`` and use the ``PETSC_INTPTR_T_FMT`` format specifier instead:

     ::

        MPI_Comm comm;
        MPI_Win  win;

        // old
        PetscCall(PetscPrintf(..., "MPI Comm %" PETSC_MPI_COMM_FMT, comm));
        PetscCall(PetscPrintf(..., "MPI Window %" PETSC_MPI_WIN_FMT, win));

        // new
        PetscCall(PetscPrintf(..., "MPI Comm %" PETSC_INTPTR_T_FMT, (PETSC_INTPTR_T)comm));
        PetscCall(PetscPrintf(..., "MPI Window %" PETSC_INTPTR_T_FMT, (PETSC_INTPTR_T)win));


- Deprecate ``PETSC_NULL`` in favor of ``PETSC_NULLPTR`` as it does the right thing in both C and C++
- Significantly improve lookup and deletion performance of ``PetscFunctionList``. This also improves performance of ``PetscObjectComposeFunction()`` and ``PetscObjectQueryFunction()``.
- Optionally define ``PetscErrorCode`` as an ``enum``, and tag it as ``PETSC_NODISCARD``. This feature may be enabled by configuring PETSc with ``--with-strict-petscerrorcode`` configure option. This feature allows catching the following logical errors at compile-time:

  #. Not properly checking the return-code of PETSc calls via ``PetscCall()``. PETSc is left in an inconsistent state when errors are detected and cannot generally recover from them, so is not supported.
  #. Using the wrong ``PetscCall()`` variant, for example using ``PetscCall()`` on MPI functions (instead of ``PetscCallMPI()``).
  #. Returning ``PetscErrorCode`` from ``main()`` instead of ``int``.

  Users should note that this comes with the following additional changes:

  #. Add ``PETSC_SUCCESS`` to indicate success, always guaranteed to equal ``0``.
  #. ``PetscFunctionReturn(0)`` should be changed to ``PetscFunctionReturn(PETSC_SUCCESS)``. While the original ``0``-form will continue to work in C, it is required for C++.
  #. Any user-defined macros using boolean short-circuiting to chain multiple calls in the same line, which logically return a ``PetscErrorCode``, should now explicitly cast the "result" of the macro with ``PetscErrorCode``:


     ::

        // Both foo() and bar() defined as returning PetscErrorCode
        extern PetscErrorCode foo(int);
        extern PetscErrorCode bar(int);

        // The following macros logically "return" a PetscErrorCode, i.e. can
        // be used:
        //
        // PetscCall(MY_USER_MACRO(a, b));
        //
        // but use boolean short-circuiting to chain the calls together. bar()
        // only executes if foo() returns PETSC_SUCCESS

        // old
        #define MY_USER_MACRO(a, b) (foo(a) || bar(b))

        // new
        #define MY_BETTER_USER_MACRO(a, b) ((PetscErrorCode)(foo(a) || bar(b)))


  While currently opt-in, this feature **will be enabled by default in a future release**. Users are highly encourage to enable it and fix any discrepancies before that point. Note that ``PETSC_SUCCESS`` is defined whether or not the feature is enabled, so users may incrementally update.

.. rubric:: Event Logging:

.. rubric:: PetscViewer:

- The VTK viewers (``.vts``, ``.vtr``, and ``.vtu``) now use ``header_type="UInt64"`` to enable writing large binary appended blocks

.. rubric:: PetscDraw:

- Add ``PetscDrawSetVisible()`` to set if the drawing surface (the 'window') is visible on its display

.. rubric:: AO:

.. rubric:: IS:

- Change ``ISDuplicate()`` to preserve the block size of the input in the output

.. rubric:: VecScatter / PetscSF:

- Change ``PetscSFConcatenate()`` to accept ``PetscSFConcatenateRootMode`` parameter; add option to concatenate root spaces globally
- Add ``PetscSFSetGraphFromCoordinates()`` to construct a graph from fuzzy matching of coordinates; such as occurs for projections between different dimensions or for overlapping meshes

.. rubric:: PF:

.. rubric:: Vec:

- Document ``VecOperation``
- Add ``VECOP_SET``
- Significantly improve performance of ``VecMDot()``, ``VecMAXPY()`` and ``VecDotNorm2()`` for CUDA and HIP vector types. These routines should be between 2x and 4x faster.
- Enforce the rule that ``VecAssemblyBegin()`` and ``VecAssemblyEnd()`` must be called on even sequential vectors after calls to ``VecSetValues()``. This also applies to assignment of vector entries in petsc4py

.. rubric:: PetscSection:

.. rubric:: PetscPartitioner:

.. rubric:: Mat:

- ``MatSetValues()`` and friends will now provide reasonable performance when no preallocation information is provided
- Add ``MatEliminateZeros()``
- Improve efficiency of ``MatConvert()`` from ``MATNORMAL`` to ``MATHYPRE``
- Add ``MatDenseGetArrayAndMemType()``, ``MatDenseRestoreArrayAndMemType()``, ``MatDenseGetArrayReadAndMemType()``, ``MatDenseRestoreArrayReadAndMemType()``, ``MatDenseGetArrayWriteAndMemType()`` and ``MatDenseRestoreArrayWriteAndMemType()`` to return the array and memory type of a dense matrix

.. rubric:: MatCoarsen:

.. rubric:: PC:

- Add ``PCHPDDMSetSTShareSubKSP()``

.. rubric:: KSP:

- Add ``KSPMonitorDynamicToleranceCreate()`` and ``KSPMonitorDynamicToleranceSetCoefficient()``
- Change ``-sub_ksp_dynamic_tolerance_param`` to ``-sub_ksp_dynamic_tolerance``
- Add support for ``MATAIJCUSPARSE`` and ``VECCUDA`` to ``KSPHPDDM``

.. rubric:: SNES:

- Add ``SNESPruneJacobianColor()`` to improve the MFFD coloring
- Add ``SNESVIGetVariableBounds()`` to access variable bounds of a ``SNESVI``

.. rubric:: SNESLineSearch:

.. rubric:: TS:

- Add ``TSPruneIJacobianColor()`` to improve the MFFD coloring
- Add argument to ``TSMonitorSPCtxCreate()`` to enable multispecies plots
- Add ``TSMonitorHGCtxCreate()``, ``TSMonitorHGCtxDestroy()``, ``TSMonitorHGSwarmSolution()`` to support histogram plots of particle swarms

.. rubric:: TAO:

.. rubric:: DM/DA:

- Add ``DMLabelGetType()``, ``DMLabelSetType()``, ``DMLabelSetUp()``, ``DMLabelRegister()``, ``DMLabelRegisterAll()``, ``DMLabelRegisterDestroy()``
- Add ``DMLabelEphemeralGetLabel()``, ``DMLabelEphemeralSetLabel()``, ``DMLabelEphemeralGetTransform()``, ``DMLabelEphemeralSetTransform()``

.. rubric:: DMSwarm:

- Add ``DMSwarmGetMigrateType()`` and ``DMSwarmSetMigrateType()``

.. rubric:: DMPlex:

- Add ``DMPlexGetOrientedCone()`` and ``DMPlexRestoreOrientedCone()`` to return both cone and orientation together
- Add ``DMPlexTransformGetChart()``, ``DMPlexTransformGetCellType()``, ``DMPlexTransformGetDepth()``, ``DMPlexTransformGetDepthStratum()``, ``DMPlexTransformGetConeSize()`` to enable ephemeral meshes
- Remove ``DMPlexAddConeSize()``
- Add ``DMPlexCreateEphemeral()``
- Both ``DMView()`` and ``DMLoad()`` now support parallel I/O with a new HDF5 format (see the manual for details)
- Remove ``DMPlexComputeGeometryFEM()`` since it was broken
- Change ``DMPlexMarkBoundaryFaces()`` to avoid marking faces on the parallel boundary. To get the prior behavior, you can temporarily remove the ``PointSF`` from the ``DM``
- Add ``-dm_localize_height`` to localize edges and faces
- Add ``DMPlexCreateHypercubicMesh()`` to create hypercubic meshes needed for QCD
- Add ``-dm_plex_shape zbox`` option to ``DMSetFromOptions()`` to generated born-parallel meshes in Z-ordering (a space-filling curve). This may be used as-is with ``-petscpartitioner_type simple`` or redistributed using ``-petscpartitioner_type parmetis`` (or ``ptscotch``, etc.), which is more scalable than creating a serial mesh to partition and distribute.
- Add ``DMPlexSetIsoperiodicFaceSF()`` to wrap a non-periodic mesh into periodic while preserving the local point representation for both donor and image sheet. This is supported with ``zbox`` above, and allows single-element periodicity.

.. rubric:: FE/FV:

.. rubric:: DMNetwork:
  - Add DMNetworkGetNumVertices to retrieve the local and global number of vertices in DMNetwork
  - Add DMNetworkGetNumEdges to retrieve the local and global number of edges in DMNetwork
  - Add the ability to use ``DMView()`` on a DMNetwork with a PetscViewer with format ``PETSC_VIEWER_ASCII_CSV``
  - Add the ability to use ``-dmnetwork_view draw`` and ``-dmnetwork_view_distributed draw`` to visualize a DMNetwork with an associated coordinate DM. This currently requires the configured Python environment to have ``matplotlib`` and ``pandas`` installed

.. rubric:: DMStag:

.. rubric:: DT:

.. rubric:: Fortran:

- Add ``MatMPIAIJGetSeqAIJF90()``, ``MatMPIAIJRestoreSeqAIJF90()``
- Deprecate ``ISGetIndices()`` in favor of ``ISGetIndicesF90()``
- Deprecate ``ISRestoreIndices()`` in favor of ``ISRestoreIndicesF90()``
- Deprecate ``ISLocalToGlobalMappingGetIndices()`` in favor of ``ISLocalToGlobalMappingGetIndicesF90()``
- Deprecate ``ISLocalToGlobalMappingRestoreIndices()`` in favor of ``ISLocalToGlobalMappingRestoreIndicesF90()``
- Deprecate ``VecGetArray()`` in favor of ``VecGetArrayF90()``
- Deprecate ``VecRestoreArray()`` in favor of ``VecRestoreArrayF90()``
- Deprecate ``VecGetArrayRead()`` in favor of ``VecGetArrayReadF90()``
- Deprecate ``VecRestoreArrayRead()`` in favor of ``VecRestoreArrayReadF90()``
- Deprecate ``VecDuplicateVecs()`` in favor of ``VecDuplicateVecsF90()``
- Deprecate ``VecDestroyVecs()`` in favor of ``VecDestroyVecsF90()``
- Deprecate ``DMDAVecGetArray()`` in favor of ``DMDAVecGetArrayF90()``
- Deprecate ``DMDAVecRestoreArray()`` in favor of ``DMDAVecRestoreArrayF90()``
- Deprecate ``DMDAVecGetArrayRead()`` in favor of ``DMDAVecGetArrayReadF90()``
- Deprecate ``DMDAVecRestoreArrayRead()`` in favor of ``DMDAVecRestoreArrayReadF90()``
- Deprecate ``DMDAVecGetArrayWrite()`` in favor of ``DMDAVecGetArrayWriteF90()``
- Deprecate ``DMDAVecRestoreArrayWrite()`` in favor of ``DMDAVecRestoreArrayWriteF90()``
- Deprecate ``MatGetRowIJ()`` in favor of ``MatGetRowIJF90()``
- Deprecate ``MatRestoreRowIJ()`` in favor of ``MatRestoreRowIJF90()``
- Deprecate ``MatSeqAIJGetArray()`` in favor of ``MatSeqAIJGetArrayF90()``
- Deprecate ``MatSeqAIJRestoreArray()`` in favor of ``MatSeqAIJRestoreArrayF90()``
- Deprecate ``MatMPIAIJGetSeqAIJ()`` in favor of ``MatMPIAIJGetSeqAIJF90()``
- Deprecate ``MatDenseGetArray()`` in favor of ``MatDenseGetArrayF90()``
- Deprecate ``MatDenseRestoreArray()`` in favor of ``MatDenseRestoreArrayF90()``
