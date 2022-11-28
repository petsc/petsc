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

- Remove unused preprocessor variables ``PETSC_HAVE_VPRINTF_CHAR``, ``PETSC_HAVE_VFPRINTF_CHAR``, ``PETSC_STAT_MACROS_BROKEN``, ``PETSC_HAVE_FORTRAN_GETARG``, ``PETSC_uid_t``, ``PETSC_gid_t``, ``PETSC_HAVE_PTHREAD_BARRIER_T``, ``PETSC_HAVE_SCHED_CPU_SET_T``, and ``PETSC_HAVE_SYS_SYSCTL_H``
- Deprecate ``--with-gcov`` configure option. Users should use ``--with-coverage`` instead
- Add ``--with-coverage-exec`` configure option to specify the coverage-collection tool to be used e.g. ``gcov`` or ``/path/to/llvm-cov-15``

.. rubric:: Sys:

- Change ``PetscOptionsMonitorDefault()`` to also take in the option source, and ``PetscOptionsMonitorSet()`` to take the new monitor function.
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

.. rubric:: Event Logging:

.. rubric:: PetscViewer:

.. rubric:: PetscDraw:

.. rubric:: AO:

.. rubric:: IS:

.. rubric:: VecScatter / PetscSF:

- Change ``PetscSFConcatenate()`` to accept ``PetscSFConcatenateRootMode`` parameter; add option to concatenate root spaces globally

.. rubric:: PF:

.. rubric:: Vec:

- Document ``VecOperation``
- Add ``VECOP_SET``
- Significantly improve performance of ``VecMDot()``, ``VecMAXPY()`` and ``VecDotNorm2()`` for CUDA and HIP vector types. These routines should be between 2x and 4x faster.

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

- Add ``SNESPruneJacobianColor()`` to improve the MFFD coloring

.. rubric:: SNESLineSearch:

.. rubric:: TS:

- Add ``TSPruneIJacobianColor()`` to improve the MFFD coloring

.. rubric:: TAO:

.. rubric:: DM/DA:

- Add ``DMLabelGetType()``, ``DMLabelSetType()``, ``DMLabelSetUp()``, ``DMLabelRegister()``, ``DMLabelRegisterAll()``, ``DMLabelRegisterDestroy()``
- Add ``DMLabelEphemeralGetLabel()``, ``DMLabelEphemeralSetLabel()``, ``DMLabelEphemeralGetTransform()``, ``DMLabelEphemeralSetTransform()``

.. rubric:: DMSwarm:

.. rubric:: DMPlex:

- Add ``DMPlexGetOrientedCone()`` and ``DMPlexRestoreOrientedCone()`` to return both cone and orientation together
- Add ``DMPlexTransformGetChart()``, ``DMPlexTransformGetCellType()``, ``DMPlexTransformGetDepth()``, ``DMPlexTransformGetDepthStratum()``, ``DMPlexTransformGetConeSize()`` to enable ephemeral meshes
- Remove ``DMPlexAddConeSize()``
- Add ``DMPlexCreateEphemeral()``
- Both ``DMView()`` and ``DMLoad()`` now support parallel I/O with a new HDF5 format (see the manual for details)
- Remove ``DMPlexComputeGeometryFEM()`` since it was broken

.. rubric:: FE/FV:

.. rubric:: DMNetwork:
  - Add DMNetworkGetNumVertices to retrieve the local and global number of vertices in DMNetwork 
  - Add DMNetworkGetNumEdges to retrieve the local and global number of edges in DMNetwork 

.. rubric:: DMStag:

.. rubric:: DT:

.. rubric:: Fortran:
