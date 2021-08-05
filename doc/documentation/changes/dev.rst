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

-  Change ``MPIU_Allreduce()`` to always returns a MPI error code that
   should be checked with ``CHKERRMPI(ierr)``
-  Add support for ESSL 5.2 and later; drop support for ESSL <=5.1.

.. rubric:: Configure/Build:
-  Remove --with-kokkos-cuda-arch. One can use -with-cuda-gencodearch to specify the cuda arch for Kokkos. Usually not needed since PETSc auto detects that
-  For --download-hdf5, disable --download-hdf5-fortran-bindings by default

.. rubric:: Sys:
-  Add GPU event timers to capture kernel execution time accurately.
-  Remove ``WaitForCUDA()`` and ``WaitForHIP()`` before ``PetscLogGpuTimeEnd()``
-  Add MPIU_REAL_INT and MPIU_SCALAR_INT datatypes to be used for reduction operations
-  Add MPIU_MAXLOC and MPIU_MINLOC operations
-  Add ``CHKERRCXX()`` to catch C++ exceptions and return a PETSc error code

.. rubric:: PetscViewer:

-  ``PetscViewerHDF5PushGroup()``: if input path begins with ``/``, it is
   taken as absolute, otherwise relative to the current group
-  Add ``PetscViewerHDF5HasDataset()``
-  ``PetscViewerHDF5HasAttribute()``,
   ``PetscViewerHDF5ReadAttribute()``,
   ``PetscViewerHDF5WriteAttribute()``,
   ``PetscViewerHDF5HasDataset()`` and
   ``PetscViewerHDF5HasGroup()``
   support absolute paths (starting with ``/``)
   and paths relative to the current pushed group
-  Add input argument to ``PetscViewerHDF5ReadAttribute()`` for default
   value that is used if attribute is not found in the HDF5 file
-  Add ``PetscViewerHDF5PushTimestepping()``,
   ``PetscViewerHDF5PopTimestepping()`` and
   ``PetscViewerHDF5IsTimestepping()`` to control timestepping mode.
-  One can call ``PetscViewerHDF5IncrementTimestep()``,
   ``PetscViewerHDF5SetTimestep()`` or ``PetscViewerHDF5GetTimestep()`` only
   if timestepping mode is active
-  Error if timestepped dataset is read/written out of timestepping mode, or
   vice-versa

.. rubric:: PetscDraw:

.. rubric:: AO:

.. rubric:: IS:

.. rubric:: VecScatter / PetscSF:

.. rubric:: PF:

.. rubric:: Vec:

.. rubric:: PetscSection:

-  Extend ``PetscSectionView()`` for section saving to HDF5
-  Add ``PetscSectionLoad()`` for section loading from HDF5

.. rubric:: PetscPartitioner:

.. rubric:: Mat:

-  Factorization types now provide their preferred ordering (which
   may be ``MATORDERINGEXTERNAL``) to prevent PETSc PCFactor from, by
   default, picking an ordering when it is not ideal
-  Deprecate ``MatFactorGetUseOrdering()``; Use
   ``MatFactorGetCanUseOrdering()`` instead
-  Add ``--download-htool`` to use hierarchical matrices with the new
   type ``MATHTOOL``
-  Add ``MATCENTERING`` special matrix type that implements action of the
   centering matrix
-  Remove -mat_mumps_icntl_7 1 option, use -pc_factor_mat_ordering_type <type> to have PETSc perform the ordering (sequential only)
-  Add ``MATSOLVERSPQR`` - interface to SuiteSparse QR factorization
-  Add ``MatSeqAIJKron()`` - Kronecker product of two ``MatSeqAIJ``
-  Add ``MatNormalGetMat()`` to retrieve the underlying ``Mat`` of a ``MATNORMAL``
-  Add ``VecCreateMPICUDA()`` and ``VecCreateMPIHIP()`` to create MPI device vectors

.. rubric:: PC:

-  Add ``PCQR`` - interface to SuiteSparse QR factorization for ``MatSeqAIJ`` and
   ``MATNORMAL``
-  Add support for ``MATNORMAL`` in ``PCASM`` and ``PCHPDDM``
-  ``PCShellGetContext()`` now takes ``void*`` as return argument

.. rubric:: KSP:

-  ``KSPGetMonitorContext()`` now takes ``void*`` as return argument
-  ``KSPGetConvergenceContext()`` now takes ``void*`` as return argument

.. rubric:: SNES:

-  Add ``SNESSetComputeMFFunction()``

-  Add support for ``-snes_mf_operator`` for use with ``SNESSetPicard()``
-  ``SNESShellGetContext()`` now takes ``void*`` as return argument

.. rubric:: SNESLineSearch:

.. rubric:: TS:

.. rubric:: TAO:

-  ``TaoShellGetContext()`` now takes ``void*`` as return argument

.. rubric:: DM/DA:

-  Change management of auxiliary data in DM from object composition
   to ``DMGetAuxiliaryVec()``/``DMSetAuxiliaryVec()``, ``DMCopyAuxiliaryVec()``
-  Remove ``DMGetNumBoundary()`` and ``DMGetBoundary()`` in favor of DS
   counterparts
-  Remove ``DMCopyBoundary()``
-  Change interface for ``DMAddBoundary()``, ``PetscDSAddBoundary()``,
   ``PetscDSGetBoundary()``, ``PetscDSUpdateBoundary()``
-  Add ``DMDAVecGetArrayDOFWrite()`` and ``DMDAVecRestoreArrayDOFWrite()``
-  ``DMShellGetContext()`` now takes ``void*`` as return argument

.. rubric:: DMSwarm:

-  Add ``DMSwarmGetCellSwarm()`` and ``DMSwarmRestoreCellSwarm()``

.. rubric:: DMPlex:

-  Add a ``PETSCVIEWEREXODUSII`` viewer type for ``DMView()``/``DMLoad()`` and
   ``VecView()``/``VecLoad()``. Note that not all DMPlex can be saved in exodusII
   format since this file format requires that the numbering of cell
   sets be compact
-  Add ``PetscViewerExodusIIOpen()`` convenience function
-  Add ``PetscViewerExodusIISetOrder()`` to
   generate "2nd order" elements (i.e. tri6, tet10, hex27) when using
   ``DMView`` with a ``PETSCVIEWEREXODUSII`` viewer
-  Change ``DMPlexComputeBdResidualSingle()`` and
   ``DMPlexComputeBdJacobianSingle()`` to take a form key
-  Add ``DMPlexTopologyLoad()``, ``DMPlexCoordinatesLoad()``, and
   ``DMPlexLabelsLoad()`` for incremental loading of a ``DMPlex`` object
   from an HDF5 file
-  Add ``DMPlexTopologyView()``, ``DMPlexCoordinatesView()``, and
   ``DMPlexLabelsView()`` for incremental saving of a ``DMPlex`` object
   to an HDF5 file
-  Add ``DMPlexSectionView()`` saving a ``PetscSection`` in
   association with a ``DMPlex`` mesh
-  Add ``DMPlexSectionLoad()`` loading a ``PetscSection`` in
   association with a ``DMPlex`` mesh
-  Add ``DMPlexGlobalVectorView()`` and ``DMPlexLocalVectorView()`` saving
   global and local vectors in association with a data layout on a ``DMPlex`` mesh
-  Add ``DMPlexGlobalVectorLoad()`` and ``DMPlexLocalVectorLoad()`` loading
   global and local vectors in association with a data layout on a ``DMPlex`` mesh
- Add ``DMPlexIsSimplex()`` to check the shape of the first cell
- Add ``DMPlexShape`` to describe prebuilt mesh domains
- Add ``DMPlexCreateCoordinateSpace()`` to make an FE space for the coordinates
- Add the automatic creation of a Plex from options, see ``DMSetFromOptions()``
- The old options for ``DMPlexCreateBoxMesh()`` NO LONGER WORK. They have been changed to make the interface more uniform
- Replace ``DMPlexCreateSquareBoundary()`` and ``DMPlexCreateCubeBoundary()`` with ``DMPlexCreateBoxSurfaceMesh()``
- Remove ``DMPlexCreateReferenceCellByType()``
- The number of refinements is no longer an argument to ``DMPlexCreateHexCylinderMesh()``
- Add ``DMSetLabel()``
- Replace ``DMPlexComputeJacobianAction()`` with ``DMSNESComputeJacobianAction()``
- Add ``DMSNESCreateJacobianMF()``
- Change ``DMPlexComputeBdResidualSingle()`` to take ``PetscFormKey`` instead of explicit label/value/field arguments

.. rubric:: FE/FV:

-  Change ``PetscFEIntegrateBdResidual()`` and
   ``PetscFEIntegrateBdJacobian()`` to take both ``PetscWeakForm`` and form
   key
- Add ``PetscFEGeomGetPoint()`` and ``PetscFEGeomGetCellPoint`` to package up geometry handling

.. rubric:: DMNetwork:

-  Add ``DMNetworkCreateIS()`` and ``DMNetworkCreateLocalIS()``

.. rubric:: DMStag:

-  Add ``DMStagStencilToIndexLocal()``

.. rubric:: DT:

-  Add ``PetscWeakFormCopy()``, ``PetscWeakFormClear()``, ``PetscWeakFormRewriteKeys()`` and ``PetscWeakFormClearIndex()``
-  Add ``PetscDSDestroyBoundary()`` and ``PetscDSCopyExactSolutions()``
-  ``PetscDSGetContext()`` now takes ``void*`` as return argument

.. rubric:: Fortran:

-  Add support for ``PetscInitialize(filename,help,ierr)``,
   ``PetscInitialize(ierr)`` in addition to current ``PetscInitialize(filename,ierr)``
