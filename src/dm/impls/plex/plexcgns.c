#define PETSCDM_DLL
#include <petsc/private/dmpleximpl.h> /*I   "petscdmplex.h"   I*/

/*@C
  DMPlexCreateCGNSFromFile - Create a `DMPLEX` mesh from a CGNS file.

  Collective

  Input Parameters:
+ comm        - The MPI communicator
. filename    - The name of the CGNS file
- interpolate - Create faces and edges in the mesh

  Output Parameter:
. dm - The `DM` object representing the mesh

  Level: beginner

  Note:
  https://cgns.github.io

.seealso: [](ch_unstructured), `DM`, `DMPLEX`, `DMPlexCreate()`, `DMPlexCreateCGNS()`, `DMPlexCreateExodus()`
@*/
PetscErrorCode DMPlexCreateCGNSFromFile(MPI_Comm comm, const char filename[], PetscBool interpolate, DM *dm)
{
  PetscFunctionBegin;
  PetscAssertPointer(filename, 2);
#if defined(PETSC_HAVE_CGNS)
  PetscCall(DMPlexCreateCGNSFromFile_Internal(comm, filename, interpolate, dm));
#else
  SETERRQ(comm, PETSC_ERR_SUP, "Loading meshes requires CGNS support. Reconfigure using --with-cgns-dir");
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexCreateCGNS - Create a `DMPLEX` mesh from a CGNS file ID.

  Collective

  Input Parameters:
+ comm        - The MPI communicator
. cgid        - The CG id associated with a file and obtained using cg_open
- interpolate - Create faces and edges in the mesh

  Output Parameter:
. dm - The `DM` object representing the mesh

  Level: beginner

  Note:
  https://cgns.github.io

.seealso: [](ch_unstructured), `DM`, `DMPLEX`, `DMPlexCreate()`, `DMPlexCreateExodus()`
@*/
PetscErrorCode DMPlexCreateCGNS(MPI_Comm comm, PetscInt cgid, PetscBool interpolate, DM *dm)
{
  PetscFunctionBegin;
#if defined(PETSC_HAVE_CGNS)
  {
    PetscBool use_parallel_viewer = PETSC_FALSE;

    PetscCall(PetscOptionsGetBool(NULL, NULL, "-dm_plex_cgns_parallel", &use_parallel_viewer, NULL));
    if (use_parallel_viewer) PetscCall(DMPlexCreateCGNS_Internal_Parallel(comm, cgid, interpolate, dm));
    else PetscCall(DMPlexCreateCGNS_Internal_Serial(comm, cgid, interpolate, dm));
  }
#else
  SETERRQ(comm, PETSC_ERR_SUP, "Loading meshes requires CGNS support. Reconfigure using --download-cgns");
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}
