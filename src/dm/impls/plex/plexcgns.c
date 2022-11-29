#define PETSCDM_DLL
#include <petsc/private/dmpleximpl.h> /*I   "petscdmplex.h"   I*/

/*@C
  DMPlexCreateCGNS - Create a `DMPLEX` mesh from a CGNS file.

  Collective

  Input Parameters:
+ comm  - The MPI communicator
. filename - The name of the CGNS file
- interpolate - Create faces and edges in the mesh

  Output Parameter:
. dm  - The `DM` object representing the mesh

  Level: beginner

  Note:
  https://cgns.github.io

.seealso: [](chapter_unstructured), `DM`, `DMPLEX`, `DMPlexCreate()`, `DMPlexCreateCGNS()`, `DMPlexCreateExodus()`
@*/
PetscErrorCode DMPlexCreateCGNSFromFile(MPI_Comm comm, const char filename[], PetscBool interpolate, DM *dm)
{
  PetscFunctionBegin;
  PetscValidCharPointer(filename, 2);
#if defined(PETSC_HAVE_CGNS)
  PetscCall(DMPlexCreateCGNSFromFile_Internal(comm, filename, interpolate, dm));
#else
  SETERRQ(comm, PETSC_ERR_SUP, "Loading meshes requires CGNS support. Reconfigure using --with-cgns-dir");
#endif
  PetscFunctionReturn(0);
}

/*@
  DMPlexCreateCGNS - Create a `DMPLEX` mesh from a CGNS file ID.

  Collective

  Input Parameters:
+ comm  - The MPI communicator
. cgid - The CG id associated with a file and obtained using cg_open
- interpolate - Create faces and edges in the mesh

  Output Parameter:
. dm  - The `DM` object representing the mesh

  Level: beginner

  Note:
  https://cgns.github.io

.seealso: [](chapter_unstructured), `DM`, `DMPLEX`, `DMPlexCreate()`, `DMPlexCreateExodus()`
@*/
PetscErrorCode DMPlexCreateCGNS(MPI_Comm comm, PetscInt cgid, PetscBool interpolate, DM *dm)
{
  PetscFunctionBegin;
#if defined(PETSC_HAVE_CGNS)
  PetscCall(DMPlexCreateCGNS_Internal(comm, cgid, interpolate, dm));
#else
  SETERRQ(comm, PETSC_ERR_SUP, "Loading meshes requires CGNS support. Reconfigure using --download-cgns");
#endif
  PetscFunctionReturn(0);
}
