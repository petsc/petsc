#define PETSCDM_DLL
#include <petsc/private/dmpleximpl.h> /*I   "petscdmplex.h"   I*/

/*@
  DMPlexCreateExodusFromFile - Create a `DMPLEX` mesh from an ExodusII file.

  Collective

  Input Parameters:
+ comm        - The MPI communicator
. filename    - The name of the ExodusII file
- interpolate - Create faces and edges in the mesh

  Output Parameter:
. dm - The `DM` object representing the mesh

  Level: beginner

.seealso: [](ch_unstructured), `DM`, `PETSCVIEWEREXODUSII`, `DMPLEX`, `DMCreate()`, `DMPlexCreateExodus()`
@*/
PetscErrorCode DMPlexCreateExodusFromFile(MPI_Comm comm, const char filename[], PetscBool interpolate, DM *dm)
{
  PetscFunctionBegin;
#if defined(PETSC_HAVE_EXODUSII)
  PetscMPIInt      rank;
  PetscExodusIIInt CPU_word_size = sizeof(PetscReal), IO_word_size = 0, exoid = -1;
  float            version;

  PetscAssertPointer(filename, 2);
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  if (rank == 0) {
    exoid = ex_open(filename, EX_READ, &CPU_word_size, &IO_word_size, &version);
    PetscCheck(exoid > 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "ex_open(\"%s\",...) did not return a valid file ID", filename);
  }
  PetscCall(DMPlexCreateExodus(comm, exoid, interpolate, dm));
  if (rank == 0) PetscCallExternal(ex_close, exoid);
  PetscFunctionReturn(PETSC_SUCCESS);
#else
  SETERRQ(comm, PETSC_ERR_SUP, "Loading meshes requires EXODUSII support. Reconfigure using --with-exodusii-dir or -download-exodusii");
#endif
}
