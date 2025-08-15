#define PETSCDM_DLL
#include <petsc/private/dmpleximpl.h> /*I   "petscdmplex.h"   I*/
#include <petsc/private/viewerimpl.h>

/*@
  PetscViewerExodusIIGetId - Get the file id of the `PETSCVIEWEREXODUSII` file

  Logically Collective

  Input Parameter:
. viewer - the `PetscViewer`

  Output Parameter:
. exoid - The ExodusII file id

  Level: intermediate

.seealso: `PETSCVIEWEREXODUSII`, `PetscViewer`, `PetscViewerFileSetMode()`, `PetscViewerCreate()`, `PetscViewerSetType()`, `PetscViewerBinaryOpen()`
@*/
PetscErrorCode PetscViewerExodusIIGetId(PetscViewer viewer, int *exoid)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscTryMethod(viewer, "PetscViewerGetId_C", (PetscViewer, int *), (viewer, exoid));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscViewerExodusIISetOrder - Set the elements order in the ExodusII file.

  Collective

  Input Parameters:
+ viewer - the `PETSCVIEWEREXODUSII` viewer
- order  - elements order

  Output Parameter:

  Level: beginner

.seealso: `PETSCVIEWEREXODUSII`, `PetscViewer`, `PetscViewerExodusIIGetId()`, `PetscViewerExodusIIGetOrder()`
@*/
PetscErrorCode PetscViewerExodusIISetOrder(PetscViewer viewer, PetscInt order)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscTryMethod(viewer, "PetscViewerSetOrder_C", (PetscViewer, PetscInt), (viewer, order));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscViewerExodusIIGetOrder - Get the elements order in the ExodusII file.

  Collective

  Input Parameters:
+ viewer - the `PETSCVIEWEREXODUSII` viewer
- order  - elements order

  Output Parameter:

  Level: beginner

.seealso: `PETSCVIEWEREXODUSII`, `PetscViewer`, `PetscViewerExodusIIGetId()`, `PetscViewerExodusIISetOrder()`
@*/
PetscErrorCode PetscViewerExodusIIGetOrder(PetscViewer viewer, PetscInt *order)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscTryMethod(viewer, "PetscViewerGetOrder_C", (PetscViewer, PetscInt *), (viewer, order));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscViewerExodusIIOpen - Opens a file for ExodusII input/output.

  Collective

  Input Parameters:
+ comm - MPI communicator
. name - name of file
- type - type of file
.vb
    FILE_MODE_WRITE - create new file for binary output
    FILE_MODE_READ - open existing file for binary input
    FILE_MODE_APPEND - open existing file for binary output
.ve

  Output Parameter:
. exo - `PETSCVIEWEREXODUSII` `PetscViewer` for Exodus II input/output to use with the specified file

  Level: beginner

.seealso: `PETSCVIEWEREXODUSII`, `PetscViewer`, `PetscViewerPushFormat()`, `PetscViewerDestroy()`,
          `DMLoad()`, `PetscFileMode`, `PetscViewerSetType()`, `PetscViewerFileSetMode()`, `PetscViewerFileSetName()`
@*/
PetscErrorCode PetscViewerExodusIIOpen(MPI_Comm comm, const char name[], PetscFileMode type, PetscViewer *exo)
{
  PetscFunctionBegin;
  PetscCall(PetscViewerCreate(comm, exo));
  PetscCall(PetscViewerSetType(*exo, PETSCVIEWEREXODUSII));
  PetscCall(PetscViewerFileSetMode(*exo, type));
  PetscCall(PetscViewerFileSetName(*exo, name));
  PetscCall(PetscViewerSetFromOptions(*exo));
  PetscFunctionReturn(PETSC_SUCCESS);
}

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
