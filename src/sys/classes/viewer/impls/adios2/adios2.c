#include <petsc/private/viewerimpl.h>    /*I   "petscsys.h"   I*/
#include <adios2_c.h>
#include <petsc/private/vieweradios2impl.h>

static PetscErrorCode PetscViewerSetFromOptions_ADIOS2(PetscOptionItems *PetscOptionsObject,PetscViewer v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"ADIOS2 PetscViewer Options");CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerFileClose_ADIOS2(PetscViewer viewer)
{
  PetscViewer_ADIOS2 *adios2 = (PetscViewer_ADIOS2*)viewer->data;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  switch (adios2->btype) {
  case FILE_MODE_READ:
    /* ierr = adios2_read_close(adios2->adios2_fp);CHKERRQ(ierr); */
    break;
  case FILE_MODE_APPEND:
    break;
  case FILE_MODE_WRITE:
    /* ierr = adios2_close(adios2->adios2_handle);CHKERRQ(ierr); */
    break;
  default:
    break;
  }
  ierr = PetscFree(adios2->filename);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscViewerDestroy_ADIOS2(PetscViewer viewer)
{
  PetscViewer_ADIOS2 *adios2 = (PetscViewer_ADIOS2*) viewer->data;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscViewerFileClose_ADIOS2(viewer);CHKERRQ(ierr);
  ierr = PetscFree(adios2);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)viewer,"PetscViewerFileSetName_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)viewer,"PetscViewerFileGetName_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)viewer,"PetscViewerFileSetMode_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode  PetscViewerFileSetMode_ADIOS2(PetscViewer viewer, PetscFileMode type)
{
  PetscViewer_ADIOS2 *adios2 = (PetscViewer_ADIOS2*) viewer->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  adios2->btype = type;
  PetscFunctionReturn(0);
}

PetscErrorCode  PetscViewerFileSetName_ADIOS2(PetscViewer viewer, const char name[])
{
  PetscViewer_ADIOS2 *adios2 = (PetscViewer_ADIOS2*) viewer->data;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscFree(adios2->filename);CHKERRQ(ierr);
  ierr = PetscStrallocpy(name, &adios2->filename);CHKERRQ(ierr);
  /* Create or open the file collectively */
  switch (adios2->btype) {
  case FILE_MODE_READ:
    /* adios2->adios2_fp = adios2_read_open_file(adios2->filename,ADIOS2_READ_METHOD_BP,PetscObjectComm((PetscObject)viewer)); */
    break;
  case FILE_MODE_APPEND:
    break;
  case FILE_MODE_WRITE:
    /* adios2_open(&adios2->adios2_handle,"PETSc",adios2->filename,"w",PetscObjectComm((PetscObject)viewer)); */
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER, "Must call PetscViewerFileSetMode() before PetscViewerFileSetName()");
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerFileGetName_ADIOS2(PetscViewer viewer,const char **name)
{
  PetscViewer_ADIOS2 *vadios2 = (PetscViewer_ADIOS2*)viewer->data;

  PetscFunctionBegin;
  *name = vadios2->filename;
  PetscFunctionReturn(0);
}

/*MC
   PETSCVIEWERADIOS2 - A viewer that writes to an ADIOS2 file


.seealso:  PetscViewerADIOS2Open(), PetscViewerStringSPrintf(), PetscViewerSocketOpen(), PetscViewerDrawOpen(), PETSCVIEWERSOCKET,
           PetscViewerCreate(), PetscViewerASCIIOpen(), PetscViewerBinaryOpen(), PETSCVIEWERBINARY, PETSCVIEWERDRAW, PETSCVIEWERSTRING,
           PetscViewerMatlabOpen(), VecView(), DMView(), PetscViewerMatlabPutArray(), PETSCVIEWERASCII, PETSCVIEWERMATLAB,
           PetscViewerFileSetName(), PetscViewerFileSetMode(), PetscViewerFormat, PetscViewerType, PetscViewerSetType()

  Level: beginner
M*/

PETSC_EXTERN PetscErrorCode PetscViewerCreate_ADIOS2(PetscViewer v)
{
  PetscViewer_ADIOS2 *adios2;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(v,&adios2);CHKERRQ(ierr);

  v->data                = (void*) adios2;
  v->ops->destroy        = PetscViewerDestroy_ADIOS2;
  v->ops->setfromoptions = PetscViewerSetFromOptions_ADIOS2;
  v->ops->flush          = 0;
  adios2->btype            = (PetscFileMode) -1;
  adios2->filename         = 0;
  adios2->timestep         = -1;

  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerFileSetName_C",PetscViewerFileSetName_ADIOS2);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerFileGetName_C",PetscViewerFileGetName_ADIOS2);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerFileSetMode_C",PetscViewerFileSetMode_ADIOS2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscViewerADIOS2Open - Opens a file for ADIOS2 input/output.

   Collective

   Input Parameters:
+  comm - MPI communicator
.  name - name of file
-  type - type of file
$    FILE_MODE_WRITE - create new file for binary output
$    FILE_MODE_READ - open existing file for binary input
$    FILE_MODE_APPEND - open existing file for binary output

   Output Parameter:
.  adios2v - PetscViewer for ADIOS2 input/output to use with the specified file

   Level: beginner

   Note:
   This PetscViewer should be destroyed with PetscViewerDestroy().


.seealso: PetscViewerASCIIOpen(), PetscViewerPushFormat(), PetscViewerDestroy(), PetscViewerHDF5Open(),
          VecView(), MatView(), VecLoad(), PetscViewerSetType(), PetscViewerFileSetMode(), PetscViewerFileSetName()
          MatLoad(), PetscFileMode, PetscViewer
@*/
PetscErrorCode  PetscViewerADIOS2Open(MPI_Comm comm, const char name[], PetscFileMode type, PetscViewer *adios2v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerCreate(comm, adios2v);CHKERRQ(ierr);
  ierr = PetscViewerSetType(*adios2v, PETSCVIEWERADIOS2);CHKERRQ(ierr);
  ierr = PetscViewerFileSetMode(*adios2v, type);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(*adios2v, name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
