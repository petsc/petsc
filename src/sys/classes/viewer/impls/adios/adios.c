#include <petsc/private/viewerimpl.h>    /*I   "petscsys.h"   I*/
#include <adios.h>
#include <adios_read.h>

#include <petsc/private/vieweradiosimpl.h>

static PetscErrorCode PetscViewerSetFromOptions_ADIOS(PetscOptionItems *PetscOptionsObject,PetscViewer v)
{
  PetscFunctionBegin;
  PetscCall(PetscOptionsHead(PetscOptionsObject,"ADIOS PetscViewer Options"));
  PetscCall(PetscOptionsTail());
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerFileClose_ADIOS(PetscViewer viewer)
{
  PetscViewer_ADIOS *adios = (PetscViewer_ADIOS*)viewer->data;

  PetscFunctionBegin;
  switch (adios->btype) {
  case FILE_MODE_READ:
    PetscCall(adios_read_close(adios->adios_fp));
    break;
  case FILE_MODE_WRITE:
     PetscCall(adios_close(adios->adios_handle));
    break;
  default:
    break;
  }
  PetscCall(PetscFree(adios->filename));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscViewerDestroy_ADIOS(PetscViewer viewer)
{
  PetscViewer_ADIOS *adios = (PetscViewer_ADIOS*) viewer->data;

  PetscFunctionBegin;
  PetscCall(PetscViewerFileClose_ADIOS(viewer));
  PetscCall(PetscFree(adios));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer,"PetscViewerFileSetName_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer,"PetscViewerFileGetName_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer,"PetscViewerFileSetMode_C",NULL));
  PetscFunctionReturn(0);
}

PetscErrorCode  PetscViewerFileSetMode_ADIOS(PetscViewer viewer, PetscFileMode type)
{
  PetscViewer_ADIOS *adios = (PetscViewer_ADIOS*) viewer->data;

  PetscFunctionBegin;
  adios->btype = type;
  PetscFunctionReturn(0);
}

PetscErrorCode  PetscViewerFileSetName_ADIOS(PetscViewer viewer, const char name[])
{
  PetscViewer_ADIOS *adios = (PetscViewer_ADIOS*) viewer->data;

  PetscFunctionBegin;
  if (adios->filename) PetscCall(PetscFree(adios->filename));
  PetscCall(PetscStrallocpy(name, &adios->filename));
  /* Create or open the file collectively */
  switch (adios->btype) {
  case FILE_MODE_READ:
    adios->adios_fp = adios_read_open_file(adios->filename,ADIOS_READ_METHOD_BP,PetscObjectComm((PetscObject)viewer));
    break;
  case FILE_MODE_WRITE:
    adios_open(&adios->adios_handle,"PETSc",adios->filename,"w",PetscObjectComm((PetscObject)viewer));
    break;
  case FILE_MODE_UNDEFINED:
    SETERRQ(PetscObjectComm((PetscObject)viewer),PETSC_ERR_ORDER,"Must call PetscViewerFileSetMode() before PetscViewerFileSetName()");
  default:
    SETERRQ(PetscObjectComm((PetscObject)viewer),PETSC_ERR_SUP,"Unsupported file mode %s",PetscFileModes[adios->btype]);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerFileGetName_ADIOS(PetscViewer viewer,const char **name)
{
  PetscViewer_ADIOS *vadios = (PetscViewer_ADIOS*)viewer->data;

  PetscFunctionBegin;
  *name = vadios->filename;
  PetscFunctionReturn(0);
}

/*MC
   PETSCVIEWERADIOS - A viewer that writes to an ADIOS file

.seealso:  PetscViewerADIOSOpen(), PetscViewerStringSPrintf(), PetscViewerSocketOpen(), PetscViewerDrawOpen(), PETSCVIEWERSOCKET,
           PetscViewerCreate(), PetscViewerASCIIOpen(), PetscViewerBinaryOpen(), PETSCVIEWERBINARY, PETSCVIEWERDRAW, PETSCVIEWERSTRING,
           PetscViewerMatlabOpen(), VecView(), DMView(), PetscViewerMatlabPutArray(), PETSCVIEWERASCII, PETSCVIEWERMATLAB,
           PetscViewerFileSetName(), PetscViewerFileSetMode(), PetscViewerFormat, PetscViewerType, PetscViewerSetType()

  Level: beginner
M*/

PETSC_EXTERN PetscErrorCode PetscViewerCreate_ADIOS(PetscViewer v)
{
  PetscViewer_ADIOS *adios;

  PetscFunctionBegin;
  PetscCall(PetscNewLog(v,&adios));

  v->data                = (void*) adios;
  v->ops->destroy        = PetscViewerDestroy_ADIOS;
  v->ops->setfromoptions = PetscViewerSetFromOptions_ADIOS;
  v->ops->flush          = NULL;
  adios->btype           = FILE_MODE_UNDEFINED;
  adios->filename        = NULL;
  adios->timestep        = -1;

  PetscCall(PetscObjectComposeFunction((PetscObject)v,"PetscViewerFileSetName_C",PetscViewerFileSetName_ADIOS));
  PetscCall(PetscObjectComposeFunction((PetscObject)v,"PetscViewerFileGetName_C",PetscViewerFileGetName_ADIOS));
  PetscCall(PetscObjectComposeFunction((PetscObject)v,"PetscViewerFileSetMode_C",PetscViewerFileSetMode_ADIOS));
  PetscFunctionReturn(0);
}

/*@C
   PetscViewerADIOSOpen - Opens a file for ADIOS input/output.

   Collective

   Input Parameters:
+  comm - MPI communicator
.  name - name of file
-  type - type of file
$    FILE_MODE_WRITE - create new file for binary output
$    FILE_MODE_READ - open existing file for binary input
$    FILE_MODE_APPEND - open existing file for binary output

   Output Parameter:
.  adiosv - PetscViewer for ADIOS input/output to use with the specified file

   Level: beginner

   Note:
   This PetscViewer should be destroyed with PetscViewerDestroy().

.seealso: PetscViewerASCIIOpen(), PetscViewerPushFormat(), PetscViewerDestroy(), PetscViewerHDF5Open(),
          VecView(), MatView(), VecLoad(), PetscViewerSetType(), PetscViewerFileSetMode(), PetscViewerFileSetName()
          MatLoad(), PetscFileMode, PetscViewer
@*/
PetscErrorCode  PetscViewerADIOSOpen(MPI_Comm comm, const char name[], PetscFileMode type, PetscViewer *adiosv)
{
  PetscFunctionBegin;
  PetscCall(PetscViewerCreate(comm, adiosv));
  PetscCall(PetscViewerSetType(*adiosv, PETSCVIEWERADIOS));
  PetscCall(PetscViewerFileSetMode(*adiosv, type));
  PetscCall(PetscViewerFileSetName(*adiosv, name));
  PetscFunctionReturn(0);
}

/*@C
  PetscDataTypeToADIOSDataType - Converts the PETSc name of a datatype to its ADIOS name.

  Not collective

  Input Parameter:
. ptype - the PETSc datatype name (for example PETSC_DOUBLE)

  Output Parameter:
. mtype - the MPI datatype (for example MPI_DOUBLE, ...)

  Level: advanced

  Developer Notes: These have not been verified

.seealso: PetscDataType, PetscADIOSDataTypeToPetscDataType()
@*/
PetscErrorCode PetscDataTypeToADIOSDataType(PetscDataType ptype, enum ADIOS_DATATYPES *htype)
{
  PetscFunctionBegin;
  if (ptype == PETSC_INT)
#if defined(PETSC_USE_64BIT_INDICES)
                                       *htype = adios_long;
#else
                                       *htype = adios_integer;
#endif
  else if (ptype == PETSC_ENUM)        *htype = adios_integer;
  else if (ptype == PETSC_DOUBLE)      *htype = adios_double;
  else if (ptype == PETSC_LONG)        *htype = adios_long;
  else if (ptype == PETSC_SHORT)       *htype = adios_short;
  else if (ptype == PETSC_FLOAT)       *htype = adios_real;
  else if (ptype == PETSC_CHAR)        *htype = adios_string_array;
  else if (ptype == PETSC_STRING)      *htype = adios_string;
  else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Unsupported PETSc datatype");
  PetscFunctionReturn(0);
}

/*@C
  PetscADIOSDataTypeToPetscDataType - Finds the PETSc name of a datatype from its ADIOS name

  Not collective

  Input Parameter:
. htype - the ADIOS datatype (for example H5T_NATIVE_DOUBLE, ...)

  Output Parameter:
. ptype - the PETSc datatype name (for example PETSC_DOUBLE)

  Level: advanced

  Developer Notes: These have not been verified

.seealso: PetscDataType, PetscADIOSDataTypeToPetscDataType()
@*/
PetscErrorCode PetscADIOSDataTypeToPetscDataType(enum ADIOS_DATATYPES htype, PetscDataType *ptype)
{
  PetscFunctionBegin;
#if defined(PETSC_USE_64BIT_INDICES)
  if      (htype == adios_integer)     *ptype = PETSC_ENUM;
  else if (htype == adios_long)        *ptype = PETSC_INT;
#else
  if      (htype == adios_integer)     *ptype = PETSC_INT;
#endif
  else if (htype == adios_double)      *ptype = PETSC_DOUBLE;
  else if (htype == adios_long)        *ptype = PETSC_LONG;
  else if (htype == adios_short)       *ptype = PETSC_SHORT;
  else if (htype == adios_real)        *ptype = PETSC_FLOAT;
  else if (htype == adios_string_array) *ptype = PETSC_CHAR;
  else if (htype == adios_string)       *ptype = PETSC_STRING;
  else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Unsupported ADIOS datatype");
  PetscFunctionReturn(0);
}
