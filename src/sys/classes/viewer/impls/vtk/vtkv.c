#include <../src/sys/classes/viewer/impls/vtk/vtkvimpl.h> /*I "petscviewer.h" I*/

/*MC
    PetscViewerVTKWriteFunction - functional form used to provide a writer to the `PETSCVIEWERVTK`

     Synopsis:
     #include <petscviewer.h>
     PetscViewerVTKWriteFunction(PetscObject object,PetscViewer viewer)

     Input Parameters:
+      object - the PETSc object to be written
-      viewer - viewer it is to be written to

   Level: developer

.seealso: [](sec_viewers), `PETSCVIEWERVTK`, `PetscViewerVTKAddField()`
M*/

/*@C
   PetscViewerVTKAddField - Add a field to the viewer

   Collective

   Input Parameters:
+  viewer - `PETSCVIEWERVTK`
.  dm - `DM` on which `Vec` lives
.  PetscViewerVTKWriteFunction - function to write this `Vec`
.  fieldnum - which field of the `DM` to write (`PETSC_DEFAULT` if the while vector should be written)
.  fieldtype - Either `PETSC_VTK_POINT_FIELD` or `PETSC_VTK_CELL_FIELD`
.  checkdm - whether to check for identical dm arguments as fields are added
-  vec - `Vec` from which to write

   Level: developer

   Note:
   This routine keeps exclusive ownership of the `Vec`. The caller should not use or destroy the `Vec` after calling it.

.seealso: [](sec_viewers), `PETSCVIEWERVTK`, `PetscViewerVTKOpen()`, `DMDAVTKWriteAll()`, `PetscViewerVTKWriteFunction`, `PetscViewerVTKGetDM()`
@*/
PetscErrorCode PetscViewerVTKAddField(PetscViewer viewer, PetscObject dm, PetscErrorCode (*PetscViewerVTKWriteFunction)(PetscObject, PetscViewer), PetscInt fieldnum, PetscViewerVTKFieldType fieldtype, PetscBool checkdm, PetscObject vec)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscValidHeader(dm, 2);
  PetscValidHeader(vec, 7);
  PetscUseMethod(viewer, "PetscViewerVTKAddField_C", (PetscViewer, PetscObject, PetscErrorCode(*)(PetscObject, PetscViewer), PetscInt, PetscViewerVTKFieldType, PetscBool, PetscObject), (viewer, dm, PetscViewerVTKWriteFunction, fieldnum, fieldtype, checkdm, vec));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscViewerVTKGetDM - get the `DM` associated with the `PETSCVIEWERVTK` viewer

   Collective

   Input Parameters:
+  viewer - `PETSCVIEWERVTK` viewer
-  dm - `DM` associated with the viewer (as a `PetscObject`)

   Level: developer

.seealso: [](sec_viewers), `PETSCVIEWERVTK`, `PetscViewerVTKOpen()`, `DMDAVTKWriteAll()`, `PetscViewerVTKWriteFunction`, `PetscViewerVTKAddField()`
@*/
PetscErrorCode PetscViewerVTKGetDM(PetscViewer viewer, PetscObject *dm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscUseMethod(viewer, "PetscViewerVTKGetDM_C", (PetscViewer, PetscObject *), (viewer, dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerDestroy_VTK(PetscViewer viewer)
{
  PetscViewer_VTK *vtk = (PetscViewer_VTK *)viewer->data;

  PetscFunctionBegin;
  PetscCall(PetscFree(vtk->filename));
  PetscCall(PetscFree(vtk));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerFileSetName_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerFileGetName_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerFileSetMode_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerFileGetMode_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerVTKAddField_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerVTKGetDM_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerFlush_VTK(PetscViewer viewer)
{
  PetscViewer_VTK         *vtk = (PetscViewer_VTK *)viewer->data;
  PetscViewerVTKObjectLink link, next;

  PetscFunctionBegin;
  PetscCheck(!vtk->link || !(!vtk->dm || !vtk->write), PetscObjectComm((PetscObject)viewer), PETSC_ERR_ARG_WRONGSTATE, "No fields or no grid");
  if (vtk->write) PetscCall((*vtk->write)(vtk->dm, viewer));
  for (link = vtk->link; link; link = next) {
    next = link->next;
    PetscCall(PetscObjectDestroy(&link->vec));
    PetscCall(PetscFree(link));
  }
  PetscCall(PetscObjectDestroy(&vtk->dm));
  vtk->write = NULL;
  vtk->link  = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscViewerFileSetName_VTK(PetscViewer viewer, const char name[])
{
  PetscViewer_VTK *vtk = (PetscViewer_VTK *)viewer->data;
  PetscBool        isvtk, isvts, isvtu, isvtr;
  size_t           len;

  PetscFunctionBegin;
  PetscCall(PetscViewerFlush(viewer));
  PetscCall(PetscFree(vtk->filename));
  PetscCall(PetscStrlen(name, &len));
  if (!len) {
    isvtk = PETSC_TRUE;
  } else {
    PetscCall(PetscStrcasecmp(name + len - 4, ".vtk", &isvtk));
    PetscCall(PetscStrcasecmp(name + len - 4, ".vts", &isvts));
    PetscCall(PetscStrcasecmp(name + len - 4, ".vtu", &isvtu));
    PetscCall(PetscStrcasecmp(name + len - 4, ".vtr", &isvtr));
  }
  if (isvtk) {
    if (viewer->format == PETSC_VIEWER_DEFAULT) viewer->format = PETSC_VIEWER_ASCII_VTK_DEPRECATED;
    PetscCheck(viewer->format == PETSC_VIEWER_ASCII_VTK_DEPRECATED, PetscObjectComm((PetscObject)viewer), PETSC_ERR_ARG_INCOMP, "Cannot use file '%s' with format %s, should have '.vtk' extension", name, PetscViewerFormats[viewer->format]);
  } else if (isvts) {
    if (viewer->format == PETSC_VIEWER_DEFAULT) viewer->format = PETSC_VIEWER_VTK_VTS;
    PetscCheck(viewer->format == PETSC_VIEWER_VTK_VTS, PetscObjectComm((PetscObject)viewer), PETSC_ERR_ARG_INCOMP, "Cannot use file '%s' with format %s, should have '.vts' extension", name, PetscViewerFormats[viewer->format]);
  } else if (isvtu) {
    if (viewer->format == PETSC_VIEWER_DEFAULT) viewer->format = PETSC_VIEWER_VTK_VTU;
    PetscCheck(viewer->format == PETSC_VIEWER_VTK_VTU, PetscObjectComm((PetscObject)viewer), PETSC_ERR_ARG_INCOMP, "Cannot use file '%s' with format %s, should have '.vtu' extension", name, PetscViewerFormats[viewer->format]);
  } else if (isvtr) {
    if (viewer->format == PETSC_VIEWER_DEFAULT) viewer->format = PETSC_VIEWER_VTK_VTR;
    PetscCheck(viewer->format == PETSC_VIEWER_VTK_VTR, PetscObjectComm((PetscObject)viewer), PETSC_ERR_ARG_INCOMP, "Cannot use file '%s' with format %s, should have '.vtr' extension", name, PetscViewerFormats[viewer->format]);
  } else SETERRQ(PetscObjectComm((PetscObject)viewer), PETSC_ERR_ARG_UNKNOWN_TYPE, "File '%s' has unrecognized extension", name);
  PetscCall(PetscStrallocpy(len ? name : "stdout", &vtk->filename));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscViewerFileGetName_VTK(PetscViewer viewer, const char **name)
{
  PetscViewer_VTK *vtk = (PetscViewer_VTK *)viewer->data;
  PetscFunctionBegin;
  *name = vtk->filename;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscViewerFileSetMode_VTK(PetscViewer viewer, PetscFileMode type)
{
  PetscViewer_VTK *vtk = (PetscViewer_VTK *)viewer->data;

  PetscFunctionBegin;
  vtk->btype = type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscViewerFileGetMode_VTK(PetscViewer viewer, PetscFileMode *type)
{
  PetscViewer_VTK *vtk = (PetscViewer_VTK *)viewer->data;

  PetscFunctionBegin;
  *type = vtk->btype;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscViewerVTKAddField_VTK(PetscViewer viewer, PetscObject dm, PetscErrorCode (*PetscViewerVTKWriteFunction)(PetscObject, PetscViewer), PetscInt fieldnum, PetscViewerVTKFieldType fieldtype, PetscBool checkdm, PetscObject vec)
{
  PetscViewer_VTK         *vtk = (PetscViewer_VTK *)viewer->data;
  PetscViewerVTKObjectLink link, tail = vtk->link;

  PetscFunctionBegin;
  if (vtk->dm) {
    PetscCheck(!checkdm || dm == vtk->dm, PetscObjectComm((PetscObject)viewer), PETSC_ERR_ARG_INCOMP, "Refusing to write a field from more than one grid to the same VTK file. Set checkdm = PETSC_FALSE to skip this check.");
  } else {
    PetscCall(PetscObjectReference(dm));
    vtk->dm = dm;
  }
  vtk->write = PetscViewerVTKWriteFunction;
  PetscCall(PetscNew(&link));
  link->ft    = fieldtype;
  link->vec   = vec;
  link->field = fieldnum;
  link->next  = NULL;
  /* Append to list */
  if (tail) {
    while (tail->next) tail = tail->next;
    tail->next = link;
  } else vtk->link = link;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscViewerVTKGetDM_VTK(PetscViewer viewer, PetscObject *dm)
{
  PetscViewer_VTK *vtk = (PetscViewer_VTK *)viewer->data;

  PetscFunctionBegin;
  *dm = vtk->dm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   PETSCVIEWERVTK - A viewer that writes to a VTK file

  Level: beginner

.seealso: [](sec_viewers), `PetscViewerVTKOpen()`, `PetscViewerHDF5Open()`, `PetscViewerStringSPrintf()`, `PetscViewerSocketOpen()`, `PetscViewerDrawOpen()`, `PETSCVIEWERSOCKET`,
          `PetscViewerCreate()`, `PetscViewerASCIIOpen()`, `PetscViewerBinaryOpen()`, `PETSCVIEWERBINARY`, `PETSCVIEWERDRAW`, `PETSCVIEWERSTRING`,
          `PetscViewerMatlabOpen()`, `VecView()`, `DMView()`, `PetscViewerMatlabPutArray()`, `PETSCVIEWERASCII`, `PETSCVIEWERMATLAB`,
          `PetscViewerFileSetName()`, `PetscViewerFileSetMode()`, `PetscViewerFormat`, `PetscViewerType`, `PetscViewerSetType()`
M*/

PETSC_EXTERN PetscErrorCode PetscViewerCreate_VTK(PetscViewer v)
{
  PetscViewer_VTK *vtk;

  PetscFunctionBegin;
  PetscCall(PetscNew(&vtk));

  v->data         = (void *)vtk;
  v->ops->destroy = PetscViewerDestroy_VTK;
  v->ops->flush   = PetscViewerFlush_VTK;
  vtk->btype      = FILE_MODE_UNDEFINED;
  vtk->filename   = NULL;

  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerFileSetName_C", PetscViewerFileSetName_VTK));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerFileGetName_C", PetscViewerFileGetName_VTK));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerFileSetMode_C", PetscViewerFileSetMode_VTK));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerFileGetMode_C", PetscViewerFileGetMode_VTK));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerVTKAddField_C", PetscViewerVTKAddField_VTK));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerVTKGetDM_C", PetscViewerVTKGetDM_VTK));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscViewerVTKOpen - Opens a `PETSCVIEWERVTK` viewer file.

   Collective

   Input Parameters:
+  comm - MPI communicator
.  name - name of file
-  type - type of file
.vb
    FILE_MODE_WRITE - create new file for binary output
    FILE_MODE_READ - open existing file for binary input (not currently supported)
    FILE_MODE_APPEND - open existing file for binary output (not currently supported)
.ve

   Output Parameter:
.  vtk - `PetscViewer` for VTK input/output to use with the specified file

   Level: beginner

.seealso: [](sec_viewers), `PETSCVIEWERVTK`, `PetscViewerASCIIOpen()`, `PetscViewerPushFormat()`, `PetscViewerDestroy()`,
          `VecView()`, `MatView()`, `VecLoad()`, `MatLoad()`,
          `PetscFileMode`, `PetscViewer`
@*/
PetscErrorCode PetscViewerVTKOpen(MPI_Comm comm, const char name[], PetscFileMode type, PetscViewer *vtk)
{
  PetscFunctionBegin;
  PetscCall(PetscViewerCreate(comm, vtk));
  PetscCall(PetscViewerSetType(*vtk, PETSCVIEWERVTK));
  PetscCall(PetscViewerFileSetMode(*vtk, type));
  PetscCall(PetscViewerFileSetName(*vtk, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscViewerVTKFWrite - write binary data preceded by 32-bit int length (in bytes), does not do byte swapping.

   Logically Collective

   Input Parameters:
+  viewer - logically collective viewer, data written from rank 0
.  fp - file pointer valid on rank 0
.  data - data pointer valid on rank 0
.  n - number of data items
-  dtype - data type

   Level: developer

   Note:
    If `PetscScalar` is `__float128` then the binary files are written in double precision

.seealso: [](sec_viewers), `PETSCVIEWERVTK`, `DMDAVTKWriteAll()`, `DMPlexVTKWriteAll()`, `PetscViewerPushFormat()`, `PetscViewerVTKOpen()`, `PetscBinaryWrite()`
@*/
PetscErrorCode PetscViewerVTKFWrite(PetscViewer viewer, FILE *fp, const void *data, PetscInt n, MPI_Datatype dtype)
{
  PetscMPIInt  rank;
  MPI_Datatype vdtype = dtype;
#if defined(PETSC_USE_REAL___FLOAT128)
  double    *tmp;
  PetscInt   i;
  PetscReal *ttmp = (PetscReal *)data;
#endif

  PetscFunctionBegin;
  PetscCheck(n >= 0, PetscObjectComm((PetscObject)viewer), PETSC_ERR_ARG_OUTOFRANGE, "Trying to write a negative amount of data %" PetscInt_FMT, n);
  if (!n) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)viewer), &rank));
  if (rank == 0) {
    size_t      count;
    PetscMPIInt dsize;
    PetscInt64  bytes;

#if defined(PETSC_USE_REAL___FLOAT128)
    if (dtype == MPIU___FLOAT128) {
      PetscCall(PetscMalloc1(n, &tmp));
      for (i = 0; i < n; i++) tmp[i] = ttmp[i];
      data   = (void *)tmp;
      vdtype = MPI_DOUBLE;
    }
#endif
    PetscCallMPI(MPI_Type_size(vdtype, &dsize));
    bytes = (PetscInt64)dsize * n;

    count = fwrite(&bytes, sizeof(bytes), 1, fp);
    PetscCheck(count == 1, PETSC_COMM_SELF, PETSC_ERR_FILE_WRITE, "Error writing byte count");
    count = fwrite(data, dsize, (size_t)n, fp);
    PetscCheck((PetscInt)count == n, PETSC_COMM_SELF, PETSC_ERR_FILE_WRITE, "Wrote %" PetscInt_FMT "/%" PetscInt_FMT " array members of size %d", (PetscInt)count, n, dsize);
#if defined(PETSC_USE_REAL___FLOAT128)
    if (dtype == MPIU___FLOAT128) PetscCall(PetscFree(tmp));
#endif
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
