
#include <petsc/private/viewerimpl.h> /*I  "petscsys.h"  I*/

typedef struct {
  char     *string; /* string where info is stored */
  char     *head;   /* pointer to beginning of unused portion */
  size_t    curlen, maxlen;
  PetscBool ownstring; /* string viewer is responsable for freeing the string */
} PetscViewer_String;

static PetscErrorCode PetscViewerDestroy_String(PetscViewer viewer)
{
  PetscViewer_String *vstr = (PetscViewer_String *)viewer->data;

  PetscFunctionBegin;
  if (vstr->ownstring) PetscCall(PetscFree(vstr->string));
  PetscCall(PetscFree(vstr));
  PetscFunctionReturn(0);
}

/*@C
    PetscViewerStringSPrintf - Prints information to a `PETSCVIEWERSTRING` `PetscViewer` object

    Logically Collective on viewer

    Input Parameters:
+   v - a string `PetscViewer`, formed by `PetscViewerStringOpen()`
-   format - the format of the input

    Level: developer

    Note:
    Though this is collective each MPI rank maintains a separate string

    Fortran Note:
    This routine is not supported in Fortran.

.seealso: `PETSCVIEWERSTRING`, `PetscViewerStringOpen()`, `PetscViewerStringGetStringRead()`, `PetscViewerStringSetString()`
@*/
PetscErrorCode PetscViewerStringSPrintf(PetscViewer viewer, const char format[], ...)
{
  va_list             Argp;
  size_t              fullLength;
  size_t              shift, cshift;
  PetscBool           isstring;
  char                tmp[4096];
  PetscViewer_String *vstr = (PetscViewer_String *)viewer->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscValidCharPointer(format, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERSTRING, &isstring));
  if (!isstring) PetscFunctionReturn(0);
  PetscCheck(vstr->string, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Must call PetscViewerStringSetString() before using");

  va_start(Argp, format);
  PetscCall(PetscVSNPrintf(tmp, 4096, format, &fullLength, Argp));
  va_end(Argp);
  PetscCall(PetscStrlen(tmp, &shift));
  cshift = shift + 1;
  if (cshift >= vstr->maxlen - vstr->curlen - 1) cshift = vstr->maxlen - vstr->curlen - 1;
  PetscCall(PetscStrncpy(vstr->head, tmp, cshift));
  vstr->head += shift;
  vstr->curlen += shift;
  PetscFunctionReturn(0);
}

/*@C
    PetscViewerStringOpen - Opens a string as a `PETSCVIEWERSTRING` `PetscViewer`. This is a very
    simple `PetscViewer`; information on the object is simply stored into
    the string in a fairly nice way.

    Collective

    Input Parameters:
+   comm - the communicator
.   string - the string to use
-   len    - the string length

    Output Parameter:
.   lab - the `PetscViewer`

    Level: advanced

    Fortran Note:
    This routine is not supported in Fortran.

.seealso: `PETSCVIEWERSTRING`, `PetscViewerDestroy()`, `PetscViewerStringSPrintf()`, `PetscViewerStringGetStringRead()`, `PetscViewerStringSetString()`
@*/
PetscErrorCode PetscViewerStringOpen(MPI_Comm comm, char string[], size_t len, PetscViewer *lab)
{
  PetscFunctionBegin;
  PetscCall(PetscViewerCreate(comm, lab));
  PetscCall(PetscViewerSetType(*lab, PETSCVIEWERSTRING));
  PetscCall(PetscViewerStringSetString(*lab, string, len));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscViewerGetSubViewer_String(PetscViewer viewer, MPI_Comm comm, PetscViewer *sviewer)
{
  PetscViewer_String *vstr = (PetscViewer_String *)viewer->data;

  PetscFunctionBegin;
  PetscCall(PetscViewerStringOpen(PETSC_COMM_SELF, vstr->head, vstr->maxlen - vstr->curlen, sviewer));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscViewerRestoreSubViewer_String(PetscViewer viewer, MPI_Comm comm, PetscViewer *sviewer)
{
  PetscViewer_String *iviewer = (PetscViewer_String *)(*sviewer)->data;
  PetscViewer_String *vstr    = (PetscViewer_String *)viewer->data;

  PetscFunctionBegin;
  vstr->head = iviewer->head;
  vstr->curlen += iviewer->curlen;
  PetscCall(PetscViewerDestroy(sviewer));
  PetscFunctionReturn(0);
}

/*MC
   PETSCVIEWERSTRING - A viewer that writes to a string

  Level: beginner

.seealso: `PetscViewerStringOpen()`, `PetscViewerStringSPrintf()`, `PetscViewerSocketOpen()`, `PetscViewerDrawOpen()`, `PETSCVIEWERSOCKET`,
          `PetscViewerCreate()`, `PetscViewerASCIIOpen()`, `PetscViewerBinaryOpen()`, `PETSCVIEWERBINARY`, `PETSCVIEWERDRAW`,
          `PetscViewerMatlabOpen()`, `VecView()`, `DMView()`, `PetscViewerMatlabPutArray()`, `PETSCVIEWERASCII`, `PETSCVIEWERMATLAB`,
          `PetscViewerFileSetName()`, `PetscViewerFileSetMode()`, `PetscViewerFormat`, `PetscViewerType`, `PetscViewerSetType()`
M*/

PETSC_EXTERN PetscErrorCode PetscViewerCreate_String(PetscViewer v)
{
  PetscViewer_String *vstr;

  PetscFunctionBegin;
  v->ops->destroy          = PetscViewerDestroy_String;
  v->ops->view             = NULL;
  v->ops->flush            = NULL;
  v->ops->getsubviewer     = PetscViewerGetSubViewer_String;
  v->ops->restoresubviewer = PetscViewerRestoreSubViewer_String;
  PetscCall(PetscNew(&vstr));
  v->data      = (void *)vstr;
  vstr->string = NULL;
  PetscFunctionReturn(0);
}

/*@C

   PetscViewerStringGetStringRead - Returns the string that a `PETSCVIEWERSTRING` uses

   Logically Collective on viewer

  Input Parameter:
.   viewer -  `PETSCVIEWERSTRING` viewer

  Output Parameters:
+    string - the string, optional use NULL if you do not need
-   len - the length of the string, optional use NULL if you do

  Note:
  Do not write to the string nor free it

  Level: advanced

.seealso: `PetscViewerStringOpen()`, `PETSCVIEWERSTRING`, `PetscViewerStringSetString()`, `PetscViewerStringSPrintf()`,
          `PetscViewerStringSetOwnString()`
@*/
PetscErrorCode PetscViewerStringGetStringRead(PetscViewer viewer, const char *string[], size_t *len)
{
  PetscViewer_String *vstr = (PetscViewer_String *)viewer->data;
  PetscBool           isstring;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERSTRING, &isstring));
  PetscCheck(isstring, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Only for PETSCVIEWERSTRING");
  if (string) *string = vstr->string;
  if (len) *len = vstr->maxlen;
  PetscFunctionReturn(0);
}

/*@C

   PetscViewerStringSetString - sets the string that a string viewer will print to

   Logically Collective on viewer

  Input Parameters:
+   viewer - string viewer you wish to attach string to
.   string - the string to print data into
-   len - the length of the string

  Note: The function does not copy the string, it uses it directly therefore you cannot free
   the string until the viewer is destroyed. If you call `PetscViewerStringSetOwnString()` the ownership
   passes to the viewer and it will be responsable for freeing it. In this case the string must be
   obtained with `PetscMalloc()`.

  Level: advanced

.seealso: `PetscViewerStringOpen()`, `PETSCVIEWERSTRING`, `PetscViewerStringGetStringRead()`, `PetscViewerStringSPrintf()`,
          `PetscViewerStringSetOwnString()`
@*/
PetscErrorCode PetscViewerStringSetString(PetscViewer viewer, char string[], size_t len)
{
  PetscViewer_String *vstr = (PetscViewer_String *)viewer->data;
  PetscBool           isstring;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscValidCharPointer(string, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERSTRING, &isstring));
  if (!isstring) PetscFunctionReturn(0);
  PetscCheck(len > 2, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "String must have length at least 2");

  PetscCall(PetscArrayzero(string, len));
  vstr->string = string;
  vstr->head   = string;
  vstr->curlen = 0;
  vstr->maxlen = len;
  PetscFunctionReturn(0);
}

/*@C

   PetscViewerStringSetOwnString - tells the viewer that it now owns the string and is responsible for freeing it

   Logically Collective on viewer

  Input Parameters:
.   viewer - string viewer

  Note:
  If you call this the string must have been obtained with `PetscMalloc()` and you cannot free the string

  Level: advanced

.seealso: `PetscViewerStringOpen()`, `PETSCVIEWERSTRING`, `PetscViewerStringGetStringRead()`, `PetscViewerStringSPrintf()`,
          `PetscViewerStringSetString()`
@*/
PetscErrorCode PetscViewerStringSetOwnString(PetscViewer viewer)
{
  PetscViewer_String *vstr = (PetscViewer_String *)viewer->data;
  PetscBool           isstring;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERSTRING, &isstring));
  if (!isstring) PetscFunctionReturn(0);

  vstr->ownstring = PETSC_TRUE;
  PetscFunctionReturn(0);
}
