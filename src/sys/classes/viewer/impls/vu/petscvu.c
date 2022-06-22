
#include <petsc/private/viewerimpl.h>  /*I     "petscsys.h"   I*/

#define QUEUESTRINGSIZE 1024

typedef struct _PrintfQueue *PrintfQueue;
struct _PrintfQueue {
  char        string[QUEUESTRINGSIZE];
  PrintfQueue next;
};

typedef struct {
  FILE          *fd;
  PetscFileMode mode;     /* The mode in which to open the file */
  char          *filename;
  PetscBool     vecSeen;  /* The flag indicating whether any vector has been viewed so far */
  PrintfQueue   queue, queueBase;
  int           queueLength;
} PetscViewer_VU;

static PetscErrorCode PetscViewerFileClose_VU(PetscViewer viewer)
{
  PetscViewer_VU *vu = (PetscViewer_VU*) viewer->data;

  PetscFunctionBegin;
  if (vu->vecSeen) {
    PetscCall(PetscViewerVUPrintDeferred(viewer, "};\n\n"));
  }
  PetscCall(PetscViewerVUFlushDeferred(viewer));
  PetscCall(PetscFClose(PetscObjectComm((PetscObject)viewer), vu->fd));
  vu->fd = NULL;
  PetscCall(PetscFree(vu->filename));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscViewerDestroy_VU(PetscViewer viewer)
{
  PetscViewer_VU *vu = (PetscViewer_VU*) viewer->data;

  PetscFunctionBegin;
  PetscCall(PetscViewerFileClose_VU(viewer));
  PetscCall(PetscFree(vu));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer,"PetscViewerFileSetName_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer,"PetscViewerFileGetName_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer,"PetscViewerFileSetMode_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer,"PetscViewerFileGetMode_C",NULL));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscViewerFlush_VU(PetscViewer viewer)
{
  PetscViewer_VU *vu = (PetscViewer_VU*) viewer->data;
  PetscMPIInt    rank;
  int            err;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)viewer), &rank));
  if (rank == 0) {
    err = fflush(vu->fd);
    PetscCheck(!err,PETSC_COMM_SELF,PETSC_ERR_SYS,"fflush() failed on file");
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode  PetscViewerFileSetMode_VU(PetscViewer viewer, PetscFileMode mode)
{
  PetscViewer_VU *vu = (PetscViewer_VU*) viewer->data;

  PetscFunctionBegin;
  vu->mode = mode;
  PetscFunctionReturn(0);
}

static PetscErrorCode  PetscViewerFileGetMode_VU(PetscViewer viewer, PetscFileMode *type)
{
  PetscViewer_VU *vu = (PetscViewer_VU*) viewer->data;

  PetscFunctionBegin;
  *type = vu->mode;
  PetscFunctionReturn(0);
}

static PetscErrorCode  PetscViewerFileGetName_VU(PetscViewer viewer, const char **name)
{
  PetscViewer_VU *vu = (PetscViewer_VU*) viewer->data;

  PetscFunctionBegin;
  *name = vu->filename;
  PetscFunctionReturn(0);
}

static PetscErrorCode  PetscViewerFileSetName_VU(PetscViewer viewer, const char name[])
{
  PetscViewer_VU *vu = (PetscViewer_VU*) viewer->data;
  char           fname[PETSC_MAX_PATH_LEN];
  int            rank;

  PetscFunctionBegin;
  if (!name) PetscFunctionReturn(0);
  PetscCall(PetscViewerFileClose_VU(viewer));
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)viewer), &rank));
  if (rank != 0) PetscFunctionReturn(0);
  PetscCall(PetscStrallocpy(name, &vu->filename));
  PetscCall(PetscFixFilename(name, fname));
  switch (vu->mode) {
  case FILE_MODE_READ:
    vu->fd = fopen(fname, "r");
    break;
  case FILE_MODE_WRITE:
    vu->fd = fopen(fname, "w");
    break;
  case FILE_MODE_APPEND:
    vu->fd = fopen(fname, "a");
    break;
  case FILE_MODE_UPDATE:
    vu->fd = fopen(fname, "r+");
    if (!vu->fd) vu->fd = fopen(fname, "w+");
    break;
  case FILE_MODE_APPEND_UPDATE:
    /* I really want a file which is opened at the end for updating,
       not a+, which opens at the beginning, but makes writes at the end.
    */
    vu->fd = fopen(fname, "r+");
    if (!vu->fd) vu->fd = fopen(fname, "w+");
    else {
      PetscCall(fseek(vu->fd, 0, SEEK_END));
    }
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)viewer),PETSC_ERR_SUP, "Unsupported file mode %s",PetscFileModes[vu->mode]);
  }

  PetscCheck(vu->fd,PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN, "Cannot open PetscViewer file: %s", fname);
#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject) viewer, "File: %s", name);
#endif
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode PetscViewerCreate_VU(PetscViewer viewer)
{
  PetscViewer_VU *vu;

  PetscFunctionBegin;
  PetscCall(PetscNewLog(viewer,&vu));
  viewer->data = (void*) vu;

  viewer->ops->destroy          = PetscViewerDestroy_VU;
  viewer->ops->flush            = PetscViewerFlush_VU;
  viewer->ops->getsubviewer     = NULL;
  viewer->ops->restoresubviewer = NULL;

  vu->fd          = NULL;
  vu->mode        = FILE_MODE_WRITE;
  vu->filename    = NULL;
  vu->vecSeen     = PETSC_FALSE;
  vu->queue       = NULL;
  vu->queueBase   = NULL;
  vu->queueLength = 0;

  PetscCall(PetscObjectComposeFunction((PetscObject) viewer,"PetscViewerFileSetName_C",PetscViewerFileSetName_VU));
  PetscCall(PetscObjectComposeFunction((PetscObject) viewer,"PetscViewerFileGetName_C",PetscViewerFileGetName_VU));
  PetscCall(PetscObjectComposeFunction((PetscObject) viewer,"PetscViewerFileSetMode_C",PetscViewerFileSetMode_VU));
  PetscCall(PetscObjectComposeFunction((PetscObject) viewer,"PetscViewerFileGetMode_C",PetscViewerFileGetMode_VU));
  PetscFunctionReturn(0);
}

/*@C
  PetscViewerVUGetPointer - Extracts the file pointer from a VU PetscViewer.

  Not Collective

  Input Parameter:
. viewer - The PetscViewer

  Output Parameter:
. fd     - The file pointer

  Level: intermediate

.seealso: `PetscViewerASCIIGetPointer()`
@*/
PetscErrorCode  PetscViewerVUGetPointer(PetscViewer viewer, FILE **fd)
{
  PetscViewer_VU *vu = (PetscViewer_VU*) viewer->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  PetscValidPointer(fd,2);
  *fd = vu->fd;
  PetscFunctionReturn(0);
}

/*@C
  PetscViewerVUSetVecSeen - Sets the flag which indicates whether we have viewed
  a vector. This is usually called internally rather than by a user.

  Not Collective

  Input Parameters:
+ viewer  - The PetscViewer
- vecSeen - The flag which indicates whether we have viewed a vector

  Level: advanced

.seealso: `PetscViewerVUGetVecSeen()`
@*/
PetscErrorCode  PetscViewerVUSetVecSeen(PetscViewer viewer, PetscBool vecSeen)
{
  PetscViewer_VU *vu = (PetscViewer_VU*) viewer->data;

  PetscFunctionBegin;
  vu->vecSeen = vecSeen;
  PetscFunctionReturn(0);
}

/*@C
  PetscViewerVUGetVecSeen - Gets the flag which indicates whether we have viewed
  a vector. This is usually called internally rather than by a user.

  Not Collective

  Input Parameter:
. viewer  - The PetscViewer

  Output Parameter:
. vecSeen - The flag which indicates whether we have viewed a vector

  Level: advanced

.seealso: `PetscViewerVUGetVecSeen()`
@*/
PetscErrorCode  PetscViewerVUGetVecSeen(PetscViewer viewer, PetscBool  *vecSeen)
{
  PetscViewer_VU *vu = (PetscViewer_VU*) viewer->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  PetscValidBoolPointer(vecSeen,2);
  *vecSeen = vu->vecSeen;
  PetscFunctionReturn(0);
}

/*@C
  PetscViewerVUPrintDeferred - Prints to the deferred write cache instead of the file.

  Not Collective

  Input Parameters:
+ viewer - The PetscViewer
- format - The format string

  Level: intermediate

.seealso: `PetscViewerVUFlushDeferred()`
@*/
PetscErrorCode  PetscViewerVUPrintDeferred(PetscViewer viewer, const char format[], ...)
{
  PetscViewer_VU *vu = (PetscViewer_VU*) viewer->data;
  va_list        Argp;
  size_t         fullLength;
  PrintfQueue    next;

  PetscFunctionBegin;
  PetscCall(PetscNew(&next));
  if (vu->queue) {
    vu->queue->next = next;
    vu->queue       = next;
    vu->queue->next = NULL;
  } else {
    vu->queueBase   = vu->queue = next;
  }
  vu->queueLength++;

  va_start(Argp, format);
  PetscCall(PetscArrayzero(next->string,QUEUESTRINGSIZE));
  PetscCall(PetscVSNPrintf(next->string, QUEUESTRINGSIZE,format,&fullLength, Argp));
  va_end(Argp);
  PetscFunctionReturn(0);
}

/*@C
  PetscViewerVUFlushDeferred - Flushes the deferred write cache to the file.

  Not Collective

  Input Parameter:
. viewer - The PetscViewer

  Level: intermediate

.seealso: `PetscViewerVUPrintDeferred()`
@*/
PetscErrorCode  PetscViewerVUFlushDeferred(PetscViewer viewer)
{
  PetscViewer_VU *vu  = (PetscViewer_VU*) viewer->data;
  PrintfQueue    next = vu->queueBase;
  PrintfQueue    previous;
  int            i;

  PetscFunctionBegin;
  for (i = 0; i < vu->queueLength; i++) {
    PetscFPrintf(PetscObjectComm((PetscObject)viewer), vu->fd, "%s", next->string);
    previous = next;
    next     = next->next;
    PetscCall(PetscFree(previous));
  }
  vu->queue       = NULL;
  vu->queueLength = 0;
  PetscFunctionReturn(0);
}
