/* $Id: petscvu.c,v 1.0 2001/04/10 19:34:05 knepley Exp $ */

#include "src/sys/src/viewer/viewerimpl.h"  /*I     "petsc.h"   I*/
#include "petscfix.h"

typedef struct {
  FILE         *fd;
  PetscFileMode mode;           /* The mode in which to open the file */
  char         *filename;
} PetscViewer_VU;

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerDestroy_VU" 
int PetscViewerDestroy_VU(PetscViewer viewer)
{
  PetscViewer_VU *vu = (PetscViewer_VU *) viewer->data;
  int             ierr;

  PetscFunctionBegin;
  ierr = PetscFClose(viewer->comm, vu->fd);                                                               CHKERRQ(ierr);
  ierr = PetscStrfree(vu->filename);                                                                      CHKERRQ(ierr);
  ierr = PetscFree(vu);                                                                                   CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerFlush_VU" 
int PetscViewerFlush_VU(PetscViewer viewer)
{
  PetscViewer_VU *vu = (PetscViewer_VU *) viewer->data;
  int             rank;
  int             ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(viewer->comm, &rank);                                                              CHKERRQ(ierr);
  if (rank == 0) fflush(vu->fd);
  PetscFunctionReturn(0);  
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PetscViewerGetFilename_VU" 
int PetscViewerGetFilename_VU(PetscViewer viewer, char **name)
{
  PetscViewer_VU *vu = (PetscViewer_VU *) viewer->data;

  PetscFunctionBegin;
  *name = vu->filename;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PetscViewerSetFilename_VU" 
int PetscViewerSetFilename_VU(PetscViewer viewer, const char name[])
{
  PetscViewer_VU *vu = (PetscViewer_VU *) viewer->data;
  char            fname[256];
  int             rank;
  int             ierr;

  PetscFunctionBegin;
  if (name == PETSC_NULL) PetscFunctionReturn(0);
  ierr = MPI_Comm_rank(viewer->comm, &rank);                                                              CHKERRQ(ierr);
  if (rank != 0) PetscFunctionReturn(0);
  ierr = PetscStrallocpy(name, &vu->filename);                                                            CHKERRQ(ierr);
  ierr = PetscFixFilename(name, fname);                                                                   CHKERRQ(ierr);
  switch(vu->mode) {
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
    if (vu->fd == PETSC_NULL) {
      vu->fd = fopen(fname, "w+");
    }
    break;
  case FILE_MODE_APPEND_UPDATE:
    /* I really want a file which is opened at the end for updating,
       not a+, which opens at the beginning, but makes writes at the end.
    */
    vu->fd = fopen(fname, "r+");
    if (vu->fd == PETSC_NULL) {
      vu->fd = fopen(fname, "w+");
    } else {
      ierr = fseek(vu->fd, 0, SEEK_END);                                                                  CHKERRQ(ierr);
    }
    break;
  default:
    SETERRQ1(PETSC_ERR_ARG_WRONG, "Invalid file mode %d", vu->mode);
  }

  if (!vu->fd) SETERRQ1(PETSC_ERR_FILE_OPEN, "Cannot open PetscViewer file: %s", fname);
#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject) viewer, "File: %s", name);
#endif

  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PetscViewerCreate_VU" 
int PetscViewerCreate_VU(PetscViewer viewer)
{
  PetscViewer_VU *vu;
  int             ierr;

  PetscFunctionBegin;
  ierr         = PetscNew(PetscViewer_VU, &vu);                                                           CHKERRQ(ierr);
  viewer->data = (void*) vu;

  viewer->ops->destroy          = PetscViewerDestroy_VU;
  viewer->ops->flush            = PetscViewerFlush_VU;
  viewer->ops->getsingleton     = PETSC_NULL;
  viewer->ops->restoresingleton = PETSC_NULL;
  viewer->format                = PETSC_VIEWER_ASCII_DEFAULT;
  viewer->iformat               = 0;

  vu->fd       = PETSC_NULL;
  vu->mode     = FILE_MODE_WRITE;
  vu->filename = PETSC_NULL;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject) viewer,"PetscViewerSetFilename_C", "PetscViewerSetFilename_VU",
                                           PetscViewerSetFilename_VU);                                    CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject) viewer,"PetscViewerGetFilename_C", "PetscViewerGetFilename_VU",
                                           PetscViewerGetFilename_VU);                                    CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerVUGetPointer" 
/*@C
  PetscViewerVUGetPointer - Extracts the file pointer from a VU PetscViewer.

  Not Collective

  Input Parameter:
. viewer - The PetscViewer

  Output Parameter:
. fd     - The file pointer

  Level: intermediate

  Concepts: PetscViewer^file pointer
  Concepts: file pointer^getting from PetscViewer

.seealso: PetscViewerASCIIGetPointer()
@*/
int PetscViewerVUGetPointer(PetscViewer viewer, FILE **fd)
{
  PetscViewer_VU *vu = (PetscViewer_VU *) viewer->data;

  PetscFunctionBegin;
  PetscValidPointer(fd);
  *fd = vu->fd;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerVUSetMode"
/*@C
  PetscViewerVUSetMode - Sets the mode in which to open the file.

  Not Collective

  Input Parameters:
+ viewer - The PetscViewer
- mode   - The file mode

  Level: intermediate

.keywords: Viewer, file, get, pointer
.seealso: PetscViewerASCIISetMode()
@*/
int PetscViewerVUSetMode(PetscViewer viewer, PetscFileMode mode)
{
  PetscViewer_VU *vu = (PetscViewer_VU *) viewer->data;

  PetscFunctionBegin;
  vu->mode = mode;
  PetscFunctionReturn(0);
}
