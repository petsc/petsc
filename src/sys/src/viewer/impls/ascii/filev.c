#ifndef lint
static char vcid[] = "$Id: filev.c,v 1.5 1995/04/17 02:17:36 bsmith Exp curfman $";
#endif

#include "ptscimpl.h"
#include <stdarg.h>

struct _Viewer {
  PETSCHEADER
  FILE        *fd;
};

Viewer STDOUT_VIEWER,STDERR_VIEWER,SYNC_STDOUT_VIEWER;

int ViewerInitialize()
{
  ViewerFileOpen("stderr",&STDERR_VIEWER);
  ViewerFileOpen("stdout",&STDOUT_VIEWER);
  ViewerSyncFileOpen("stdout",MPI_COMM_WORLD,&SYNC_STDOUT_VIEWER);
  return 0;
}

static int ViewerDestroy_File(PetscObject obj)
{
  Viewer v = (Viewer) obj;
  int    mytid = 0;
  if (v->type == FILES_VIEWER) {MPI_Comm_rank(v->comm,&mytid);} 
  if (!mytid && v->fd != stderr && v->fd != stdout) fclose(v->fd);
  PLogObjectDestroy(obj);
  PETSCHEADERDESTROY(obj);
  return 0;
}

FILE *ViewerFileGetPointer(Viewer viewer)
{
  return viewer->fd;
}

/*@
   ViewerFileOpen - Opens an ASCII file as a viewer.

   Input Parameter:
.  name - the file name

   Output Parameter:
.  lab - the viewer to use with that file

.keywords: Viewer, file, open

.seealso: ViewerFileSyncOpen()
@*/
int ViewerFileOpen(char *name,Viewer *lab)
{
  Viewer v;
  PETSCHEADERCREATE(v,_Viewer,VIEWER_COOKIE,FILE_VIEWER,MPI_COMM_SELF);
  PLogObjectCreate(v);
  v->destroy     = ViewerDestroy_File;

  if (!strcmp(name,"stderr")) v->fd = stderr;
  else if (!strcmp(name,"stdout")) v->fd = stdout;
  else {
    v->fd          = fopen(name,"w"); if (!v->fd) SETERR(1,0);
  }
#if defined(PETSC_LOG)
  PLogObjectState((PetscObject)v,"File: %s",name);
#endif
  *lab           = v;
  return 0;
}
/*@
   ViewerSyncFileOpen - Opens an ASCII file as a viewer, where only the first
   processor opens the file. All other processors send their data to the 
   first processor to print. 

   Input Parameters:
.  name - the file name
.  comm - the communicator

   Output Parameter:
.  lab - the viewer to use with that file

.keywords: Viewer, file, open

.seealso: ViewerFileOpen()
@*/
int ViewerSyncFileOpen(char *name,MPI_Comm comm,Viewer *lab)
{
  Viewer v;
  PETSCHEADERCREATE(v,_Viewer,VIEWER_COOKIE,FILES_VIEWER,comm);
  PLogObjectCreate(v);
  v->destroy     = ViewerDestroy_File;

  if (!strcmp(name,"stderr")) v->fd = stderr;
  else if (!strcmp(name,"stdout")) v->fd = stdout;
  else {
    v->fd        = fopen(name,"w"); if (!v->fd) SETERR(1,0);
  }
#if defined(PETSC_LOG)
  PLogObjectState((PetscObject)v,"File: %s",name);
#endif
  *lab           = v;
  return 0;
}



