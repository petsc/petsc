#ifndef lint
static char vcid[] = "$Id: filev.c,v 1.7 1995/04/26 19:33:09 curfman Exp curfman $";
#endif

#include "ptscimpl.h"
#include <stdarg.h>

struct _Viewer {
  PETSCHEADER
  FILE        *fd;
  int         format;
  char        *outputname;
};

Viewer STDOUT_VIEWER,STDERR_VIEWER,SYNC_STDOUT_VIEWER;

int ViewerInitialize()
{
  ViewerFileOpen("stderr",&STDERR_VIEWER);
  ViewerFileOpen("stdout",&STDOUT_VIEWER);
  ViewerFileOpenSync("stdout",MPI_COMM_WORLD,&SYNC_STDOUT_VIEWER);
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

FILE *ViewerFileGetPointer_Private(Viewer viewer)
{
  return viewer->fd;
}

char *ViewerFileGetOutputname_Private(Viewer viewer)
{
  return viewer->outputname;
}

int ViewerFileGetFormat_Private(Viewer viewer,int *fmt)
{
  *fmt = viewer->format;
  return 0;
}

/*@
   ViewerFileOpen - Opens an ASCII file as a viewer.

   Input Parameter:
.  name - the file name

   Output Parameter:
.  lab - the viewer to use with that file

   Notes:
   As shown below, ViewerFileOpen() is useful in conjunction with 
   MatView() and VecView()
$
$     ViewerFileOpen("mat.output", &viewer);
$     MatView(matrix, viewer);

.keywords: Viewer, file, open

.seealso: ViewerFileSyncOpen(), MatView(), VecView()
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
  v->format        = FILE_FORMAT_DEFAULT;
  v->outputname    = 0;
#if defined(PETSC_LOG)
  PLogObjectState((PetscObject)v,"File: %s",name);
#endif
  *lab           = v;
  return 0;
}
/*@
   ViewerFileOpenSync - Opens an ASCII file as a viewer, where only the first
   processor opens the file. All other processors send their data to the 
   first processor to print. 

   Input Parameters:
.  name - the file name
.  comm - the communicator

   Output Parameter:
.  lab - the viewer to use with that file

   Notes:
   As shown below, ViewerFileOpenSync() is useful in conjunction with 
   MatView() and VecView()
$
$     ViewerFileOpenSycn("mat.output", &viewer);
$     MatView(matrix, viewer);

.keywords: Viewer, file, open

.seealso: ViewerFileOpen(), MatView(), VecView()
@*/
int ViewerFileOpenSync(char *name,MPI_Comm comm,Viewer *lab)
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
  v->format        = FILE_FORMAT_DEFAULT;
  v->outputname    = 0;
#if defined(PETSC_LOG)
  PLogObjectState((PetscObject)v,"File: %s",name);
#endif
  *lab           = v;
  return 0;
}

/*@
   ViewerFileSetFormat - Sets the format for file viewers.

   Input Parameters:
.  v - the viewer
.  format - the format
.  char - optional object name

   Notes:
   Available formats include
$    FILE_FORMAT_DEFAULT - default
$    FILE_FORMAT_MATLAB - Matlab format
$    FILE_FORMAT_IMPL - implementation-specific format
$      (which is in many cases the same as the default)

   Currently, the object name is used only in the Matlab format.

.keywords: Viewer, file, set, format

.seealso: ViewerFileOpen(), ViewerFileOpenSync(), MatView(), VecView()
@*/
int ViewerFileSetFormat(Viewer v,int format,char *name)
{
  VALIDHEADER(v,VIEWER_COOKIE);
  if (v->type == FILES_VIEWER || v->type == FILE_VIEWER) {
    v->format = format;
    v->outputname = name;
  }
  return 0;
}
