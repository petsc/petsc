#ifndef lint
static char vcid[] = "$Id: filev.c,v 1.1 1995/03/23 00:29:32 bsmith Exp bsmith $";
#endif

#include "ptscimpl.h"
#include <stdarg.h>

struct _Viewer {
  PETSCHEADER
  FILE        *fd;
};

Viewer STDOUT_VIEWER,STDERR_VIEWER;

int ViewerInitialize()
{
  ViewerFileCreate("stderr",&STDERR_VIEWER);
  ViewerFileCreate("stout",&STDOUT_VIEWER);
  return 0;
}

static int FileDestroy(PetscObject obj)
{
  Viewer v = (Viewer) obj;
  fclose(v->fd);
  PETSCOBJECTDESTROY(obj);
  return 0;
}

/*@
    ViewerPrintf  - Prints to the file pointed to by viewer, some 
         viewers may not support this option.

  Input Parameters:
.   viewer - file to print to
.   comm - communicator
.   format - printf style format string 
.   arguments
@*/
int ViewerPrintf(Viewer viewer,char *format,...)
{
  va_list Argp;
  if (!viewer) return 0;
  if (viewer->type != FILE_VIEWER) return 0;
  va_start( Argp, format );
  vfprintf(viewer->fd,format,Argp);
  va_end( Argp );
  return 0;
}

int ViewerFlush(Viewer viewer)
{
  if (!viewer) return 0;
  if (viewer->type != FILE_VIEWER) return 0;
  fflush(viewer->fd);
  return 0;
}

/*@
     ViewerFileOpen - Opens an ASCI file as a viewer.

  Input Parameters:
.   name - the file name

  Output Parameter:
.   lab - the viewer to use with that file.
@*/
int ViewerFileOpen(char *name,Viewer *lab)
{
  Viewer v;
  PETSCHEADERCREATE(v,_Viewer,VIEWER_COOKIE,FILE_VIEWER,MPI_COMM_SELF);
  PLogObjectCreate(v);
  v->destroy     = FileDestroy;

  if (!strcmp(name,"stderr")) v->fd = stderr;
  else if (!strcmp(name,"stdout")) v->fd = stdout;
  else {
    v->fd          = fopen(name,"w"); if (!v->fd) SETERR(1,0);
  }
  *lab           = v;
  return 0;
}



