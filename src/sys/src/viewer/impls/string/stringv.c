#ifndef lint
static char vcid[] = "$Id: stringv.c,v 1.4 1996/03/19 21:28:48 bsmith Exp bsmith $";
#endif

#include "petsc.h"
#include "pinclude/pviewer.h"
#include <stdio.h>
#include <stdarg.h>
#if defined(HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#include "pinclude/petscfix.h"

struct _Viewer {
  VIEWERHEADER
  char         *string;   /* string where info is stored */
  char         *head;     /* pointer to begining of unused portion */
};

static int ViewerDestroy_String(PetscObject obj)
{
  PLogObjectDestroy(obj);
  PetscHeaderDestroy(obj);
  return 0;
}

/*@C
      ViewerStringSPrintf - Prints information to a viewer string

  Input Parameters:
.   v - the viewer
.   format - the format of the input

@*/
int ViewerStringSPrintf(Viewer v,char *format,...)
{
  va_list Argp;
  int     shift;

  PetscValidHeaderSpecific(v,VIEWER_COOKIE);
  if (v->type != STRING_VIEWER) return 0;
  va_start( Argp, format );
  vsprintf(v->head,format,Argp);
  va_end( Argp );

  /* update the position of v->head */
  for ( shift=0; shift<256; shift++ ) {
    if (v->head[shift] == 0) break;
  }
  v->head += shift;
  return 0;
}

/*@C
   ViewerStringOpen - Opens a string as a viewer. This is a very 
        simply viewer, information on the object is simply stored into 
        the string in a fairly nice way.

   Input Parameters:
.  comm - the communicator
.  string - the string to use

   Output Parameter:
.  lab - the viewer

.keywords: Viewer, file, open

.seealso: MatView(), VecView(), ViewerDestroy(), ViewerFileOpenBinary(),
          ViewerASCIIGetPointer(), SLESView(), MatView()
@*/
int ViewerStringOpen(MPI_Comm comm,char *string,int len, Viewer *lab)
{
  Viewer v;
  PetscHeaderCreate(v,_Viewer,VIEWER_COOKIE,STRING_VIEWER,comm);
  PLogObjectCreate(v);
  v->destroy     = ViewerDestroy_String;

  v->string      = string;
  v->head        = string;

  *lab           = v;
  return 0;
}






