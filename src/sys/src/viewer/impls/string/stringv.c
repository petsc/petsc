#ifndef lint
static char vcid[] = "$Id: stringv.c,v 1.10 1996/12/19 01:13:20 balay Exp bsmith $";
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
  int          curlen,maxlen;
};

static int ViewerDestroy_String(PetscObject obj)
{
  PLogObjectDestroy(obj);
  PetscHeaderDestroy(obj);
  return 0;
}

#undef __FUNCTION__  
#define __FUNCTION__ "ViewerStringSPrintf"
/*@C
    ViewerStringSPrintf - Prints information to a viewer string.

    Input Parameters:
.   v - a string viewer, formed by ViewerStringOpen()
.   format - the format of the input

    Fortran Note:
    This routine is not supported in Fortran.

.keywords: Viewer, string, printf

.seealso: ViewerStringOpen()
@*/
int ViewerStringSPrintf(Viewer v,char *format,...)
{
  va_list Argp;
  int     shift;
  char    tmp[512];

  PetscValidHeaderSpecific(v,VIEWER_COOKIE);
  if (v->type != STRING_VIEWER) return 0;

  va_start( Argp, format );
#if (__GNUC__ == 2 && __GNUC_MINOR__ >= 7 && defined(PARCH_freebsd) )
  vsprintf(tmp,format,(char *)Argp);
#else
  vsprintf(tmp,format,Argp);
#endif
  va_end( Argp );

  shift = PetscStrlen(tmp);
  if (shift > 512) SETERRQ(1,0,"String too long");
  
  if (shift >= v->maxlen - v->curlen - 1) shift = v->maxlen - v->curlen - 1;
  PetscStrncpy(v->head,tmp,shift);

  v->head   += shift;
  v->curlen += shift;
  return 0;
}

#undef __FUNCTION__  
#define __FUNCTION__ "ViewerStringOpen"
/*@C
    ViewerStringOpen - Opens a string as a viewer. This is a very 
    simple viewer; information on the object is simply stored into 
    the string in a fairly nice way.

    Input Parameters:
.   comm - the communicator
.   string - the string to use

    Output Parameter:
.   lab - the viewer

    Fortran Note:
    This routine is not supported in Fortran.

.keywords: Viewer, string, open

.seealso: ViewerDestroy(), ViewerStringSPrintf()
@*/
int ViewerStringOpen(MPI_Comm comm,char *string,int len, Viewer *lab)
{
  Viewer v;
  PetscHeaderCreate(v,_Viewer,VIEWER_COOKIE,STRING_VIEWER,comm);
  PLogObjectCreate(v);
  v->destroy     = ViewerDestroy_String;

  PetscMemzero(string,len*sizeof(char));
  v->string      = string;
  v->head        = string;

  v->curlen      = 0;
  v->maxlen      = len;

  *lab           = v;
  return 0;
}






