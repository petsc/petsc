#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: stringv.c,v 1.23 1998/06/15 20:32:33 bsmith Exp balay $";
#endif

#include "petsc.h"
#include "pinclude/pviewer.h"
#include <stdarg.h>
#if defined(HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#include "pinclude/petscfix.h"

struct _p_Viewer {
  VIEWERHEADER
  char         *string;   /* string where info is stored */
  char         *head;     /* pointer to begining of unused portion */
  int          curlen,maxlen;
};

static int ViewerDestroy_String(Viewer viewer)
{
  PetscFunctionBegin;

  PLogObjectDestroy((PetscObject)viewer);
  PetscHeaderDestroy((PetscObject)viewer);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerStringSPrintf"
/*@C
    ViewerStringSPrintf - Prints information to a viewer string.

    Collective on Viewer (Hmmm, each processor maintains a seperate string)

    Input Parameters:
+   v - a string viewer, formed by ViewerStringOpen()
-   format - the format of the input

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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VIEWER_COOKIE);
  if (v->type != STRING_VIEWER) PetscFunctionReturn(0);

  va_start( Argp, format );
#if defined(HAVE_VPRINTF_CHAR)
  vsprintf(tmp,format,(char *)Argp);
#else
  vsprintf(tmp,format,Argp);
#endif
  va_end( Argp );

  shift = PetscStrlen(tmp);
  if (shift > 512) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"String too long");
  
  if (shift >= v->maxlen - v->curlen - 1) shift = v->maxlen - v->curlen - 1;
  PetscStrncpy(v->head,tmp,shift);

  v->head   += shift;
  v->curlen += shift;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerStringOpen"
/*@C
    ViewerStringOpen - Opens a string as a viewer. This is a very 
    simple viewer; information on the object is simply stored into 
    the string in a fairly nice way.

    Collective on MPI_Comm

    Input Parameters:
+   comm - the communicator
-   string - the string to use

    Output Parameter:
.   lab - the viewer

    Fortran Note:
    This routine is not supported in Fortran.

.keywords: Viewer, string, open

.seealso: ViewerDestroy(), ViewerStringSPrintf()
@*/
int ViewerStringOpen(MPI_Comm comm,char string[],int len, Viewer *lab)
{
  Viewer v;

  PetscFunctionBegin;
  PetscHeaderCreate(v,_p_Viewer,int,VIEWER_COOKIE,STRING_VIEWER,comm,ViewerDestroy,0);
  PLogObjectCreate(v);
  v->destroy     = ViewerDestroy_String;

  PetscMemzero(string,len*sizeof(char));
  v->string      = string;
  v->head        = string;

  v->curlen      = 0;
  v->maxlen      = len;

  *lab           = v;
  PetscFunctionReturn(0);
}






