#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: stringv.c,v 1.24 1998/08/26 22:03:57 balay Exp bsmith $";
#endif

#include "src/viewer/viewerimpl.h"   /*I  "petsc.h"  I*/
#include <stdarg.h>
#if defined(HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#include "pinclude/petscfix.h"

typedef struct  {
  char         *string;   /* string where info is stored */
  char         *head;     /* pointer to begining of unused portion */
  int          curlen,maxlen;
} Viewer_String;

static int ViewerDestroy_String(Viewer viewer)
{
  Viewer_String *vstr = (Viewer_String *)viewer->data;

  PetscFunctionBegin;
  PetscFree(vstr);
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
  va_list       Argp;
  int           shift;
  char          tmp[512];
  Viewer_String *vstr = (Viewer_String *) v->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VIEWER_COOKIE);
  if (PetscStrcmp(v->type_name,"string")) PetscFunctionReturn(0);

  va_start( Argp, format );
#if defined(HAVE_VPRINTF_CHAR)
  vsprintf(tmp,format,(char *)Argp);
#else
  vsprintf(tmp,format,Argp);
#endif
  va_end( Argp );

  shift = PetscStrlen(tmp);
  if (shift > 512) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"String too long");
  
  if (shift >= vstr->maxlen - vstr->curlen - 1) shift = vstr->maxlen - vstr->curlen - 1;
  PetscStrncpy(vstr->head,tmp,shift);

  vstr->head   += shift;
  vstr->curlen += shift;
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
  Viewer        v;
  Viewer_String *vstr;

  PetscFunctionBegin;
  PetscHeaderCreate(v,_p_Viewer,struct _ViewerOps,VIEWER_COOKIE,0,comm,ViewerDestroy,0);
  PLogObjectCreate(v);
  v->ops->destroy = ViewerDestroy_String;
  v->ops->view    = 0;
  v->ops->flush   = 0;
  vstr            = PetscNew(Viewer_String);
  v->data         = (void *) vstr;
  v->type_name    = (char *) PetscMalloc((1+PetscStrlen(STRING_VIEWER))*sizeof(char));CHKPTRQ(v->type_name);
  PetscStrcpy(v->type_name,STRING_VIEWER);

  PetscMemzero(string,len*sizeof(char));
  vstr->string      = string;
  vstr->head        = string;

  vstr->curlen      = 0;
  vstr->maxlen      = len;

  *lab           = v;
  PetscFunctionReturn(0);
}






