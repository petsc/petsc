#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: stringv.c,v 1.29 1999/03/17 23:21:04 bsmith Exp bsmith $";
#endif

#include "src/sys/src/viewer/viewerimpl.h"   /*I  "petsc.h"  I*/
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

    Level: developer

    Fortran Note:
    This routine is not supported in Fortran.

.keywords: Viewer, string, printf

.seealso: ViewerStringOpen()
@*/
int ViewerStringSPrintf(Viewer v,char *format,...)
{
  va_list       Argp;
  int           shift,ierr;
  char          tmp[512];
  Viewer_String *vstr = (Viewer_String *) v->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VIEWER_COOKIE);
  if (!PetscTypeCompare(v->type_name,"string")) PetscFunctionReturn(0);
  if (!vstr->string) SETERRQ(1,1,"Must call ViewerStringSetString() before using");

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
  ierr = PetscStrncpy(vstr->head,tmp,shift);CHKERRQ(ierr);

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

    Level: advanced

    Fortran Note:
    This routine is not supported in Fortran.

.keywords: Viewer, string, open

.seealso: ViewerDestroy(), ViewerStringSPrintf()
@*/
int ViewerStringOpen(MPI_Comm comm,char string[],int len, Viewer *lab)
{
  int ierr;
  
  PetscFunctionBegin;
  ierr = ViewerCreate(comm,lab);CHKERRQ(ierr);
  ierr = ViewerSetType(*lab,STRING_VIEWER);CHKERRQ(ierr);
  ierr = ViewerStringSetString(*lab,string,len);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "ViewerCreate_String"
int ViewerCreate_String(Viewer v)
{
  Viewer_String *vstr;
  int           ierr;

  PetscFunctionBegin;
  v->ops->destroy = ViewerDestroy_String;
  v->ops->view    = 0;
  v->ops->flush   = 0;
  vstr            = PetscNew(Viewer_String);
  v->data         = (void *) vstr;
  v->type_name    = (char *) PetscMalloc((1+PetscStrlen(STRING_VIEWER))*sizeof(char));CHKPTRQ(v->type_name);
  ierr = PetscStrcpy(v->type_name,STRING_VIEWER);CHKERRQ(ierr);
  vstr->string    = 0;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNC__  
#define __FUNC__ "ViewerStringSetString"
int ViewerStringSetString(Viewer v,char string[],int len)
{
  Viewer_String *vstr = (Viewer_String *) v->data;

  PetscFunctionBegin;
  if (!PetscTypeCompare(v->type_name,"string")) PetscFunctionReturn(0);
  PetscMemzero(string,len*sizeof(char));
  vstr->string      = string;
  vstr->head        = string;

  vstr->curlen      = 0;
  vstr->maxlen      = len;
  PetscFunctionReturn(0);
}






