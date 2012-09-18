
#include <petsc-private/viewerimpl.h>   /*I  "petscsys.h"  I*/
#include <stdarg.h>
#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif

typedef struct  {
  char         *string;   /* string where info is stored */
  char         *head;     /* pointer to begining of unused portion */
  size_t       curlen,maxlen;
} PetscViewer_String;

#undef __FUNCT__
#define __FUNCT__ "PetscViewerDestroy_String"
static PetscErrorCode PetscViewerDestroy_String(PetscViewer viewer)
{
  PetscViewer_String *vstr = (PetscViewer_String *)viewer->data;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscFree(vstr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscViewerStringSPrintf"
/*@C
    PetscViewerStringSPrintf - Prints information to a PetscViewer string.

    Logically Collective on PetscViewer (Hmmm, each processor maintains a separate string)

    Input Parameters:
+   v - a string PetscViewer, formed by PetscViewerStringOpen()
-   format - the format of the input

    Level: developer

    Fortran Note:
    This routine is not supported in Fortran.

   Concepts: printing^to string

.seealso: PetscViewerStringOpen()
@*/
PetscErrorCode  PetscViewerStringSPrintf(PetscViewer viewer,const char format[],...)
{
  va_list            Argp;
  size_t             fullLength;
  size_t             shift;
  PetscErrorCode     ierr;
  PetscBool          isstring;
  char               tmp[4096];
  PetscViewer_String *vstr = (PetscViewer_String*)viewer->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  PetscValidCharPointer(format,2);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERSTRING,&isstring);CHKERRQ(ierr);
  if (!isstring) PetscFunctionReturn(0);
  if (!vstr->string) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Must call PetscViewerStringSetString() before using");

  va_start(Argp,format);
  ierr = PetscVSNPrintf(tmp,4096,format,&fullLength,Argp);CHKERRQ(ierr);
  va_end(Argp);

  ierr = PetscStrlen(tmp,&shift);CHKERRQ(ierr);
  if (shift >= vstr->maxlen - vstr->curlen - 1) shift = vstr->maxlen - vstr->curlen - 1;
  ierr = PetscStrncpy(vstr->head,tmp,shift);CHKERRQ(ierr);

  vstr->head   += shift;
  vstr->curlen += shift;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscViewerStringOpen"
/*@C
    PetscViewerStringOpen - Opens a string as a PetscViewer. This is a very
    simple PetscViewer; information on the object is simply stored into
    the string in a fairly nice way.

    Collective on MPI_Comm

    Input Parameters:
+   comm - the communicator
.   string - the string to use
-   len    - the string length

    Output Parameter:
.   lab - the PetscViewer

    Level: advanced

    Fortran Note:
    This routine is not supported in Fortran.

  Concepts: PetscViewerString^creating

.seealso: PetscViewerDestroy(), PetscViewerStringSPrintf()
@*/
PetscErrorCode  PetscViewerStringOpen(MPI_Comm comm,char string[],PetscInt len,PetscViewer *lab)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerCreate(comm,lab);CHKERRQ(ierr);
  ierr = PetscViewerSetType(*lab,PETSCVIEWERSTRING);CHKERRQ(ierr);
  ierr = PetscViewerStringSetString(*lab,string,len);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscViewerGetSingleton_String"
PetscErrorCode PetscViewerGetSingleton_String(PetscViewer viewer,PetscViewer *sviewer)
{
  PetscViewer_String *vstr = (PetscViewer_String*)viewer->data;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscViewerStringOpen(PETSC_COMM_SELF,vstr->head,vstr->maxlen-vstr->curlen,sviewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscViewerRestoreSingleton_String"
PetscErrorCode PetscViewerRestoreSingleton_String(PetscViewer viewer,PetscViewer *sviewer)
{
  PetscErrorCode     ierr;
  PetscViewer_String *iviewer = (PetscViewer_String*)(*sviewer)->data;
  PetscViewer_String *vstr = (PetscViewer_String*)viewer->data;

  PetscFunctionBegin;
  vstr->head    = iviewer->head;
  vstr->curlen += iviewer->curlen;
  ierr = PetscViewerDestroy(sviewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PetscViewerCreate_String"
PetscErrorCode  PetscViewerCreate_String(PetscViewer v)
{
  PetscViewer_String *vstr;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  v->ops->destroy          = PetscViewerDestroy_String;
  v->ops->view             = 0;
  v->ops->flush            = 0;
  v->ops->getsingleton     = PetscViewerGetSingleton_String;
  v->ops->restoresingleton = PetscViewerRestoreSingleton_String;
  ierr                     = PetscNewLog(v,PetscViewer_String,&vstr);CHKERRQ(ierr);
  v->data                  = (void*)vstr;
  vstr->string             = 0;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "PetscViewerStringSetString"
/*@C

   PetscViewerStringSetString - sets the string that a string viewer will print to

   Logically Collective on PetscViewer

  Input Parameters:
+   viewer - string viewer you wish to attach string to
.   string - the string to print data into
-   len - the length of the string

  Level: advanced

.seealso: PetscViewerStringOpen()
@*/
PetscErrorCode  PetscViewerStringSetString(PetscViewer viewer,char string[],PetscInt len)
{
  PetscViewer_String *vstr = (PetscViewer_String*)viewer->data;
  PetscErrorCode     ierr;
  PetscBool          isstring;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  PetscValidCharPointer(string,2);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERSTRING,&isstring);CHKERRQ(ierr);
  if (!isstring)  PetscFunctionReturn(0);
  if (len <= 2) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"String must have length at least 2");

  ierr = PetscMemzero(string,len*sizeof(char));CHKERRQ(ierr);
  vstr->string      = string;
  vstr->head        = string;
  vstr->curlen      = 0;
  vstr->maxlen      = len;
  PetscFunctionReturn(0);
}






