#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dlregis.c,v 1.3 1999/02/03 04:29:01 bsmith Exp bsmith $";
#endif

#include "petsc.h"

extern int DLLibraryRegister_Petsc(char *);
extern int DLLibraryInfo_Petsc(char *,char *,char **);

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "DLLibraryRegister"
/*
  DLLibraryRegister - This function is called when the dynamic library it is in is opened.


  Input Parameter:
.  path - library path
 */
int DLLibraryRegister(char *path)
{
  int ierr;

  PetscFunctionBegin;
  ierr =  DLLibraryRegister_Petsc(path);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END


/* --------------------------------------------------------------------------*/
EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "DLLibraryInfo"
int DLLibraryInfo(char *path,char *type,char **mess) 
{ 
  int ierr;

  PetscFunctionBegin;
  ierr = DLLibraryInfo_Petsc(path,type,mess);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
