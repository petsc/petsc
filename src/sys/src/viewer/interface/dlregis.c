#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dlregis.c,v 1.2 1999/01/27 19:49:10 bsmith Exp bsmith $";
#endif

#include "petsc.h"

extern int DLLibraryRegister_Petsc(char *);
extern int DLLibraryInfo_Petsc(char *,char *,char **);

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "DLLibraryRegister"
/*
  DLLibraryRegister - This function is called when the dynamic library it is in is opened.

  This one registers all the KSP and PC methods that are in the basic PETSc libpetscsles
  library.

  Input Parameter:
  path - library path
 */
int DLLibraryRegister(char *path)
{
  return DLLibraryRegister_Petsc(path);
}
EXTERN_C_END


/* --------------------------------------------------------------------------*/
EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "DLLibraryInfo"
int DLLibraryInfo(char *path,char *type,char **mess) 
{ 
  return DLLibraryInfo_Petsc(path,type,mess);
}
EXTERN_C_END
