/*$Id: spartition.c,v 1.15 1999/11/05 14:46:04 bsmith Exp bsmith $*/
 
#include "petsc.h"
#include "mat.h"

EXTERN_C_BEGIN
extern int MatPartitioningCreate_Current(MatPartitioning);
extern int MatPartitioningCreate_Parmetis(MatPartitioning);
EXTERN_C_END

#undef __FUNC__  
#define __FUNC__ "MatPartitioningRegisterAll" 
/*@C
  MatPartitioningRegisterAll - Registers all of the matrix Partitioning routines in PETSc.

  Not Collective

  Level: developer

  Adding new methods:
  To add a new method to the registry. Copy this routine and 
  modify it to incorporate a call to MatPartitioningRegisterDynamic() for 
  the new method, after the current list.

  Restricting the choices: To prevent all of the methods from being
  registered and thus save memory, copy this routine and modify it to
  register a zero, instead of the function name, for those methods you
  do not wish to register.  Make sure that the replacement routine is
  linked before libpetscmat.a.

.keywords: matrix, Partitioning, register, all

.seealso: MatPartitioningRegisterDynamic(), MatPartitioningRegisterDestroy()
@*/
int MatPartitioningRegisterAll(char *path)
{
  int         ierr;

  PetscFunctionBegin;
  ierr = MatPartitioningRegisterDynamic(MATPARTITIONING_CURRENT,path,"MatPartitioningCreate_Current",MatPartitioningCreate_Current);CHKERRQ(ierr);
#if defined(PETSC_HAVE_PARMETIS)
  ierr = MatPartitioningRegisterDynamic(MATPARTITIONING_PARMETIS,path,"MatPartitioningCreate_Parmetis",MatPartitioningCreate_Parmetis);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}



