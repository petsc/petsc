/*$Id: spartition.c,v 1.23 2001/03/23 23:22:49 balay Exp $*/
 
#include "petsc.h"
#include "petscmat.h"

EXTERN_C_BEGIN
EXTERN int MatPartitioningCreate_Current(MatPartitioning);
EXTERN int MatPartitioningCreate_Square(MatPartitioning);
EXTERN int MatPartitioningCreate_Parmetis(MatPartitioning);
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatPartitioningRegisterAll" 
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
int MatPartitioningRegisterAll(const char path[])
{
  int         ierr;

  PetscFunctionBegin;
  ierr = MatPartitioningRegisterDynamic(MAT_PARTITIONING_CURRENT,path,"MatPartitioningCreate_Current",MatPartitioningCreate_Current);CHKERRQ(ierr);
  ierr = MatPartitioningRegisterDynamic("square",path,"MatPartitioningCreate_Square",MatPartitioningCreate_Square);CHKERRQ(ierr);
#if defined(PETSC_HAVE_PARMETIS)
  ierr = MatPartitioningRegisterDynamic(MAT_PARTITIONING_PARMETIS,path,"MatPartitioningCreate_Parmetis",MatPartitioningCreate_Parmetis);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}



