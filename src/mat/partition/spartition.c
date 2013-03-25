
#include <petscmat.h>

PETSC_EXTERN PetscErrorCode MatPartitioningCreate_Current(MatPartitioning);
PETSC_EXTERN PetscErrorCode MatPartitioningCreate_Square(MatPartitioning);
PETSC_EXTERN PetscErrorCode MatPartitioningCreate_Parmetis(MatPartitioning);
PETSC_EXTERN PetscErrorCode MatPartitioningCreate_Chaco(MatPartitioning);
PETSC_EXTERN PetscErrorCode MatPartitioningCreate_Party(MatPartitioning);
PETSC_EXTERN PetscErrorCode MatPartitioningCreate_PTScotch(MatPartitioning);

#undef __FUNCT__
#define __FUNCT__ "MatPartitioningRegisterAll"
/*@C
  MatPartitioningRegisterAll - Registers all of the matrix Partitioning routines in PETSc.

  Not Collective

  Level: developer

  Adding new methods:
  To add a new method to the registry. Copy this routine and
  modify it to incorporate a call to MatPartitioningRegister() for
  the new method, after the current list.

  Restricting the choices: To prevent all of the methods from being
  registered and thus save memory, copy this routine and modify it to
  register a zero, instead of the function name, for those methods you
  do not wish to register.  Make sure that the replacement routine is
  linked before libpetscmat.a.

.keywords: matrix, Partitioning, register, all

.seealso: MatPartitioningRegister(), MatPartitioningRegisterDestroy()
@*/
PetscErrorCode  MatPartitioningRegisterAll(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  MatPartitioningRegisterAllCalled = PETSC_TRUE;

  ierr = MatPartitioningRegister(MATPARTITIONINGCURRENT,path,"MatPartitioningCreate_Current",MatPartitioningCreate_Current);CHKERRQ(ierr);
  ierr = MatPartitioningRegister("square",path,"MatPartitioningCreate_Square",MatPartitioningCreate_Square);CHKERRQ(ierr);
#if defined(PETSC_HAVE_PARMETIS)
  ierr = MatPartitioningRegister(MATPARTITIONINGPARMETIS,path,"MatPartitioningCreate_Parmetis",MatPartitioningCreate_Parmetis);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_CHACO)
  ierr = MatPartitioningRegister(MATPARTITIONINGCHACO,path,"MatPartitioningCreate_Chaco",MatPartitioningCreate_Chaco);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_PARTY)
  ierr = MatPartitioningRegister(MATPARTITIONINGPARTY,path,"MatPartitioningCreate_Party",MatPartitioningCreate_Party);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_PTSCOTCH)
  ierr = MatPartitioningRegister(MATPARTITIONINGPTSCOTCH,path,"MatPartitioningCreate_PTScotch",MatPartitioningCreate_PTScotch);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}



