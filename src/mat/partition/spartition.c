
#include <petscmat.h>

EXTERN_C_BEGIN
extern PetscErrorCode  MatPartitioningCreate_Current(MatPartitioning);
extern PetscErrorCode  MatPartitioningCreate_Square(MatPartitioning);
extern PetscErrorCode  MatPartitioningCreate_Parmetis(MatPartitioning);
extern PetscErrorCode  MatPartitioningCreate_Chaco(MatPartitioning);
extern PetscErrorCode  MatPartitioningCreate_Party(MatPartitioning);
extern PetscErrorCode  MatPartitioningCreate_PTScotch(MatPartitioning);
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
PetscErrorCode  MatPartitioningRegisterAll(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  MatPartitioningRegisterAllCalled = PETSC_TRUE;
  ierr = MatPartitioningRegisterDynamic(MATPARTITIONINGCURRENT,path,"MatPartitioningCreate_Current",MatPartitioningCreate_Current);CHKERRQ(ierr);
  ierr = MatPartitioningRegisterDynamic("square",path,"MatPartitioningCreate_Square",MatPartitioningCreate_Square);CHKERRQ(ierr);
#if defined(PETSC_HAVE_PARMETIS)
  ierr = MatPartitioningRegisterDynamic(MATPARTITIONINGPARMETIS,path,"MatPartitioningCreate_Parmetis",MatPartitioningCreate_Parmetis);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_CHACO)
  ierr = MatPartitioningRegisterDynamic(MATPARTITIONINGCHACO,path,"MatPartitioningCreate_Chaco",MatPartitioningCreate_Chaco);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_PARTY)
  ierr = MatPartitioningRegisterDynamic(MATPARTITIONINGPARTY,path,"MatPartitioningCreate_Party",MatPartitioningCreate_Party);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_PTSCOTCH)
  ierr = MatPartitioningRegisterDynamic(MATPARTITIONINGPTSCOTCH,path,"MatPartitioningCreate_PTScotch",MatPartitioningCreate_PTScotch);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}



