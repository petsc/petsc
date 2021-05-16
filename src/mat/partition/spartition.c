
#include <petscmat.h>
#include <petsc/private/matimpl.h>

PETSC_EXTERN PetscErrorCode MatPartitioningCreate_Current(MatPartitioning);
PETSC_EXTERN PetscErrorCode MatPartitioningCreate_Average(MatPartitioning part);
PETSC_EXTERN PetscErrorCode MatPartitioningCreate_Square(MatPartitioning);
PETSC_EXTERN PetscErrorCode MatPartitioningCreate_Parmetis(MatPartitioning);
PETSC_EXTERN PetscErrorCode MatPartitioningCreate_Hierarchical(MatPartitioning);
#if defined(PETSC_HAVE_CHACO)
PETSC_EXTERN PetscErrorCode MatPartitioningCreate_Chaco(MatPartitioning);
#endif
#if defined(PETSC_HAVE_PARTY)
PETSC_EXTERN PetscErrorCode MatPartitioningCreate_Party(MatPartitioning);
#endif
#if defined(PETSC_HAVE_PTSCOTCH)
PETSC_EXTERN PetscErrorCode MatPartitioningCreate_PTScotch(MatPartitioning);
#endif

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

.seealso: MatPartitioningRegister(), MatPartitioningRegisterDestroy()
@*/
PetscErrorCode  MatPartitioningRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (MatPartitioningRegisterAllCalled) PetscFunctionReturn(0);
  MatPartitioningRegisterAllCalled = PETSC_TRUE;

  ierr = MatPartitioningRegister(MATPARTITIONINGCURRENT, MatPartitioningCreate_Current);CHKERRQ(ierr);
  ierr = MatPartitioningRegister(MATPARTITIONINGAVERAGE, MatPartitioningCreate_Average);CHKERRQ(ierr);
  ierr = MatPartitioningRegister(MATPARTITIONINGSQUARE,  MatPartitioningCreate_Square);CHKERRQ(ierr);
  ierr = MatPartitioningRegister(MATPARTITIONINGHIERARCH,MatPartitioningCreate_Hierarchical);CHKERRQ(ierr);
#if defined(PETSC_HAVE_PARMETIS)
  ierr = MatPartitioningRegister(MATPARTITIONINGPARMETIS,MatPartitioningCreate_Parmetis);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_CHACO)
  ierr = MatPartitioningRegister(MATPARTITIONINGCHACO,   MatPartitioningCreate_Chaco);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_PARTY)
  ierr = MatPartitioningRegister(MATPARTITIONINGPARTY,   MatPartitioningCreate_Party);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_PTSCOTCH)
  ierr = MatPartitioningRegister(MATPARTITIONINGPTSCOTCH,MatPartitioningCreate_PTScotch);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

