#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: spartition.c,v 1.9 1998/10/19 22:18:30 bsmith Exp bsmith $";
#endif
 
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
int MatPartitioningRegisterAll(void)
{
  int         ierr;

  PetscFunctionBegin;
  MatPartitioningRegisterAllCalled = 1;  
  ierr = MatPartitioningRegister(MATPARTITIONING_CURRENT,0,"current",MatPartitioningCreate_Current);CHKERRQ(ierr);
#if defined(HAVE_PARMETIS)
  ierr = MatPartitioningRegister(MATPARTITIONING_PARMETIS,0,"parmetis",MatPartitioningCreate_Parmetis);CHKERRQ(ierr);
#endif

  PetscFunctionReturn(0);
}



