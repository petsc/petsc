
#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: spartition.c,v 1.4 1997/10/28 14:23:24 bsmith Exp bsmith $";
#endif
 
#include "petsc.h"
#include "mat.h"

extern int PartitioningCreate_Current(Partitioning);
extern int PartitioningCreate_Parmetis(Partitioning);

#undef __FUNC__  
#define __FUNC__ "PartitioningRegisterAll" 
/*@C
  PartitioningRegisterAll - Registers all of the matrix Partitioning routines in PETSc.

  Adding new methods:
  To add a new method to the registry. Copy this routine and 
  modify it to incorporate a call to PartitioningRegister() for 
  the new method, after the current list.

  Restricting the choices: To prevent all of the methods from being
  registered and thus save memory, copy this routine and modify it to
  register a zero, instead of the function name, for those methods you
  do not wish to register.  Make sure that the replacement routine is
  linked before libpetscmat.a.

.keywords: matrix, Partitioning, register, all

.seealso: PartitioningRegister(), PartitioningRegisterDestroy()
@*/
int PartitioningRegisterAll()
{
  int         ierr;

  PetscFunctionBegin;
  PartitioningRegisterAllCalled = 1;  
  ierr = PartitioningRegister(PARTITIONING_CURRENT,0,"current",PartitioningCreate_Current);CHKERRQ(ierr);
#if defined(HAVE_PARMETIS)
  ierr = PartitioningRegister(PARTITIONING_PARMETIS,0,"parmetis",PartitioningCreate_Parmetis);CHKERRQ(ierr);
#endif

  PetscFunctionReturn(0);
}



