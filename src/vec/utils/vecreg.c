#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: vecreg.c,v 1.1 1998/06/10 17:43:23 bsmith Exp bsmith $";
#endif

#include "src/vec/vecimpl.h"  /*I "vec.h" I*/

extern int VecCreateMPI(MPI_Comm,int,int,Vec *);
extern int VecCreateShared(MPI_Comm,int,int,Vec *);
extern int VecCreateSeq_Stub(MPI_Comm,int,int,Vec *);
  

/*
    This is used by VecCreate() to make sure that at least one 
    VecRegisterAll() is called. In general, if there is more than one
    DLL, then VecRegisterAll() may be called several times.
*/
extern int VecRegisterAllCalled;

#undef __FUNC__  
#define __FUNC__ "VecRegisterAll"
/*@C
  VecRegisterAll - Registers all of the Vec components in the PETSc package.

  Not Collective

.keywords: Vec, register, all

.seealso:  VecRegisterDestroy()
@*/
int VecRegisterAll(char *path)
{
  int ierr;

  PetscFunctionBegin;
  VecRegisterAllCalled = 1;

  ierr = VecRegister("PETSc#VecMPI",    path,"VecCreateMPI",     VecCreateMPI);CHKERRQ(ierr);
  ierr = VecRegister("PETSc#VecShared", path,"VecCreateShared",  VecCreateShared);CHKERRQ(ierr);
  ierr = VecRegister("PETSc#VecSeq",    path,"VecCreateSeq_Stub",VecCreateSeq_Stub);CHKERRQ(ierr);

  ierr = VecRegister("mpi",             path,"VecCreateMPI",     VecCreateMPI);CHKERRQ(ierr);
  ierr = VecRegister("shared",          path,"VecCreateShared",  VecCreateShared);CHKERRQ(ierr);
  ierr = VecRegister("seq",             path,"VecCreateSeq_Stub",VecCreateSeq_Stub);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
