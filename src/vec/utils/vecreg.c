#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: vecreg.c,v 1.2 1998/06/11 19:54:45 bsmith Exp bsmith $";
#endif

#include "src/vec/vecimpl.h"  /*I "vec.h" I*/

/*
    We need these stubs since with C++ we need to compile with 
  extern "C"
*/
EXTERN_C_BEGIN
int VecCreate_Seq(MPI_Comm comm, int n, int N, Vec *v)
{
  return VecCreateSeq(comm,PetscMax(n,N),v);
}
int VecCreate_MPI(MPI_Comm comm, int n, int N, Vec *v)
{
  return VecCreateMPI(comm,n,N,v);
}
int VecCreate_Shared(MPI_Comm comm, int n, int N, Vec *v)
{
  return VecCreateShared(comm,n,N,v);
}
EXTERN_C_END



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

  ierr = VecRegister("PETSc#VecMPI",    path,"VecCreate_MPI",     VecCreate_MPI);CHKERRQ(ierr);
  ierr = VecRegister("PETSc#VecShared", path,"VecCreate_Shared",  VecCreate_Shared);CHKERRQ(ierr);
  ierr = VecRegister("PETSc#VecSeq",    path,"VecCreate_Seq",VecCreate_Seq);CHKERRQ(ierr);

  ierr = VecRegister("mpi",             path,"VecCreate_MPI",     VecCreate_MPI);CHKERRQ(ierr);
  ierr = VecRegister("shared",          path,"VecCreate_Shared",  VecCreate_Shared);CHKERRQ(ierr);
  ierr = VecRegister("seq",             path,"VecCreate_Seq",VecCreate_Seq);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
