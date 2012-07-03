
static char help[] = "Demonstrates calling a Fortran computational routine from C.\n\
Also demonstrates passing  PETSc objects, MPI Communicators from C to Fortran\n\
and from Fortran to C\n\n";

#include <petscvec.h>
/*
  Ugly stuff to insure the function names match between Fortran 
  and C. Sorry, but this is out of our PETSc hands to cleanup.
*/
#include <petsc-private/fortranimpl.h>
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define ex7f_ EX7F
#define ex7c_ EX7C
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define ex7f_ ex7f
#define ex7c_ ex7c
#endif
EXTERN_C_BEGIN
extern void PETSC_STDCALL ex7f_(Vec *,int*);
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  PetscErrorCode   ierr;
  PetscInt         m = 10;
  int              fcomm;
  Vec              vec;

  PetscInitialize(&argc,&args,(char *)0,help);

  /* This function should be called to be able to use PETSc routines
     from the FORTRAN subroutines needed by this program */

  PetscInitializeFortran();  

  ierr = VecCreate(PETSC_COMM_WORLD,&vec);CHKERRQ(ierr);
  ierr = VecSetSizes(vec,PETSC_DECIDE,m);CHKERRQ(ierr);
  ierr = VecSetFromOptions(vec);CHKERRQ(ierr);

  /* 
     Call Fortran routine - the use of MPI_Comm_c2f() allows
     translation of the MPI_Comm from C so that it can be properly 
     interpreted from Fortran.
  */
  fcomm = MPI_Comm_c2f(PETSC_COMM_WORLD);

  ex7f_(&vec,&fcomm);

  ierr = VecView(vec,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecDestroy(&vec);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "ex7c_"
void PETSC_STDCALL ex7c_(Vec *fvec,int *fcomm,PetscErrorCode* ierr)
{
  MPI_Comm comm;
  PetscInt vsize;

  /*
    Translate Fortran integer pointer back to C and
    Fortran Communicator back to C communicator
  */
  comm = MPI_Comm_f2c(*fcomm);
  
  /* Some PETSc/MPI operations on Vec/Communicator objects */
  *ierr = VecGetSize(*fvec,&vsize);
  *ierr = MPI_Barrier(comm);

}
EXTERN_C_END
