/*$Id: ex7.c,v 1.24 1999/05/12 03:28:40 bsmith Exp bsmith $*/

static char help[] = "Demonstrates calling a Fortran computational routine from C.\n\
Also demonstrates passing  PETSc objects, MPI Communicators from C to Fortran\n\
and from Fortran to C\n\n";

#include "vec.h"

/*
  Ugly stuff to insure the function names match between Fortran 
  and C. Sorry, but this is out of our PETSc hands to cleanup.
*/
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define ex7f_ EX7F
#define ex7c_ EX7C
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define ex7f_ ex7f
#define ex7c_ ex7c
#endif
EXTERN_C_BEGIN
extern void ex7f_(Vec *,int*);
EXTERN_C_END

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  int              ierr, m = 10;
  int              fcomm;
  Vec              vec;

  PetscInitialize(&argc,&args,(char *)0,help);

  /* This function should be called to be able to use PETSc routines
     from the FORTRAN subroutines needed by this program */

  PetscInitializeFortran();  

  ierr = VecCreate(PETSC_COMM_WORLD,PETSC_DECIDE,m,&vec);CHKERRA(ierr);
  ierr = VecSetFromOptions(vec);CHKERRA(ierr);

  /* 
     Call Fortran routine - the use of MPICCommToFortranComm() allows 
     translation of the MPI_Comm from C so that it can be properly 
     interpreted from Fortran.
  */
  ierr = MPICCommToFortranComm(PETSC_COMM_WORLD,&fcomm);CHKERRA(ierr);

  ex7f_(&vec,&fcomm);

  ierr = VecView(vec,VIEWER_STDOUT_WORLD);CHKERRA(ierr);
  ierr = VecDestroy(vec);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}

EXTERN_C_BEGIN
#undef __FUNC__
#define __FUNC__ "ex7c_"
int ex7c_(Vec *fvec, int *fcomm)
{
  MPI_Comm comm;
  int ierr,size;

  /*
    Translate Fortran integer pointer back to C and
    Fortran Communicator back to C communicator
  */
  ierr = MPIFortranCommToCComm(*fcomm,&comm);
  
  /*
    Some PETSc/MPI operations on Vec/Communicator objects 
  */
  ierr = VecGetSize(*fvec,&size);CHKERRA(ierr);
  MPI_Barrier(comm);
  
  return 0;
}
EXTERN_C_END
