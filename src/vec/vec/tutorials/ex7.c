
static char help[] = "Demonstrates calling a Fortran computational routine from C.\n\
Also demonstrates passing  PETSc objects, MPI Communicators from C to Fortran\n\
and from Fortran to C\n\n";

#include <petscvec.h>
/*
  Ugly stuff to insure the function names match between Fortran
  and C. This is out of our PETSc hands to cleanup.
*/
/*T
   Concepts: vectors^fortran-c;
   Processors: n
T*/
#include <petsc/private/fortranimpl.h>
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define ex7f_ EX7F
#define ex7c_ EX7C
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define ex7f_ ex7f
#define ex7c_ ex7c
#endif

PETSC_INTERN void ex7f_(Vec*,int*);

int main(int argc,char **args)
{
  PetscInt m = 10;
  int      fcomm;
  Vec      vec;

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  /* This function should be called to be able to use PETSc routines
     from the FORTRAN subroutines needed by this program */

  PetscInitializeFortran();

  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&vec));
  CHKERRQ(VecSetSizes(vec,PETSC_DECIDE,m));
  CHKERRQ(VecSetFromOptions(vec));

  /*
     Call Fortran routine - the use of MPI_Comm_c2f() allows
     translation of the MPI_Comm from C so that it can be properly
     interpreted from Fortran.
  */
  fcomm = MPI_Comm_c2f(PETSC_COMM_WORLD);

  ex7f_(&vec,&fcomm);

  CHKERRQ(VecView(vec,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(VecDestroy(&vec));
  CHKERRQ(PetscFinalize());
  return 0;
}

PETSC_INTERN void ex7c_(Vec *fvec,int *fcomm,PetscErrorCode *ierr)
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

/*TEST

   build:
     depends: ex7f.F
     requires: fortran

   test:
      nsize: 3
      filter: sort -b |grep -v "MPI processes"
      filter_output: sort -b

TEST*/
