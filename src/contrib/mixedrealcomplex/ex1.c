#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex1.c,v 1.1 1999/04/22 21:25:46 bsmith Exp bsmith $";
#endif

/* Program usage:  mpiexec ex1 [-help] [all PETSc options] */

static char help[] = "Demonstrates various vector routines.\n\n";

#include "vec.h"

int MyInit()
{
  int ierr;

  /* 
     Initialized the global complex variable; this is because with 
     shared libraries the constructors for global variables
     are not called; at least on IRIX.
  */
  {
    Scalar ic(0.0,1.0);
    PETSC_i = ic; 
  }
  ierr = MPI_Type_contiguous(2,MPI_DOUBLE,&MPIU_COMPLEX);CHKERRQ(ierr);
  ierr = MPI_Type_commit(&MPIU_COMPLEX);CHKERRQ(ierr);
  return 0;
}

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  Vec      x, y, w;               /* vectors */
  int      ierr;
  double   *array;

  PetscInitialize(&argc,&argv,0,help);

  ierr = VecCreate(PETSC_COMM_WORLD,10,PETSC_DETERMINE,&x);CHKERRA(ierr);
  ierr = VecSetType(x,"/tmp/bsmith/petsc/lib/libg_c++/solaris/libpetscvec:VecCreate_Seq");CHKERRA(ierr);
  ierr = VecView(x,VIEWER_STDOUT_WORLD);CHKERRA(ierr);

  ierr = VecGetArray(x,(Scalar**) &array);CHKERRA(ierr);
  
  ierr = VecDestroy(x);
  
  PetscFinalize();
  return 0;
}
 
MPI_Datatype  MPIU_COMPLEX;

