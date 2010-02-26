
static char help[] = "Computes the integral of 2*x/(1+x^2) from x=0..1 \nThis is equal to the ln(2).\n\n";

/*T
   Concepts: vectors^assembling vectors;
   Processors: n

   Contributed by Mike McCourt <mccomic@iit.edu> and Nathan Johnston <johnnat@iit.edu>
T*/

/* 
  Include "petscvec.h" so that we can use vectors.  Note that this file
  automatically includes:
     petscsys.h       - base PETSc routines   petscis.h     - index sets
     petscviewer.h - viewers
*/
#include "petscvec.h"

PetscScalar func(PetscScalar a){return 2*a/(1+a*a);}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank,nproc;
  PetscInt       rstart,rend,i,k,N,numPoints=1000000;
  PetscScalar    dummy,result=0,h=1.0/numPoints,*xarray;
  Vec            x,xend;

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&nproc);CHKERRQ(ierr);

  /*
     Create a parallel vector.
       Here we set up our x vector which will be given values below.
       The xend vector is a dummy vector to find the value of the
         elements at the endpoints for use in the trapezoid rule.
  */
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,numPoints);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecGetSize(x,&N);CHKERRQ(ierr);
  ierr = VecSet(x,result);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&xend);CHKERRQ(ierr);
  result=0.5;
  if (rank == 0){
    i=0;
    ierr = VecSetValues(xend,1,&i,&result,INSERT_VALUES);CHKERRQ(ierr);
  } else if (rank == nproc){
    i=N-1;
    ierr = VecSetValues(xend,1,&i,&result,INSERT_VALUES);CHKERRQ(ierr);
  }
  /* 
     Assemble vector, using the 2-step process:
       VecAssemblyBegin(), VecAssemblyEnd()
     Computations can be done while messages are in transition
     by placing code between these two statements.
  */
  ierr = VecAssemblyBegin(xend);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(xend);CHKERRQ(ierr);

  /*
     Set the x vector elements.
      i*h will return 0 for i=0 and 1 for i=N-1.
      The function evaluated (2x/(1+x^2)) is defined above.
      Each evaluation is put into the local array of the vector without message passing.
  */
  ierr = VecGetOwnershipRange(x,&rstart,&rend);CHKERRQ(ierr);
  ierr = VecGetArray(x,&xarray);CHKERRQ(ierr);
  k = 0;
  for (i=rstart; i<rend; i++){
    xarray[k] = i*h;
    xarray[k] = func(xarray[k]);
    k++;
  }
  ierr = VecRestoreArray(x,&xarray);CHKERRQ(ierr);

  /*
     Evaluates the integral.  First the sum of all the points is taken.
     That result is multiplied by the step size for the trapezoid rule.
     Then half the value at each endpoint is subtracted,
	this is part of the composite trapezoid rule.
  */
  ierr = VecSum(x,&result);CHKERRQ(ierr);
  result = result*h;
  ierr   = VecDot(x,xend,&dummy);CHKERRQ(ierr);
  result = result-h*dummy;   

  /*
      Return the value of the integral.
  */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"ln(2) is %A\n",result);CHKERRQ(ierr);
  ierr = VecDestroy(x);CHKERRQ(ierr);
  ierr = VecDestroy(xend);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
 
