#ifndef lint
static char vcid[] = "$Id: ex1.c,v 1.7 1996/03/21 16:43:27 bsmith Exp bsmith $";
#endif

/*
   Simple example to show how PETSc programs can be run from Matlab. 
  See ex1.m. This is the same program as src/da/examples/ex5.c
*/

static char help[] = "Solves the one dimensional heat equation.\n\n";

#include "petsc.h"
#include "da.h"
#include "sys.h"
#include "draw.h"
#include <math.h>

#define PETSC_PI 3.14159265

int main(int argc,char **argv)
{
  int       rank, size, M = 14, ierr, time_steps = 1000, w=1, s=1, a=1, flg;
  DA        da;
  Viewer    viewer;
  Vec       local, global, copy;
  Scalar    *localptr, *copyptr;
  double    h,k;
  int       localsize, j, i, mybase, myend;
 
  PetscInitialize(&argc,&argv,(char*)0,help);

  OptionsGetInt(PETSC_NULL,"-M",&M,&flg);
  OptionsGetInt(PETSC_NULL,"-time",&time_steps,&flg);
    
  /* Set up the array */ 
  ierr = DACreate1d(MPI_COMM_WORLD,DA_NONPERIODIC,M,w,s,&da); CHKERRA(ierr);
  ierr = DAGetDistributedVector(da,&global); CHKERRA(ierr);
  ierr = DAGetLocalVector(da,&local); CHKERRA(ierr);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size); 

  /* Make copy of local array for doing updates */
  ierr = VecDuplicate(local,&copy); CHKERRA(ierr);
  ierr = VecGetArray (copy,&copyptr); CHKERRA(ierr);

  /* Set Up Display to Show Heat Graph */
  ierr = ViewerMatlabOpen(MPI_COMM_WORLD,"eagle",0,&viewer); CHKERRA(ierr);

  /* determine starting point of each processor */
  ierr = VecGetOwnershipRange(global,&mybase,&myend); CHKERRA(ierr);

  /* Initialize the Array */
  ierr = VecGetLocalSize (local,&localsize); CHKERRA(ierr);
  ierr = VecGetArray (local,&localptr);  CHKERRA(ierr);
  localptr[0] = copyptr[0] = 0.0;
  localptr[localsize-1] = copyptr[localsize-1] = 1.0;
  for (i=1; i<localsize-1; i++) {
    j=(i-1)+mybase; 
    localptr[i] = sin( (PETSC_PI*j*6)/((double)M) 
                        + 1.2 * sin( (PETSC_PI*j*2)/((double)M) ) ) * 4+4;
  }

  ierr = VecRestoreArray(local,&localptr); CHKERRA(ierr);
  ierr = DALocalToGlobal(da,local,INSERT_VALUES,global); CHKERRA(ierr);

  /* Assign Parameters */
  a=1;
  h= 1.0/M; 
  k= h*h/2.2;

  for (j=0; j<time_steps; j++) {  

    /* Global to Local */
    ierr = DAGlobalToLocalBegin(da,global,INSERT_VALUES,local); CHKERRA(ierr);
    ierr = DAGlobalToLocalEnd(da,global,INSERT_VALUES,local); CHKERRA(ierr);

    /*Extract local array */ 
    ierr = VecGetArray(local,&localptr); CHKERRA(ierr);

    /* Update Locally - Make array of new values */
    /* Note: I don't do anything for the first and last entry */
    for (i=1; i< localsize-1; i++) {
      copyptr[i] = localptr[i] + (k/(h*h)) *
                           (localptr[i+1]-2*localptr[i]+localptr[i-1]);
    }
  
    ierr = VecRestoreArray(copy,&copyptr); CHKERRA(ierr);

    /* Local to Global */
    ierr = DALocalToGlobal(da,copy,INSERT_VALUES,global); CHKERRA(ierr);
  
    /* View Wave */ 
    ierr = VecView(global,viewer);  CHKERRA(ierr);

  }

  /*
     The sleep is so that the viewer port is not closed before Matlab is 
     finished reading.
  */
  PetscSleep(100);
  ierr = ViewerDestroy(viewer); CHKERRA(ierr);
  ierr = VecDestroy(copy); CHKERRA(ierr);
  ierr = VecDestroy(local); CHKERRA(ierr);
  ierr = VecDestroy(global); CHKERRA(ierr);
  ierr = DADestroy(da); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
 
