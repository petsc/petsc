/*$Id: ex5.c,v 1.33 1999/10/24 14:04:09 bsmith Exp bsmith $*/

/* This file created by Peter Mell   6/30/95 */ 

static char help[] = "Solves the one dimensional heat equation.\n\n";

#include "da.h"
#include "sys.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int       rank, size, M = 14, ierr, time_steps = 1000, w=1, s=1;
  DA        da;
  Viewer    viewer;
  Draw      draw;
  Vec       local, global, copy;
  Scalar    *localptr, *copyptr;
  double    h,k;
  int       localsize, j, i, mybase, myend;
 
  PetscInitialize(&argc,&argv,(char*)0,help);

  ierr = OptionsGetInt(PETSC_NULL,"-M",&M,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-time",&time_steps,PETSC_NULL);CHKERRA(ierr);
    
  /* Set up the array */ 
  ierr = DACreate1d(PETSC_COMM_WORLD,DA_NONPERIODIC,M,w,s,PETSC_NULL,&da);CHKERRA(ierr);
  ierr = DACreateGlobalVector(da,&global);CHKERRA(ierr);
  ierr = DACreateLocalVector(da,&local);CHKERRA(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRA(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRA(ierr);

  /* Make copy of local array for doing updates */
  ierr = VecDuplicate(local,&copy);CHKERRA(ierr);

  /* Set Up Display to Show Heat Graph */
  ierr = ViewerDrawOpen(PETSC_COMM_WORLD,0,"",80,480,500,160,&viewer);CHKERRA(ierr);
  ierr = ViewerDrawGetDraw(viewer,0,&draw);CHKERRA(ierr);
  ierr = DrawSetDoubleBuffer(draw);CHKERRA(ierr);

  /* determine starting point of each processor */
  ierr = VecGetOwnershipRange(global,&mybase,&myend);CHKERRA(ierr);

  /* Initialize the Array */
  ierr = VecGetLocalSize (local,&localsize);CHKERRA(ierr);
  ierr = VecGetArray (local,&localptr); CHKERRA(ierr);
  ierr = VecGetArray (copy,&copyptr);CHKERRA(ierr);
  localptr[0] = copyptr[0] = 0.0;
  localptr[localsize-1] = copyptr[localsize-1] = 1.0;
  for (i=1; i<localsize-1; i++) {
    j=(i-1)+mybase; 
    localptr[i] = sin( (PETSC_PI*j*6)/((double)M) 
                        + 1.2 * sin( (PETSC_PI*j*2)/((double)M) ) ) * 4+4;
  }

  ierr = VecRestoreArray(local,&localptr);CHKERRA(ierr);
  ierr = VecRestoreArray(copy,&copyptr);CHKERRA(ierr);
  ierr = DALocalToGlobal(da,local,INSERT_VALUES,global);CHKERRA(ierr);

  /* Assign Parameters */
  h= 1.0/M; 
  k= h*h/2.2;

  for (j=0; j<time_steps; j++) {  

    /* Global to Local */
    ierr = DAGlobalToLocalBegin(da,global,INSERT_VALUES,local);CHKERRA(ierr);
    ierr = DAGlobalToLocalEnd(da,global,INSERT_VALUES,local);CHKERRA(ierr);

    /*Extract local array */ 
    ierr = VecGetArray(local,&localptr);CHKERRA(ierr);
    ierr = VecGetArray (copy,&copyptr);CHKERRA(ierr);

    /* Update Locally - Make array of new values */
    /* Note: I don't do anything for the first and last entry */
    for (i=1; i< localsize-1; i++) {
      copyptr[i] = localptr[i] + (k/(h*h)) *
                           (localptr[i+1]-2.0*localptr[i]+localptr[i-1]);
    }
  
    ierr = VecRestoreArray(copy,&copyptr);CHKERRA(ierr);
    ierr = VecRestoreArray(local,&localptr);CHKERRA(ierr);

    /* Local to Global */
    ierr = DALocalToGlobal(da,copy,INSERT_VALUES,global);CHKERRA(ierr);
  
    /* View Wave */ 
    ierr = VecView(global,viewer); CHKERRA(ierr);

  }

  ierr = ViewerDestroy(viewer);CHKERRA(ierr);
  ierr = VecDestroy(copy);CHKERRA(ierr);
  ierr = VecDestroy(local);CHKERRA(ierr);
  ierr = VecDestroy(global);CHKERRA(ierr);
  ierr = DADestroy(da);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
 



