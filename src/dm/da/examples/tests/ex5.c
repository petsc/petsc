#ifndef lint
static char vcid[] = "$Id: ex6.c,v 1.3 1995/08/22 02:35:32 curfman Exp $";
#endif

/* This file created by Peter Mell   6/30/95 */ 

static char help[] = "This example simulates a heat equation.\n\n";

#include "petsc.h"
#include "da.h"
#include "sys.h"
#include "draw.h"
#include <math.h>
#include <sysio.h>

#define PI 3.14159265

int main(int argc,char **argv)
{
  int       mytid, numtid, M = 14, ierr, time_steps = 1000, w=1, s=1, a=1;
  DA        da;
  DrawCtx   win;
  Vec       local, global, copy;
  Scalar    *localptr, *copyptr;
  double    h,k;
  int       localsize, j, i, mybase, myend;
 
  PetscInitialize(&argc,&argv,(char*)0,(char*)0);
  if (OptionsHasName(0,"-help")) fprintf(stderr,"%s",help);

  OptionsGetInt(0,"-M",&M);
  OptionsGetInt(0,"-time",&time_steps);
    
  /* Set up the array */ 
  ierr = DACreate1d(MPI_COMM_WORLD,DA_NONPERIODIC,M,w,s,&da); 
  CHKERRA(ierr);
  ierr = DAGetDistributedVector(da,&global); CHKERRQ(ierr);
  ierr = DAGetLocalVector(da,&local); CHKERRQ(ierr);
  MPI_Comm_rank(MPI_COMM_WORLD,&mytid);
  MPI_Comm_size(MPI_COMM_WORLD,&numtid); 

  /* Make copy of local array for doing updates */
  ierr = VecDuplicate(local,&copy); CHKERRA(ierr);
  ierr = VecGetArray (copy,&copyptr); CHKERRA(ierr);

  /* Set Up Display to Show Heat Graph */
  ierr = DrawOpenX(MPI_COMM_WORLD,0,"",80,480,500,160,&win); CHKERRA(ierr);
  ierr = DrawSetDoubleBuffer(win); CHKERRQ(ierr);

  /* determine starting point of each processor */
  ierr = VecGetOwnershipRange(global,&mybase,&myend); CHKERRA(ierr);

  /* Initialize the Array */
  ierr = VecGetLocalSize (local,&localsize); CHKERRA(ierr);
  ierr = VecGetArray (local,&localptr);  CHKERRA(ierr);
  localptr[0] = copyptr[0] = 0.0;
  localptr[localsize-1] = copyptr[localsize-1] = 1.0;
  for (i=1; i<localsize-1; i++) {
    j=(i-1)+mybase; 
    localptr[i] = sin( (PI*j*6)/((double)M) 
                        + 1.2 * sin( (PI*j*2)/((double)M) ) ) * 4+4;
  }

  ierr = VecRestoreArray(local,&localptr); CHKERRA(ierr);
  ierr = DALocalToGlobal(da,local,INSERTVALUES,global); CHKERRQ(ierr);

  /* Assign Parameters */
  a=1;
  h= 1.0/M; 
  k= h*h/2.2;

  for (j=0; j<time_steps; j++) {  

    /* Global to Local */
    ierr = DAGlobalToLocalBegin(da,global,INSERTVALUES,local); CHKERRQ(ierr);
    ierr = DAGlobalToLocalEnd(da,global,INSERTVALUES,local); CHKERRQ(ierr);

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
    ierr = DALocalToGlobal(da,copy,INSERTVALUES,global); CHKERRQ(ierr);
  
    /* View Wave */ 
    ierr = VecView(global,(Viewer) win);  CHKERRA(ierr);

  }

  ierr = DADestroy(da); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
 



