
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
  int       mytid, numtid, M = 14, ierr;
  int       w=1, s=1, a=1;
  DA        da;
  DrawCtx   win;
  Vec       local,global,copy;
  Scalar    value, *localptr, *copyptr;
  int       time_steps, mysize;
  double    h,k;
  int       localsize, j, i, mybase,myend;
  DrawLGCtx ctx;
  double    tempx, tempy;
 
  PetscInitialize(&argc,&argv,(char*)0,(char*)0);
  if (OptionsHasName(0,"-help")) fprintf(stderr,"%s",help);

  OptionsGetInt(0,"-M",&M);
    
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
  VecGetOwnershipRange(global,&mybase,&myend);

  /* Initialize the Array */
  VecGetLocalSize (local,&localsize);
  VecGetArray (local,&localptr); 
  localptr[0] = copyptr[0] = 0.0;
  localptr[localsize-1] = copyptr[localsize-1] = 1.0;
  for (i=1; i<localsize-1; i++) {
    j=(i-1)+mybase; 
    localptr[i] = sin( (PI*j*6)/((double)M) 
                        + 1.2 * sin( (PI*j*2)/((double)M) ) ) * 4+4;
  }

  VecRestoreArray(local,&localptr);
  ierr = DALocalToGlobal(da,local,INSERTVALUES,global); CHKERRQ(ierr);
  printf ("\nInitial Local Array [%d]\n",mytid);
  for (i=0; i< localsize; i++) { printf (" %f",localptr[i]); }
  printf ("\n\n");

  /* Assign Parameters */
  a=1;
  h= 1.0/M; 
  k= h*h/2.2;
  time_steps = 100000;

  for (j=0; j<time_steps; j++) {  

    /* Global to Local */
    ierr = DAGlobalToLocalBegin (da,global,INSERTVALUES,local); CHKERRQ(ierr);
    ierr = DAGlobalToLocalEnd   (da,global,INSERTVALUES,local); CHKERRQ(ierr);

    /*Extract local array */ 
    VecGetArray (local,&localptr); 

    /* Update Locally - Make array of new values */
    /* Note: I don't do anything for the first and last entry */
    for (i=1; i< localsize-1; i++) {
      copyptr[i] = localptr[i] + (k/(h*h)) *
                           (localptr[i+1]-2*localptr[i]+localptr[i-1]);
    }
  
    VecRestoreArray(copy,&copyptr);

    /* Local to Global */
    ierr = DALocalToGlobal(da,copy,INSERTVALUES,global); CHKERRQ(ierr);
  
    /* View Wave */ 
    VecView (global,(Viewer) win); 

  }

  DADestroy(da);
  PetscFinalize();
  return 0;
}
 



