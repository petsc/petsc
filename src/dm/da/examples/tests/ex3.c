
/* This file created by Peter Mell   6/30/95 */ 

static char help[] = "This example creates a 1d wave.\n\n";

#include "petsc.h"
#include "da.h"
#include "sys.h"
#include "draw.h"
#include <math.h>
#include <sysio.h>

#define PI 3.14159265

int main(int argc,char **argv)
{
  int       mytid, numtid, M = 60, ierr;
  DA        da;
  DrawCtx   win;
  Vec       local,global,copy;
  Scalar    value, *localptr, *copyptr;
  int       time_steps, mysize;
  double    a,h,k;
  int       localsize, j, i, mybase,myend;
  DrawLGCtx ctx;
  double    tempx, tempy;
 
  PetscInitialize(&argc,&argv,(char*)0,(char*)0);
  if (OptionsHasName(0,"-help")) fprintf(stderr,"%s",help);

  OptionsGetInt(0,"-M",&M);
    
  /* Set up the array */ 
  ierr = DACreate1d(MPI_COMM_WORLD,M,1,1,DA_XPERIODIC,&da); CHKERRA(ierr);
  ierr = DAGetDistributedVector(da,&global); CHKERRQ(ierr);
  ierr = DAGetLocalVector(da,&local); CHKERRQ(ierr);
  MPI_Comm_rank(MPI_COMM_WORLD,&mytid);
  MPI_Comm_size(MPI_COMM_WORLD,&numtid); 

  /* Set Up Display to Show Wave Graph */
  ierr = DrawOpenX(MPI_COMM_WORLD,0,"",80,480,500,160,&win); CHKERRA(ierr);
  ierr = DrawSetDoubleBuffer(win); CHKERRQ(ierr);

  /* determine starting point of each processor */
  VecGetOwnershipRange(global,&mybase,&myend);

  /* Initialize the Array */
  VecGetLocalSize (local,&localsize);
  VecGetArray (local,&localptr); 
  localptr[0] = 0.0;
  localptr[localsize-1] = 0.0;
  for (i=1; i<localsize-1; i++) {
    j=(i-1)+mybase; 
    localptr[i] = sin( (PI*j*6)/((double)M) 
                        + 1.2 * sin( (PI*j*2)/((double)M) ) ) * 2;
  }

  VecRestoreArray(local,&localptr);
  ierr = DALocalToGlobal(da,local,INSERTVALUES,global); CHKERRQ(ierr);

  /* Make copy of local array for doing updates */
  ierr = VecDuplicate(local,&copy); CHKERRA(ierr);
  ierr = VecGetArray (copy,&copyptr); CHKERRA(ierr);

  /* Assign Parameters */
  a= 1.0;
  h= 1.0/M;
  k= h;
  time_steps = 100000;

  for (j=0; j<time_steps; j++) {  

    /* Global to Local */
    ierr = DAGlobalToLocalBegin(da,global,INSERTVALUES,local); CHKERRQ(ierr);
    ierr = DAGlobalToLocalEnd  (da,global,INSERTVALUES,local); CHKERRQ(ierr);

    /*Extract local array */ 
    VecGetArray (local,&localptr); 

    /* Update Locally - Make array of new values */
    /* Note: I don't do anything for the first and last entry */
    for (i=1; i< localsize-1; i++) {
      copyptr[i] = .5*(localptr[i+1]+localptr[i-1]) - 
                    (k / (2.0*a*h)) * (localptr[i+1] - localptr[i-1]);
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
 




