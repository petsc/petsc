#ifndef lint
static char vcid[] = "$Id: ex3.c,v 1.6 1995/08/22 02:45:51 curfman Exp curfman $";
#endif

/* This file created by Peter Mell   6/30/95 */ 

static char help[] = "This example creates a 1-dimensional wave equation.\n\n";

#include "petsc.h"
#include "da.h"
#include "sys.h"
#include "draw.h"
#include <math.h>
#include <sysio.h>

#define PI 3.14159265

int main(int argc,char **argv)
{
  int       mytid, numtid, M = 60, ierr,  time_steps = 100;
  DA        da;
  DrawCtx   win;
  Vec       local, global, copy;
  Scalar    *localptr, *copyptr;
  double    a, h, k;
  int       localsize, j, i, mybase, myend;
 
  PetscInitialize(&argc,&argv,(char*)0,(char*)0);
  if (OptionsHasName(0,"-help")) fprintf(stdout,"%s",help);

  OptionsGetInt(0,"-M",&M);
  OptionsGetInt(0,"-time",&time_steps);
    
  /* Set up the array */ 
  ierr = DACreate1d(MPI_COMM_WORLD,DA_XPERIODIC,M,1,1,&da); CHKERRA(ierr);
  ierr = DAGetDistributedVector(da,&global); CHKERRA(ierr);
  ierr = DAGetLocalVector(da,&local); CHKERRA(ierr);
  MPI_Comm_rank(MPI_COMM_WORLD,&mytid);
  MPI_Comm_size(MPI_COMM_WORLD,&numtid); 

  /* Set up display to show wave graph */
  ierr = DrawOpenX(MPI_COMM_WORLD,0,"",80,480,500,160,&win); CHKERRA(ierr);
  ierr = DrawSetDoubleBuffer(win); CHKERRA(ierr);

  /* determine starting point of each processor */
  ierr = VecGetOwnershipRange(global,&mybase,&myend); CHKERRA(ierr);

  /* Initialize the array */
  ierr = VecGetLocalSize(local,&localsize); CHKERRA(ierr);
  ierr = VecGetArray(local,&localptr); CHKERRA(ierr);
  localptr[0] = 0.0;
  localptr[localsize-1] = 0.0;
  for (i=1; i<localsize-1; i++) {
    j=(i-1)+mybase; 
    localptr[i] = sin( (PI*j*6)/((double)M) 
                        + 1.2 * sin( (PI*j*2)/((double)M) ) ) * 2;
  }

  ierr = VecRestoreArray(local,&localptr); CHKERRA(ierr);
  ierr = DALocalToGlobal(da,local,INSERTVALUES,global); CHKERRA(ierr);

  /* Make copy of local array for doing updates */
  ierr = VecDuplicate(local,&copy); CHKERRA(ierr);
  ierr = VecGetArray(copy,&copyptr); CHKERRA(ierr);

  /* Assign Parameters */
  a= 1.0;
  h= 1.0/M;
  k= h;

  for (j=0; j<time_steps; j++) {  

    /* Global to Local */
    ierr = DAGlobalToLocalBegin(da,global,INSERTVALUES,local); CHKERRA(ierr);
    ierr = DAGlobalToLocalEnd(da,global,INSERTVALUES,local); CHKERRA(ierr);

    /*Extract local array */ 
    ierr = VecGetArray(local,&localptr); CHKERRA(ierr);

    /* Update Locally - Make array of new values */
    /* Note: I don't do anything for the first and last entry */
    for (i=1; i< localsize-1; i++) {
      copyptr[i] = .5*(localptr[i+1]+localptr[i-1]) - 
                    (k / (2.0*a*h)) * (localptr[i+1] - localptr[i-1]);
    }
    ierr = VecRestoreArray(copy,&copyptr); CHKERRA(ierr);

    /* Local to Global */
    ierr = DALocalToGlobal(da,copy,INSERTVALUES,global); CHKERRA(ierr);
  
    /* View Wave */ 
    ierr = VecView(global,(Viewer) win); CHKERRA(ierr);
  }

  ierr = DADestroy(da); CHKERRA(ierr);
  ierr = ViewerDestroy((Viewer)win); CHKERRA(ierr);
  ierr = VecDestroy(copy); CHKERRA(ierr);
  ierr = VecDestroy(local); CHKERRA(ierr);
  ierr = VecDestroy(global); CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
 




