#ifndef lint
static char vcid[] = "$Id: ex3.c,v 1.26 1997/03/26 01:38:15 bsmith Exp bsmith $";
#endif

static char help[] = "Solves the 1-dimensional wave equation.\n\n";

#include "petsc.h"
#include "da.h"
#include "sys.h"
#include <math.h>

#define PETSC_PI 3.14159265

int main(int argc,char **argv)
{
  int       rank, size, M = 60, ierr,  time_steps = 100,flg;
  DA        da;
  Viewer    viewer,viewer_private;
  Draw      draw;
  Vec       local, global, copy;
  Scalar    *localptr, *copyptr;
  double    a, h, k;
  int       localsize, j, i, mybase, myend, width, xbase, *localnodes = PETSC_NULL;
 
  PetscInitialize(&argc,&argv,(char*)0,help);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size); 

  ierr = OptionsGetInt(PETSC_NULL,"-M",&M,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-time",&time_steps,&flg); CHKERRA(ierr);
  /*
      Test putting two nodes on each processor, exact last processor gets the rest
  */
  ierr = OptionsHasName(PETSC_NULL,"-distribute",&flg); CHKERRA(ierr);
  if (flg) {
    localnodes = (int *) PetscMalloc( size*sizeof(int) ); CHKPTRQ(localnodes);
    for ( i=0; i<size-1; i++ ) { localnodes[i] = 2;}
    localnodes[size-1] = M - 2*(size-1);
  }
    
  /* Set up the array */ 
  ierr = DACreate1d(MPI_COMM_WORLD,DA_XPERIODIC,M,1,1,localnodes,&da); CHKERRA(ierr);
  if (localnodes) PetscFree(localnodes);
  ierr = DAGetDistributedVector(da,&global); CHKERRA(ierr);
  ierr = DAGetLocalVector(da,&local); CHKERRA(ierr);

  /* Set up display to show combined wave graph */
  ierr = ViewerDrawOpenX(MPI_COMM_WORLD,0,"Entire Solution",20,480,800,200,
                         &viewer);CHKERRA(ierr);
  ierr = ViewerDrawGetDraw(viewer,&draw); CHKERRQ(ierr);
  ierr = DrawSetDoubleBuffer(draw); CHKERRA(ierr);

  /* determine starting point of each processor */
  ierr = VecGetOwnershipRange(global,&mybase,&myend); CHKERRA(ierr);

  /* set up display to show my portion of the wave */
  xbase = (int) ((mybase)*((800.0 - 4.0*size)/M) + 4.0*rank);
  width = (int) ((myend-mybase)*800./M);
  ierr = ViewerDrawOpenX(PETSC_COMM_SELF,0,"Local Portion of Solution",xbase,200,
                         width,200,&viewer_private);CHKERRA(ierr);
  ierr = ViewerDrawGetDraw(viewer_private,&draw); CHKERRQ(ierr);
  ierr = DrawSetDoubleBuffer(draw); CHKERRA(ierr);



  /* Initialize the array */
  ierr = VecGetLocalSize(local,&localsize); CHKERRA(ierr);
  ierr = VecGetArray(local,&localptr); CHKERRA(ierr);
  localptr[0] = 0.0;
  localptr[localsize-1] = 0.0;
  for (i=1; i<localsize-1; i++) {
    j=(i-1)+mybase; 
    localptr[i] = sin( (PETSC_PI*j*6)/((double)M) 
                        + 1.2 * sin( (PETSC_PI*j*2)/((double)M) ) ) * 2;
  }

  ierr = VecRestoreArray(local,&localptr); CHKERRA(ierr);
  ierr = DALocalToGlobal(da,local,INSERT_VALUES,global); CHKERRA(ierr);

  /* Make copy of local array for doing updates */
  ierr = VecDuplicate(local,&copy); CHKERRA(ierr);
  ierr = VecGetArray(copy,&copyptr); CHKERRA(ierr);

  /* Assign Parameters */
  a= 1.0;
  h= 1.0/M;
  k= h;

  for (j=0; j<time_steps; j++) {  

    /* Global to Local */
    ierr = DAGlobalToLocalBegin(da,global,INSERT_VALUES,local); CHKERRA(ierr);
    ierr = DAGlobalToLocalEnd(da,global,INSERT_VALUES,local); CHKERRA(ierr);

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
    ierr = DALocalToGlobal(da,copy,INSERT_VALUES,global); CHKERRA(ierr);
  
    /* View my part of Wave */ 
    ierr = VecView(copy,viewer_private); CHKERRA(ierr);

    /* View global Wave */ 
    ierr = VecView(global,viewer); CHKERRA(ierr);
  }

  ierr = DADestroy(da); CHKERRA(ierr);
  ierr = ViewerDestroy(viewer); CHKERRA(ierr);
  ierr = ViewerDestroy(viewer_private); CHKERRA(ierr);
  ierr = VecDestroy(copy); CHKERRA(ierr);
  ierr = VecDestroy(local); CHKERRA(ierr);
  ierr = VecDestroy(global); CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
 




