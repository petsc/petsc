/*$Id: ex3.c,v 1.41 2000/05/05 22:19:31 balay Exp bsmith $*/

static char help[] = "Solves the 1-dimensional wave equation.\n\n";

#include "petscda.h"
#include "petscsys.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int        rank,size,M = 60,ierr, time_steps = 100;
  int        localsize,j,i,mybase,myend,width,xbase,*localnodes = PETSC_NULL;
  DA         da;
  PetscViewer     viewer,viewer_private;
  PetscDraw       draw;
  Vec        local,global,copy;
  Scalar     *localptr,*copyptr;
  double     a,h,k;
  PetscTruth flg;
 
  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRA(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRA(ierr);

  ierr = PetscOptionsGetInt(PETSC_NULL,"-M",&M,PETSC_NULL);CHKERRA(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-time",&time_steps,PETSC_NULL);CHKERRA(ierr);
  /*
      Test putting two nodes on each processor, exact last processor gets the rest
  */
  ierr = PetscOptionsHasName(PETSC_NULL,"-distribute",&flg);CHKERRA(ierr);
  if (flg) {
ierr = PetscMalloc(size*sizeof(int),&(    localnodes ));CHKERRQ(ierr);
    for (i=0; i<size-1; i++) { localnodes[i] = 2;}
    localnodes[size-1] = M - 2*(size-1);
  }
    
  /* Set up the array */ 
  ierr = DACreate1d(PETSC_COMM_WORLD,DA_XPERIODIC,M,1,1,localnodes,&da);CHKERRA(ierr);
  if (localnodes) {ierr = PetscFree(localnodes);CHKERRA(ierr);}
  ierr = DACreateGlobalVector(da,&global);CHKERRA(ierr);
  ierr = DACreateLocalVector(da,&local);CHKERRA(ierr);

  /* Set up display to show combined wave graph */
  ierr = PetscViewerDrawOpen(PETSC_COMM_WORLD,0,"Entire Solution",20,480,800,200,&viewer);CHKERRA(ierr);
  ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
  ierr = PetscDrawSetDoubleBuffer(draw);CHKERRA(ierr);

  /* determine starting point of each processor */
  ierr = VecGetOwnershipRange(global,&mybase,&myend);CHKERRA(ierr);

  /* set up display to show my portion of the wave */
  xbase = (int)((mybase)*((800.0 - 4.0*size)/M) + 4.0*rank);
  width = (int)((myend-mybase)*800./M);
  ierr = PetscViewerDrawOpen(PETSC_COMM_SELF,0,"Local Portion of Solution",xbase,200,
                         width,200,&viewer_private);CHKERRA(ierr);
  ierr = PetscViewerDrawGetDraw(viewer_private,0,&draw);CHKERRQ(ierr);
  ierr = PetscDrawSetDoubleBuffer(draw);CHKERRA(ierr);



  /* Initialize the array */
  ierr = VecGetLocalSize(local,&localsize);CHKERRA(ierr);
  ierr = VecGetArray(local,&localptr);CHKERRA(ierr);
  localptr[0] = 0.0;
  localptr[localsize-1] = 0.0;
  for (i=1; i<localsize-1; i++) {
    j=(i-1)+mybase; 
    localptr[i] = sin((PETSC_PI*j*6)/((double)M) 
                        + 1.2 * sin((PETSC_PI*j*2)/((double)M))) * 2;
  }

  ierr = VecRestoreArray(local,&localptr);CHKERRA(ierr);
  ierr = DALocalToGlobal(da,local,INSERT_VALUES,global);CHKERRA(ierr);

  /* Make copy of local array for doing updates */
  ierr = VecDuplicate(local,&copy);CHKERRA(ierr);

  /* Assign Parameters */
  a= 1.0;
  h= 1.0/M;
  k= h;

  for (j=0; j<time_steps; j++) {  

    /* Global to Local */
    ierr = DAGlobalToLocalBegin(da,global,INSERT_VALUES,local);CHKERRA(ierr);
    ierr = DAGlobalToLocalEnd(da,global,INSERT_VALUES,local);CHKERRA(ierr);

    /*Extract local array */ 
    ierr = VecGetArray(local,&localptr);CHKERRA(ierr);
    ierr = VecGetArray(copy,&copyptr);CHKERRA(ierr);

    /* Update Locally - Make array of new values */
    /* Note: I don't do anything for the first and last entry */
    for (i=1; i< localsize-1; i++) {
      copyptr[i] = .5*(localptr[i+1]+localptr[i-1]) - 
                    (k / (2.0*a*h)) * (localptr[i+1] - localptr[i-1]);
    }
    ierr = VecRestoreArray(copy,&copyptr);CHKERRA(ierr);
    ierr = VecRestoreArray(local,&localptr);CHKERRA(ierr);

    /* Local to Global */
    ierr = DALocalToGlobal(da,copy,INSERT_VALUES,global);CHKERRA(ierr);
  
    /* View my part of Wave */ 
    ierr = VecView(copy,viewer_private);CHKERRA(ierr);

    /* View global Wave */ 
    ierr = VecView(global,viewer);CHKERRA(ierr);
  }

  ierr = DADestroy(da);CHKERRA(ierr);
  ierr = PetscViewerDestroy(viewer);CHKERRA(ierr);
  ierr = PetscViewerDestroy(viewer_private);CHKERRA(ierr);
  ierr = VecDestroy(copy);CHKERRA(ierr);
  ierr = VecDestroy(local);CHKERRA(ierr);
  ierr = VecDestroy(global);CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
 




