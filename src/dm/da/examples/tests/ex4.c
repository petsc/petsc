#ifndef lint
static char vcid[] = "$Id: ex4.c,v 1.24 1996/06/30 18:18:38 curfman Exp bsmith $";
#endif
  
static char help[] = "Tests various 2-dimensional DA routines.\n\n";

#include "petsc.h"
#include "da.h"
#include "sys.h"
#include "draw.h"
#include <math.h>

int main(int argc,char **argv)
{
  int            rank, M = 10, N = 8, m = PETSC_DECIDE, ierr, flg;
  int            s=2, w=2, n = PETSC_DECIDE, nloc, l, i, j, kk;
  int            Xs, Xm, Ys, Ym, iloc, *iglobal, *ltog, testorder = 0;
  DAPeriodicType wrap = DA_NONPERIODIC;
  DA             da;
  Viewer         viewer;
  Vec            local,global;
  Scalar         value;
  DAStencilType  st = DA_STENCIL_BOX;
  AO             ao;
 
  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = ViewerDrawOpenX(MPI_COMM_WORLD,0,"",300,0,400,400,&viewer);CHKERRA(ierr);
 
  /* Read options */
  ierr = OptionsGetInt(PETSC_NULL,"-M",&M,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-N",&N,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-m",&m,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-s",&s,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-w",&w,&flg); CHKERRA(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-xwrap",&flg); CHKERRA(ierr); if (flg)  wrap = DA_XPERIODIC;
  ierr = OptionsHasName(PETSC_NULL,"-ywrap",&flg); CHKERRA(ierr); if (flg)  wrap = DA_YPERIODIC;
  ierr = OptionsHasName(PETSC_NULL,"-xywrap",&flg); CHKERRA(ierr); if (flg) wrap = DA_XYPERIODIC;
  ierr = OptionsHasName(PETSC_NULL,"-star",&flg); CHKERRA(ierr); if (flg)   st = DA_STENCIL_STAR;
  ierr = OptionsHasName(PETSC_NULL,"-testorder",&testorder); CHKERRA(ierr);

  /* Create distributed array and get vectors */
  ierr = DACreate2d(MPI_COMM_WORLD,wrap,st,M,N,m,n,w,s,&da); CHKERRA(ierr);
  ierr = DAView(da,viewer); CHKERRA(ierr);
  ierr = DAGetDistributedVector(da,&global); CHKERRA(ierr);
  ierr = DAGetLocalVector(da,&local); CHKERRA(ierr);

  /* Set global vector; send ghost points to local vectors */
  value = 1;
  ierr = VecSet(&value,global); CHKERRA(ierr);
  ierr = DAGlobalToLocalBegin(da,global,INSERT_VALUES,local); CHKERRA(ierr);
  ierr = DAGlobalToLocalEnd(da,global,INSERT_VALUES,local); CHKERRA(ierr);

  /* Scale local vectors according to processor rank; pass to global vector */
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  value = rank;
  ierr = VecScale(&value,local); CHKERRA(ierr);
  ierr = DALocalToGlobal(da,local,INSERT_VALUES,global); CHKERRA(ierr);

  if (!testorder) { /* turn off printing when testing ordering mappings */
    PetscPrintf (MPI_COMM_WORLD,"\nGlobal Vectors:\n");
    ierr = VecView(global,VIEWER_STDOUT_WORLD); CHKERRA(ierr); 
    PetscPrintf (MPI_COMM_WORLD,"\n\n");
  }

  /* Send ghost points to local vectors */
  ierr = DAGlobalToLocalBegin(da,global,INSERT_VALUES,local); CHKERRA(ierr);
  ierr = DAGlobalToLocalEnd(da,global,INSERT_VALUES,local); CHKERRA(ierr);

  flg = 0;
  ierr = OptionsHasName(PETSC_NULL,"-local_print",&flg); CHKERRA(ierr);
  if (flg) {
    PetscSequentialPhaseBegin(MPI_COMM_WORLD,1);
    printf("\nLocal Vector: processor %d\n",rank);
    ierr = VecView(local,VIEWER_STDOUT_SELF); CHKERRA(ierr); 
    PetscSequentialPhaseEnd(MPI_COMM_WORLD,1);
  }

  /* Tests mappings betweeen application/PETSc orderings */
  if (testorder) {
    ierr = DAGetGhostCorners(da,&Xs,&Ys,PETSC_NULL,&Xm,&Ym,PETSC_NULL); CHKERRA(ierr);
    ierr = DAGetGlobalIndices(da,&nloc,&ltog); CHKERRQ(ierr);
    ierr = DAGetAO(da,&ao); CHKERRA(ierr);
    /* ierr = AOView(ao,VIEWER_STDOUT_WORLD); CHKERRA(ierr); */
    iglobal = (int *) PetscMalloc( nloc*sizeof(int) ); CHKPTRA(iglobal);

    /* Set iglobal to be global indices for each processor's local and ghost nodes,
       using the DA ordering of grid points */
    kk = 0;
    for (j=Ys; j<Ys+Ym; j++) {
      for (i=Xs; i<Xs+Xm; i++) {
        iloc = w*((j-Ys)*Xm + i-Xs); 
        for (l=0; l<w; l++) {
          iglobal[kk++] = ltog[iloc+l];
        }
      }
    } 

    /* Map this to the application ordering (which for DAs is just the natural ordering
       that would be used for 1 processor, numbering most rapidly by x, then y) */
    ierr = AOPetscToApplication(ao,nloc,iglobal); CHKERRA(ierr); 

    /* Then map the application ordering back to the PETSc DA ordering */
    ierr = AOApplicationToPetsc(ao,nloc,iglobal); CHKERRA(ierr); 

    /* Verify the mappings */
    kk=0;
    for (j=Ys; j<Ys+Ym; j++) {
      for (i=Xs; i<Xs+Xm; i++) {
        iloc = w*((j-Ys)*Xm + i-Xs); 
        for (l=0; l<w; l++) {
          if (iglobal[kk] != ltog[iloc+l]) {fprintf(stdout,
            "[%d] Problem with mapping: j=%d, i=%d, l=%d, petsc1=%d, petsc2=%d\n",
             rank,j,i,l,ltog[iloc+l],iglobal[kk]);}
          kk++;
        }
      }
    }
    PetscFree(iglobal);
  } 

  /* Free memory */
  ierr = ViewerDestroy(viewer); CHKERRA(ierr);
  ierr = VecDestroy(local); CHKERRA(ierr);
  ierr = VecDestroy(global); CHKERRA(ierr);
  ierr = DADestroy(da); CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
