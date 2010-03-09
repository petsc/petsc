  
static char help[] = "Tests various 2-dimensional DA routines.\n\n";

#include "petscda.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscMPIInt    rank;
  PetscErrorCode ierr;
  PetscInt       M = 10,N = 8,m = PETSC_DECIDE;
  PetscInt       s=2,w=2,n = PETSC_DECIDE,nloc,l,i,j,kk;
  PetscInt       Xs,Xm,Ys,Ym,iloc,*iglobal,*ltog;
  PetscInt       *lx = PETSC_NULL,*ly = PETSC_NULL;
  PetscTruth     testorder = PETSC_FALSE,flg;
  DAPeriodicType wrap = DA_NONPERIODIC;
  DA             da;
  PetscViewer    viewer;
  Vec            local,global;
  PetscScalar    value;
  DAStencilType  st = DA_STENCIL_BOX;
  AO             ao;
 
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr); 
  ierr = PetscViewerDrawOpen(PETSC_COMM_WORLD,0,"",300,0,400,400,&viewer);CHKERRQ(ierr);
 
  /* Readoptions */
  ierr = PetscOptionsGetInt(PETSC_NULL,"-M",&M,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-N",&N,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-m",&m,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-s",&s,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-w",&w,PETSC_NULL);CHKERRQ(ierr);
  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-xwrap",&flg,PETSC_NULL);CHKERRQ(ierr); if (flg)  wrap = DA_XPERIODIC;
  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-ywrap",&flg,PETSC_NULL);CHKERRQ(ierr); if (flg)  wrap = DA_YPERIODIC;
  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-xywrap",&flg,PETSC_NULL);CHKERRQ(ierr); if (flg) wrap = DA_XYPERIODIC;
  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-star",&flg,PETSC_NULL);CHKERRQ(ierr); if (flg)   st = DA_STENCIL_STAR;
  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-testorder",&testorder,PETSC_NULL);CHKERRQ(ierr);
  /*
      Test putting two nodes in x and y on each processor, exact last processor 
      in x and y gets the rest.
  */
  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-distribute",&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg) {
    if (m == PETSC_DECIDE) SETERRQ(1,"Must set -m option with -distribute option");
    ierr = PetscMalloc(m*sizeof(PetscInt),&lx);CHKERRQ(ierr);
    for (i=0; i<m-1; i++) { lx[i] = 4;}
    lx[m-1] = M - 4*(m-1);
    if (n == PETSC_DECIDE) SETERRQ(1,"Must set -n option with -distribute option");
    ierr = PetscMalloc(n*sizeof(PetscInt),&ly);CHKERRQ(ierr);
    for (i=0; i<n-1; i++) { ly[i] = 2;}
    ly[n-1] = N - 2*(n-1);
  }


  /* Create distributed array and get vectors */
  ierr = DACreate2d(PETSC_COMM_WORLD,wrap,st,M,N,m,n,w,s,lx,ly,&da);CHKERRQ(ierr);
  if (lx) { 
    ierr = PetscFree(lx);CHKERRQ(ierr);
    ierr = PetscFree(ly);CHKERRQ(ierr);
  }

  ierr = DAView(da,viewer);CHKERRQ(ierr);
  ierr = DACreateGlobalVector(da,&global);CHKERRQ(ierr);
  ierr = DACreateLocalVector(da,&local);CHKERRQ(ierr);

  /* Set global vector; send ghost points to local vectors */
  value = 1;
  ierr = VecSet(global,value);CHKERRQ(ierr);
  ierr = DAGlobalToLocalBegin(da,global,INSERT_VALUES,local);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(da,global,INSERT_VALUES,local);CHKERRQ(ierr);

  /* Scale local vectors according to processor rank; pass to global vector */
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  value = rank;
  ierr = VecScale(local,value);CHKERRQ(ierr);
  ierr = DALocalToGlobal(da,local,INSERT_VALUES,global);CHKERRQ(ierr);

  if (!testorder) { /* turn off printing when testing ordering mappings */
    ierr = PetscPrintf (PETSC_COMM_WORLD,"\nGlobal Vectors:\n");CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_NATIVE);CHKERRQ(ierr);
    ierr = VecView(global,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); 
    ierr = PetscPrintf (PETSC_COMM_WORLD,"\n\n");CHKERRQ(ierr);
  }

  /* Send ghost points to local vectors */
  ierr = DAGlobalToLocalBegin(da,global,INSERT_VALUES,local);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(da,global,INSERT_VALUES,local);CHKERRQ(ierr);

  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-local_print",&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg) {
    PetscViewer sviewer;
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"\nLocal Vector: processor %d\n",rank);CHKERRQ(ierr);
    ierr = PetscViewerGetSingleton(PETSC_VIEWER_STDOUT_WORLD,&sviewer);CHKERRQ(ierr);
    ierr = VecView(local,sviewer);CHKERRQ(ierr); 
    ierr = PetscViewerRestoreSingleton(PETSC_VIEWER_STDOUT_WORLD,&sviewer);CHKERRQ(ierr);
  }

  /* Tests mappings betweeen application/PETSc orderings */
  if (testorder) {
    ierr = DAGetGhostCorners(da,&Xs,&Ys,PETSC_NULL,&Xm,&Ym,PETSC_NULL);CHKERRQ(ierr);
    ierr = DAGetGlobalIndices(da,&nloc,&ltog);CHKERRQ(ierr);
    ierr = DAGetAO(da,&ao);CHKERRQ(ierr);
    ierr = PetscMalloc(nloc*sizeof(PetscInt),&iglobal);CHKERRQ(ierr);

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
    ierr = AOPetscToApplication(ao,nloc,iglobal);CHKERRQ(ierr); 

    /* Then map the application ordering back to the PETSc DA ordering */
    ierr = AOApplicationToPetsc(ao,nloc,iglobal);CHKERRQ(ierr); 

    /* Verify the mappings */
    kk=0;
    for (j=Ys; j<Ys+Ym; j++) {
      for (i=Xs; i<Xs+Xm; i++) {
        iloc = w*((j-Ys)*Xm + i-Xs); 
        for (l=0; l<w; l++) {
          if (iglobal[kk] != ltog[iloc+l]) {
            ierr = PetscFPrintf(PETSC_COMM_SELF,stdout,"[%d] Problem with mapping: j=%D, i=%D, l=%D, petsc1=%D, petsc2=%D\n",
                                rank,j,i,l,ltog[iloc+l],iglobal[kk]);}
          kk++;
        }
      }
    }
    ierr = PetscFree(iglobal);CHKERRQ(ierr);
  } 

  /* Free memory */
  ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
  ierr = VecDestroy(local);CHKERRQ(ierr);
  ierr = VecDestroy(global);CHKERRQ(ierr);
  ierr = DADestroy(da);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
