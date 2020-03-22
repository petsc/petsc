
static char help[] = "Tests various 2-dimensional DMDA routines.\n\n";

#include <petscdm.h>
#include <petscdmda.h>

int main(int argc,char **argv)
{
  PetscMPIInt      rank;
  PetscErrorCode   ierr;
  PetscInt         M = 10,N = 8,m = PETSC_DECIDE;
  PetscInt         s =2,w=2,n = PETSC_DECIDE,nloc,l,i,j,kk;
  PetscInt         Xs,Xm,Ys,Ym,iloc,*iglobal;
  const PetscInt   *ltog;
  PetscInt         *lx       = NULL,*ly = NULL;
  PetscBool        testorder = PETSC_FALSE,flg;
  DMBoundaryType   bx        = DM_BOUNDARY_NONE,by= DM_BOUNDARY_NONE;
  DM               da;
  PetscViewer      viewer;
  Vec              local,global;
  PetscScalar      value;
  DMDAStencilType  st = DMDA_STENCIL_BOX;
  AO               ao;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscViewerDrawOpen(PETSC_COMM_WORLD,0,"",300,0,400,400,&viewer);CHKERRQ(ierr);

  /* Readoptions */
  ierr = PetscOptionsGetInt(NULL,NULL,"-NX",&M,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-NY",&N,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-s",&s,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-w",&w,NULL);CHKERRQ(ierr);

  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-xperiodic",&flg,NULL);CHKERRQ(ierr); if (flg) bx = DM_BOUNDARY_PERIODIC;
  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-yperiodic",&flg,NULL);CHKERRQ(ierr); if (flg) by = DM_BOUNDARY_PERIODIC;
  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-xghosted",&flg,NULL);CHKERRQ(ierr); if (flg) bx = DM_BOUNDARY_GHOSTED;
  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-yghosted",&flg,NULL);CHKERRQ(ierr); if (flg) by = DM_BOUNDARY_GHOSTED;
  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-star",&flg,NULL);CHKERRQ(ierr); if (flg) st = DMDA_STENCIL_STAR;
  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-box",&flg,NULL);CHKERRQ(ierr); if (flg) st = DMDA_STENCIL_BOX;
  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-testorder",&testorder,NULL);CHKERRQ(ierr);
  /*
      Test putting two nodes in x and y on each processor, exact last processor
      in x and y gets the rest.
  */
  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-distribute",&flg,NULL);CHKERRQ(ierr);
  if (flg) {
    if (m == PETSC_DECIDE) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Must set -m option with -distribute option");
    ierr = PetscMalloc1(m,&lx);CHKERRQ(ierr);
    for (i=0; i<m-1; i++) { lx[i] = 4;}
    lx[m-1] = M - 4*(m-1);
    if (n == PETSC_DECIDE) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Must set -n option with -distribute option");
    ierr = PetscMalloc1(n,&ly);CHKERRQ(ierr);
    for (i=0; i<n-1; i++) { ly[i] = 2;}
    ly[n-1] = N - 2*(n-1);
  }


  /* Create distributed array and get vectors */
  ierr = DMDACreate2d(PETSC_COMM_WORLD,bx,by,st,M,N,m,n,w,s,lx,ly,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = PetscFree(lx);CHKERRQ(ierr);
  ierr = PetscFree(ly);CHKERRQ(ierr);

  ierr = DMView(da,viewer);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&global);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(da,&local);CHKERRQ(ierr);

  /* Set global vector; send ghost points to local vectors */
  value = 1;
  ierr = VecSet(global,value);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,global,INSERT_VALUES,local);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,global,INSERT_VALUES,local);CHKERRQ(ierr);

  /* Scale local vectors according to processor rank; pass to global vector */
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  value = rank;
  ierr = VecScale(local,value);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(da,local,INSERT_VALUES,global);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(da,local,INSERT_VALUES,global);CHKERRQ(ierr);

  if (!testorder) { /* turn off printing when testing ordering mappings */
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\nGlobal Vectors:\n");CHKERRQ(ierr);
    ierr = VecView(global,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\n");CHKERRQ(ierr);
  }

  /* Send ghost points to local vectors */
  ierr = DMGlobalToLocalBegin(da,global,INSERT_VALUES,local);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,global,INSERT_VALUES,local);CHKERRQ(ierr);

  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-local_print",&flg,NULL);CHKERRQ(ierr);
  if (flg) {
    PetscViewer sviewer;

    ierr = PetscViewerASCIIPushSynchronized(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"\nLocal Vector: processor %d\n",rank);CHKERRQ(ierr);
    ierr = PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);
    ierr = VecView(local,sviewer);CHKERRQ(ierr);
    ierr = PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);
    ierr = PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopSynchronized(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  /* Tests mappings betweeen application/PETSc orderings */
  if (testorder) {
    ISLocalToGlobalMapping ltogm;

    ierr = DMGetLocalToGlobalMapping(da,&ltogm);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetSize(ltogm,&nloc);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetIndices(ltogm,&ltog);CHKERRQ(ierr);
    ierr = DMDAGetGhostCorners(da,&Xs,&Ys,NULL,&Xm,&Ym,NULL);CHKERRQ(ierr);
    ierr = DMDAGetAO(da,&ao);CHKERRQ(ierr);
    ierr = PetscMalloc1(nloc,&iglobal);CHKERRQ(ierr);

    /* Set iglobal to be global indices for each processor's local and ghost nodes,
       using the DMDA ordering of grid points */
    kk = 0;
    for (j=Ys; j<Ys+Ym; j++) {
      for (i=Xs; i<Xs+Xm; i++) {
        iloc = w*((j-Ys)*Xm + i-Xs);
        for (l=0; l<w; l++) {
          iglobal[kk++] = ltog[iloc+l];
        }
      }
    }

    /* Map this to the application ordering (which for DMDAs is just the natural ordering
       that would be used for 1 processor, numbering most rapidly by x, then y) */
    ierr = AOPetscToApplication(ao,nloc,iglobal);CHKERRQ(ierr);

    /* Then map the application ordering back to the PETSc DMDA ordering */
    ierr = AOApplicationToPetsc(ao,nloc,iglobal);CHKERRQ(ierr);

    /* Verify the mappings */
    kk=0;
    for (j=Ys; j<Ys+Ym; j++) {
      for (i=Xs; i<Xs+Xm; i++) {
        iloc = w*((j-Ys)*Xm + i-Xs);
        for (l=0; l<w; l++) {
          if (iglobal[kk] != ltog[iloc+l]) {
            ierr = PetscFPrintf(PETSC_COMM_SELF,stdout,"[%d] Problem with mapping: j=%D, i=%D, l=%D, petsc1=%D, petsc2=%D\n",rank,j,i,l,ltog[iloc+l],iglobal[kk]);CHKERRQ(ierr);
          }
          kk++;
        }
      }
    }
    ierr = PetscFree(iglobal);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingRestoreIndices(ltogm,&ltog);CHKERRQ(ierr);
  }

  /* Free memory */
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = VecDestroy(&local);CHKERRQ(ierr);
  ierr = VecDestroy(&global);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}


/*TEST

   test:
      nsize: 4
      args: -nox
      filter: grep -v -i Object
      requires: x

   test:
      suffix: 2
      args: -testorder -nox
      requires: x

TEST*/
