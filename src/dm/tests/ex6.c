static char help[] = "Tests various 3-dimensional DMDA routines.\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscao.h>

int main(int argc,char **argv)
{
  PetscMPIInt      rank;
  PetscInt         M = 3,N = 5,P=3,s=1,w=2,nloc,l,i,j,k,kk,m = PETSC_DECIDE,n = PETSC_DECIDE,p = PETSC_DECIDE;
  PetscErrorCode   ierr;
  PetscInt         Xs,Xm,Ys,Ym,Zs,Zm,iloc,*iglobal;
  const PetscInt   *ltog;
  PetscInt         *lx        = NULL,*ly = NULL,*lz = NULL;
  PetscBool        test_order = PETSC_FALSE;
  DM               da;
  PetscViewer      viewer;
  Vec              local,global;
  PetscScalar      value;
  DMBoundaryType   bx           = DM_BOUNDARY_NONE,by = DM_BOUNDARY_NONE,bz = DM_BOUNDARY_NONE;
  DMDAStencilType  stencil_type = DMDA_STENCIL_BOX;
  AO               ao;
  PetscBool        flg = PETSC_FALSE;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscViewerDrawOpen(PETSC_COMM_WORLD,0,"",300,0,400,300,&viewer));

  /* Read options */
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-NX",&M,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-NY",&N,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-NZ",&P,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-p",&p,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-s",&s,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-w",&w,NULL));
  flg  = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-star",&flg,NULL));
  if (flg) stencil_type =  DMDA_STENCIL_STAR;
  flg  = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-box",&flg,NULL));
  if (flg) stencil_type =  DMDA_STENCIL_BOX;

  flg  = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-xperiodic",&flg,NULL));
  if (flg) bx = DM_BOUNDARY_PERIODIC;
  flg  = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-xghosted",&flg,NULL));
  if (flg) bx = DM_BOUNDARY_GHOSTED;
  flg  = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-xnonghosted",&flg,NULL));

  flg  = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-yperiodic",&flg,NULL));
  if (flg) by = DM_BOUNDARY_PERIODIC;
  flg  = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-yghosted",&flg,NULL));
  if (flg) by = DM_BOUNDARY_GHOSTED;
  flg  = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-ynonghosted",&flg,NULL));

  flg  = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-zperiodic",&flg,NULL));
  if (flg) bz = DM_BOUNDARY_PERIODIC;
  flg  = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-zghosted",&flg,NULL));
  if (flg) bz = DM_BOUNDARY_GHOSTED;
  flg  = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-znonghosted",&flg,NULL));

  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-testorder",&test_order,NULL));

  flg  = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-distribute",&flg,NULL));
  if (flg) {
    PetscCheckFalse(m == PETSC_DECIDE,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Must set -m option with -distribute option");
    CHKERRQ(PetscMalloc1(m,&lx));
    for (i=0; i<m-1; i++) lx[i] = 4;
    lx[m-1] = M - 4*(m-1);
    PetscCheckFalse(n == PETSC_DECIDE,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Must set -n option with -distribute option");
    CHKERRQ(PetscMalloc1(n,&ly));
    for (i=0; i<n-1; i++) ly[i] = 2;
    ly[n-1] = N - 2*(n-1);
    PetscCheckFalse(p == PETSC_DECIDE,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Must set -p option with -distribute option");
    CHKERRQ(PetscMalloc1(p,&lz));
    for (i=0; i<p-1; i++) lz[i] = 2;
    lz[p-1] = P - 2*(p-1);
  }

  /* Create distributed array and get vectors */
  CHKERRQ(DMDACreate3d(PETSC_COMM_WORLD,bx,by,bz,stencil_type,M,N,P,m,n,p,w,s,lx,ly,lz,&da));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));
  CHKERRQ(PetscFree(lx));
  CHKERRQ(PetscFree(ly));
  CHKERRQ(PetscFree(lz));
  CHKERRQ(DMView(da,viewer));
  CHKERRQ(DMCreateGlobalVector(da,&global));
  CHKERRQ(DMCreateLocalVector(da,&local));

  /* Set global vector; send ghost points to local vectors */
  value = 1;
  CHKERRQ(VecSet(global,value));
  CHKERRQ(DMGlobalToLocalBegin(da,global,INSERT_VALUES,local));
  CHKERRQ(DMGlobalToLocalEnd(da,global,INSERT_VALUES,local));

  /* Scale local vectors according to processor rank; pass to global vector */
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  value = rank;
  CHKERRQ(VecScale(local,value));
  CHKERRQ(DMLocalToGlobalBegin(da,local,INSERT_VALUES,global));
  CHKERRQ(DMLocalToGlobalEnd(da,local,INSERT_VALUES,global));

  if (!test_order) { /* turn off printing when testing ordering mappings */
    if (M*N*P<40) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nGlobal Vector:\n"));
      CHKERRQ(VecView(global,PETSC_VIEWER_STDOUT_WORLD));
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n"));
    }
  }

  /* Send ghost points to local vectors */
  CHKERRQ(DMGlobalToLocalBegin(da,global,INSERT_VALUES,local));
  CHKERRQ(DMGlobalToLocalEnd(da,global,INSERT_VALUES,local));

  flg  = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-local_print",&flg,NULL));
  if (flg) {
    PetscViewer sviewer;
    CHKERRQ(PetscViewerASCIIPushSynchronized(PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"\nLocal Vector: processor %d\n",rank));
    CHKERRQ(PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer));
    CHKERRQ(VecView(local,sviewer));
    CHKERRQ(PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer));
    CHKERRQ(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));
    CHKERRQ(PetscViewerASCIIPopSynchronized(PETSC_VIEWER_STDOUT_WORLD));
  }

  /* Tests mappings between application/PETSc orderings */
  if (test_order) {
    ISLocalToGlobalMapping ltogm;

    CHKERRQ(DMGetLocalToGlobalMapping(da,&ltogm));
    CHKERRQ(ISLocalToGlobalMappingGetSize(ltogm,&nloc));
    CHKERRQ(ISLocalToGlobalMappingGetIndices(ltogm,&ltog));

    CHKERRQ(DMDAGetGhostCorners(da,&Xs,&Ys,&Zs,&Xm,&Ym,&Zm));
    CHKERRQ(DMDAGetAO(da,&ao));
    /* CHKERRQ(AOView(ao,PETSC_VIEWER_STDOUT_WORLD)); */
    CHKERRQ(PetscMalloc1(nloc,&iglobal));

    /* Set iglobal to be global indices for each processor's local and ghost nodes,
       using the DMDA ordering of grid points */
    kk = 0;
    for (k=Zs; k<Zs+Zm; k++) {
      for (j=Ys; j<Ys+Ym; j++) {
        for (i=Xs; i<Xs+Xm; i++) {
          iloc = w*((k-Zs)*Xm*Ym + (j-Ys)*Xm + i-Xs);
          for (l=0; l<w; l++) {
            iglobal[kk++] = ltog[iloc+l];
          }
        }
      }
    }

    /* Map this to the application ordering (which for DMDAs is just the natural ordering
       that would be used for 1 processor, numbering most rapidly by x, then y, then z) */
    CHKERRQ(AOPetscToApplication(ao,nloc,iglobal));

    /* Then map the application ordering back to the PETSc DMDA ordering */
    CHKERRQ(AOApplicationToPetsc(ao,nloc,iglobal));

    /* Verify the mappings */
    kk=0;
    for (k=Zs; k<Zs+Zm; k++) {
      for (j=Ys; j<Ys+Ym; j++) {
        for (i=Xs; i<Xs+Xm; i++) {
          iloc = w*((k-Zs)*Xm*Ym + (j-Ys)*Xm + i-Xs);
          for (l=0; l<w; l++) {
            if (iglobal[kk] != ltog[iloc+l]) {
              CHKERRQ(PetscPrintf(MPI_COMM_WORLD,"[%D] Problem with mapping: z=%D, j=%D, i=%D, l=%D, petsc1=%D, petsc2=%D\n",rank,k,j,i,l,ltog[iloc+l],iglobal[kk]));
            }
            kk++;
          }
        }
      }
    }
    CHKERRQ(PetscFree(iglobal));
    CHKERRQ(ISLocalToGlobalMappingRestoreIndices(ltogm,&ltog));
  }

  /* Free memory */
  CHKERRQ(PetscViewerDestroy(&viewer));
  CHKERRQ(VecDestroy(&local));
  CHKERRQ(VecDestroy(&global));
  CHKERRQ(DMDestroy(&da));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

    test:
      args:  -testorder -nox

 TEST*/
