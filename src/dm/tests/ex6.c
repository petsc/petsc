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
  ierr = PetscViewerDrawOpen(PETSC_COMM_WORLD,0,"",300,0,400,300,&viewer);CHKERRQ(ierr);

  /* Read options */
  ierr = PetscOptionsGetInt(NULL,NULL,"-NX",&M,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-NY",&N,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-NZ",&P,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-p",&p,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-s",&s,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-w",&w,NULL);CHKERRQ(ierr);
  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-star",&flg,NULL);CHKERRQ(ierr);
  if (flg) stencil_type =  DMDA_STENCIL_STAR;
  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-box",&flg,NULL);CHKERRQ(ierr);
  if (flg) stencil_type =  DMDA_STENCIL_BOX;

  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-xperiodic",&flg,NULL);CHKERRQ(ierr);
  if (flg) bx = DM_BOUNDARY_PERIODIC;
  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-xghosted",&flg,NULL);CHKERRQ(ierr);
  if (flg) bx = DM_BOUNDARY_GHOSTED;
  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-xnonghosted",&flg,NULL);CHKERRQ(ierr);

  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-yperiodic",&flg,NULL);CHKERRQ(ierr);
  if (flg) by = DM_BOUNDARY_PERIODIC;
  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-yghosted",&flg,NULL);CHKERRQ(ierr);
  if (flg) by = DM_BOUNDARY_GHOSTED;
  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-ynonghosted",&flg,NULL);CHKERRQ(ierr);

  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-zperiodic",&flg,NULL);CHKERRQ(ierr);
  if (flg) bz = DM_BOUNDARY_PERIODIC;
  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-zghosted",&flg,NULL);CHKERRQ(ierr);
  if (flg) bz = DM_BOUNDARY_GHOSTED;
  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-znonghosted",&flg,NULL);CHKERRQ(ierr);

  ierr = PetscOptionsGetBool(NULL,NULL,"-testorder",&test_order,NULL);CHKERRQ(ierr);

  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-distribute",&flg,NULL);CHKERRQ(ierr);
  if (flg) {
    if (m == PETSC_DECIDE) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Must set -m option with -distribute option");
    ierr = PetscMalloc1(m,&lx);CHKERRQ(ierr);
    for (i=0; i<m-1; i++) lx[i] = 4;
    lx[m-1] = M - 4*(m-1);
    if (n == PETSC_DECIDE) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Must set -n option with -distribute option");
    ierr = PetscMalloc1(n,&ly);CHKERRQ(ierr);
    for (i=0; i<n-1; i++) ly[i] = 2;
    ly[n-1] = N - 2*(n-1);
    if (p == PETSC_DECIDE) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Must set -p option with -distribute option");
    ierr = PetscMalloc1(p,&lz);CHKERRQ(ierr);
    for (i=0; i<p-1; i++) lz[i] = 2;
    lz[p-1] = P - 2*(p-1);
  }

  /* Create distributed array and get vectors */
  ierr = DMDACreate3d(PETSC_COMM_WORLD,bx,by,bz,stencil_type,M,N,P,m,n,p,w,s,lx,ly,lz,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = PetscFree(lx);CHKERRQ(ierr);
  ierr = PetscFree(ly);CHKERRQ(ierr);
  ierr = PetscFree(lz);CHKERRQ(ierr);
  ierr = DMView(da,viewer);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&global);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(da,&local);CHKERRQ(ierr);

  /* Set global vector; send ghost points to local vectors */
  value = 1;
  ierr = VecSet(global,value);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,global,INSERT_VALUES,local);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,global,INSERT_VALUES,local);CHKERRQ(ierr);

  /* Scale local vectors according to processor rank; pass to global vector */
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  value = rank;
  ierr = VecScale(local,value);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(da,local,INSERT_VALUES,global);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(da,local,INSERT_VALUES,global);CHKERRQ(ierr);

  if (!test_order) { /* turn off printing when testing ordering mappings */
    if (M*N*P<40) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"\nGlobal Vector:\n");CHKERRQ(ierr);
      ierr = VecView(global,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRQ(ierr);
    }
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
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopSynchronized(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  /* Tests mappings betweeen application/PETSc orderings */
  if (test_order) {
    ISLocalToGlobalMapping ltogm;

    ierr = DMGetLocalToGlobalMapping(da,&ltogm);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetSize(ltogm,&nloc);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetIndices(ltogm,&ltog);CHKERRQ(ierr);

    ierr = DMDAGetGhostCorners(da,&Xs,&Ys,&Zs,&Xm,&Ym,&Zm);CHKERRQ(ierr);
    ierr = DMDAGetAO(da,&ao);CHKERRQ(ierr);
    /* ierr = AOView(ao,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */
    ierr = PetscMalloc1(nloc,&iglobal);CHKERRQ(ierr);

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
    ierr = AOPetscToApplication(ao,nloc,iglobal);CHKERRQ(ierr);

    /* Then map the application ordering back to the PETSc DMDA ordering */
    ierr = AOApplicationToPetsc(ao,nloc,iglobal);CHKERRQ(ierr);

    /* Verify the mappings */
    kk=0;
    for (k=Zs; k<Zs+Zm; k++) {
      for (j=Ys; j<Ys+Ym; j++) {
        for (i=Xs; i<Xs+Xm; i++) {
          iloc = w*((k-Zs)*Xm*Ym + (j-Ys)*Xm + i-Xs);
          for (l=0; l<w; l++) {
            if (iglobal[kk] != ltog[iloc+l]) {
              ierr = PetscPrintf(MPI_COMM_WORLD,"[%D] Problem with mapping: z=%D, j=%D, i=%D, l=%D, petsc1=%D, petsc2=%D\n",rank,k,j,i,l,ltog[iloc+l],iglobal[kk]);CHKERRQ(ierr);
            }
            kk++;
          }
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
      args:  -testorder -nox

 TEST*/



















