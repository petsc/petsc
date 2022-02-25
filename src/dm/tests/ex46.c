static char help[] = "Tests DMGetCompatibility() with a 3D DMDA.\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmstag.h>

int main(int argc,char **argv) {
  PetscInt         M = 3,N = 5,P=3,s=1,w=2,i,m = PETSC_DECIDE,n = PETSC_DECIDE,p = PETSC_DECIDE;
  PetscErrorCode   ierr;
  PetscInt         *lx        = NULL,*ly = NULL,*lz = NULL;
  PetscBool        test_order = PETSC_FALSE;
  DM               da;
  DMBoundaryType   bx           = DM_BOUNDARY_NONE,by = DM_BOUNDARY_NONE,bz = DM_BOUNDARY_NONE;
  DMDAStencilType  stencil_type = DMDA_STENCIL_BOX;
  PetscBool        flg = PETSC_FALSE,distribute = PETSC_FALSE;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

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
  ierr = PetscOptionsGetBool(NULL,NULL,"-distribute",&distribute,NULL);CHKERRQ(ierr);
  if (distribute) {
    PetscCheckFalse(m == PETSC_DECIDE,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Must set -m option with -distribute option");
    ierr = PetscMalloc1(m,&lx);CHKERRQ(ierr);
    for (i=0; i<m-1; i++) lx[i] = 4;
    lx[m-1] = M - 4*(m-1);
    PetscCheckFalse(n == PETSC_DECIDE,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Must set -n option with -distribute option");
    ierr = PetscMalloc1(n,&ly);CHKERRQ(ierr);
    for (i=0; i<n-1; i++) ly[i] = 2;
    ly[n-1] = N - 2*(n-1);
    PetscCheckFalse(p == PETSC_DECIDE,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Must set -p option with -distribute option");
    ierr = PetscMalloc1(p,&lz);CHKERRQ(ierr);
    for (i=0; i<p-1; i++) lz[i] = 2;
    lz[p-1] = P - 2*(p-1);
  }

  ierr = DMDACreate3d(PETSC_COMM_WORLD,bx,by,bz,stencil_type,M,N,P,m,n,p,w,s,lx,ly,lz,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);

  /* Check self-compatibility */
  {
    PetscBool compatible,set;
    ierr = DMGetCompatibility(da,da,&compatible,&set);CHKERRQ(ierr);
    if (!set || !compatible) {
      ierr = PetscPrintf(PetscObjectComm((PetscObject)da),"Error: DM not compatible with itself\n");CHKERRQ(ierr);
    }
  }

  /* Check compatibility with the same DM on a dup'd communicator */
  {
    DM        da2;
    PetscBool compatible,set;
    MPI_Comm  comm2;
    ierr = MPI_Comm_dup(PETSC_COMM_WORLD,&comm2);CHKERRMPI(ierr);
    ierr = DMDACreate3d(comm2,bx,by,bz,stencil_type,M,N,P,m,n,p,w,s,lx,ly,lz,&da2);CHKERRQ(ierr);
    ierr = DMSetFromOptions(da2);CHKERRQ(ierr);
    ierr = DMSetUp(da2);CHKERRQ(ierr);
    ierr = DMGetCompatibility(da,da2,&compatible,&set);CHKERRQ(ierr);
    if (!set || !compatible) {
      ierr = PetscPrintf(PetscObjectComm((PetscObject)da),"Error: DM not compatible with DMDA on dup'd comm\n");CHKERRQ(ierr);
    }
    ierr = DMDestroy(&da2);CHKERRQ(ierr);
    ierr = MPI_Comm_free(&comm2);CHKERRMPI(ierr);
  }

  /* Check compatibility with a derived DMDA */
  {
    DM        da2;
    PetscBool compatible,set;
    ierr = DMDACreateCompatibleDMDA(da,w*2,&da2);CHKERRQ(ierr);
    ierr = DMGetCompatibility(da,da2,&compatible,&set);CHKERRQ(ierr);
    if (!set || !compatible) {
      ierr = PetscPrintf(PetscObjectComm((PetscObject)da),"Error: DM not compatible with DMDA created with DMDACreateCompatibleDMDA()\n");CHKERRQ(ierr);
    }
    ierr = DMDestroy(&da2);CHKERRQ(ierr);
  }

  /* Confirm incompatibility with different stencil width */
  {
    DM        da2;
    PetscBool compatible,set;
    ierr = DMDACreate3d(PETSC_COMM_WORLD,bx,by,bz,stencil_type,M,N,P,m,n,p,w,0,lx,ly,lz,&da2);CHKERRQ(ierr);
    ierr = DMSetUp(da2);CHKERRQ(ierr);
    ierr = DMGetCompatibility(da,da2,&compatible,&set);CHKERRQ(ierr);
    if (!set || compatible) {
      ierr = PetscPrintf(PetscObjectComm((PetscObject)da),"Error: DM not determined incompatible with known-incompatible DMDA (different stencil width)\n");CHKERRQ(ierr);
    }
    ierr = DMDestroy(&da2);CHKERRQ(ierr);
  }

  /* Confirm incompatibility with different boundary types */
  {
    DM        da2;
    PetscBool compatible,set;
    DMBoundaryType bz2;
    bz2 = bz == DM_BOUNDARY_NONE ? DM_BOUNDARY_GHOSTED : DM_BOUNDARY_NONE;
    ierr = DMDACreate3d(PETSC_COMM_WORLD,bx,by,bz2,stencil_type,M,N,P,m,n,p,w,s,lx,ly,lz,&da2);CHKERRQ(ierr);
    ierr = DMSetUp(da2);CHKERRQ(ierr);
    ierr = DMGetCompatibility(da,da2,&compatible,&set);CHKERRQ(ierr);
    if (!set || compatible) {
      ierr = PetscPrintf(PetscObjectComm((PetscObject)da),"Error: DM not determined incompatible with known-incompatible DMDA (different boundary type)\n");CHKERRQ(ierr);
    }
    ierr = DMDestroy(&da2);CHKERRQ(ierr);
  }

  if (!distribute) {
    /* Confirm incompatibility with different global sizes */
    {
      DM        da2;
      PetscBool compatible,set;
      ierr = DMDACreate3d(PETSC_COMM_WORLD,bx,by,bz,stencil_type,M,N,P*2,m,n,p,w,s,lx,ly,lz,&da2);CHKERRQ(ierr);
      ierr = DMSetUp(da2);CHKERRQ(ierr);
      ierr = DMGetCompatibility(da,da2,&compatible,&set);CHKERRQ(ierr);
      if (!set || compatible) {
        ierr = PetscPrintf(PetscObjectComm((PetscObject)da),"Error: DM not determined incompatible with known-incompatible DMDA (different global sizes)\n");CHKERRQ(ierr);
      }
      ierr = DMDestroy(&da2);CHKERRQ(ierr);
    }
  }

  if (distribute && p > 1) {
    /* Confirm incompatibility with different local size */
    {
      DM          da2;
      PetscBool   compatible,set;
      PetscMPIInt rank;
      PetscInt    *lz2;
      ierr = PetscMalloc1(p,&lz2);CHKERRQ(ierr);
      for (i=0; i<p-1; i++) lz2[i] = 1; /* One point per rank instead of 2 */
      lz2[p-1] = P - (p-1);
      ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
      ierr = DMDACreate3d(PETSC_COMM_WORLD,bx,by,bz,stencil_type,M,N,P,m,n,p,w,s,lx,ly,lz2,&da2);CHKERRQ(ierr);
      ierr = DMSetUp(da2);CHKERRQ(ierr);
      ierr = DMGetCompatibility(da,da2,&compatible,&set);CHKERRQ(ierr);
      if (!set || compatible) {
        ierr = PetscPrintf(PetscObjectComm((PetscObject)da),"Error: DM not determined incompatible with known-incompatible DMDA (different local sizes) \n");CHKERRQ(ierr);
      }
      ierr = DMDestroy(&da2);CHKERRQ(ierr);
      ierr = PetscFree(lz2);CHKERRQ(ierr);
    }
  }

  /* Check compatibility with a DM of different type (DMStag) */
  {
    DM        dm2;
    PetscBool compatible,set;
    ierr = DMStagCreate3d(PETSC_COMM_WORLD,bx,by,bz,M,N,P,m,n,p,1,1,1,1,DMSTAG_STENCIL_STAR,w,lx,ly,lz,&dm2);CHKERRQ(ierr);
    ierr = DMSetUp(dm2);CHKERRQ(ierr);
    ierr = DMGetCompatibility(da,dm2,&compatible,&set);CHKERRQ(ierr);
    /* Don't interpret the result, but note that one can run with -info */
    ierr = DMDestroy(&dm2);CHKERRQ(ierr);
  }

  /* Free memory */
  ierr = PetscFree(lx);CHKERRQ(ierr);
  ierr = PetscFree(ly);CHKERRQ(ierr);
  ierr = PetscFree(lz);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: 1

   test:
      suffix: 2
      nsize: 3
      args: distribute -m 1 -n 1 -p 3 -NZ 20

 TEST*/
