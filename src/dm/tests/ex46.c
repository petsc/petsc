static char help[] = "Tests DMGetCompatibility() with a 3D DMDA.\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmstag.h>

int main(int argc,char **argv) {
  PetscInt         M = 3,N = 5,P=3,s=1,w=2,i,m = PETSC_DECIDE,n = PETSC_DECIDE,p = PETSC_DECIDE;
  PetscInt         *lx        = NULL,*ly = NULL,*lz = NULL;
  PetscBool        test_order = PETSC_FALSE;
  DM               da;
  DMBoundaryType   bx           = DM_BOUNDARY_NONE,by = DM_BOUNDARY_NONE,bz = DM_BOUNDARY_NONE;
  DMDAStencilType  stencil_type = DMDA_STENCIL_BOX;
  PetscBool        flg = PETSC_FALSE,distribute = PETSC_FALSE;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));

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
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-distribute",&distribute,NULL));
  if (distribute) {
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

  CHKERRQ(DMDACreate3d(PETSC_COMM_WORLD,bx,by,bz,stencil_type,M,N,P,m,n,p,w,s,lx,ly,lz,&da));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));

  /* Check self-compatibility */
  {
    PetscBool compatible,set;
    CHKERRQ(DMGetCompatibility(da,da,&compatible,&set));
    if (!set || !compatible) {
      CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject)da),"Error: DM not compatible with itself\n"));
    }
  }

  /* Check compatibility with the same DM on a dup'd communicator */
  {
    DM        da2;
    PetscBool compatible,set;
    MPI_Comm  comm2;
    CHKERRMPI(MPI_Comm_dup(PETSC_COMM_WORLD,&comm2));
    CHKERRQ(DMDACreate3d(comm2,bx,by,bz,stencil_type,M,N,P,m,n,p,w,s,lx,ly,lz,&da2));
    CHKERRQ(DMSetFromOptions(da2));
    CHKERRQ(DMSetUp(da2));
    CHKERRQ(DMGetCompatibility(da,da2,&compatible,&set));
    if (!set || !compatible) {
      CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject)da),"Error: DM not compatible with DMDA on dup'd comm\n"));
    }
    CHKERRQ(DMDestroy(&da2));
    CHKERRMPI(MPI_Comm_free(&comm2));
  }

  /* Check compatibility with a derived DMDA */
  {
    DM        da2;
    PetscBool compatible,set;
    CHKERRQ(DMDACreateCompatibleDMDA(da,w*2,&da2));
    CHKERRQ(DMGetCompatibility(da,da2,&compatible,&set));
    if (!set || !compatible) {
      CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject)da),"Error: DM not compatible with DMDA created with DMDACreateCompatibleDMDA()\n"));
    }
    CHKERRQ(DMDestroy(&da2));
  }

  /* Confirm incompatibility with different stencil width */
  {
    DM        da2;
    PetscBool compatible,set;
    CHKERRQ(DMDACreate3d(PETSC_COMM_WORLD,bx,by,bz,stencil_type,M,N,P,m,n,p,w,0,lx,ly,lz,&da2));
    CHKERRQ(DMSetUp(da2));
    CHKERRQ(DMGetCompatibility(da,da2,&compatible,&set));
    if (!set || compatible) {
      CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject)da),"Error: DM not determined incompatible with known-incompatible DMDA (different stencil width)\n"));
    }
    CHKERRQ(DMDestroy(&da2));
  }

  /* Confirm incompatibility with different boundary types */
  {
    DM        da2;
    PetscBool compatible,set;
    DMBoundaryType bz2;
    bz2 = bz == DM_BOUNDARY_NONE ? DM_BOUNDARY_GHOSTED : DM_BOUNDARY_NONE;
    CHKERRQ(DMDACreate3d(PETSC_COMM_WORLD,bx,by,bz2,stencil_type,M,N,P,m,n,p,w,s,lx,ly,lz,&da2));
    CHKERRQ(DMSetUp(da2));
    CHKERRQ(DMGetCompatibility(da,da2,&compatible,&set));
    if (!set || compatible) {
      CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject)da),"Error: DM not determined incompatible with known-incompatible DMDA (different boundary type)\n"));
    }
    CHKERRQ(DMDestroy(&da2));
  }

  if (!distribute) {
    /* Confirm incompatibility with different global sizes */
    {
      DM        da2;
      PetscBool compatible,set;
      CHKERRQ(DMDACreate3d(PETSC_COMM_WORLD,bx,by,bz,stencil_type,M,N,P*2,m,n,p,w,s,lx,ly,lz,&da2));
      CHKERRQ(DMSetUp(da2));
      CHKERRQ(DMGetCompatibility(da,da2,&compatible,&set));
      if (!set || compatible) {
        CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject)da),"Error: DM not determined incompatible with known-incompatible DMDA (different global sizes)\n"));
      }
      CHKERRQ(DMDestroy(&da2));
    }
  }

  if (distribute && p > 1) {
    /* Confirm incompatibility with different local size */
    {
      DM          da2;
      PetscBool   compatible,set;
      PetscMPIInt rank;
      PetscInt    *lz2;
      CHKERRQ(PetscMalloc1(p,&lz2));
      for (i=0; i<p-1; i++) lz2[i] = 1; /* One point per rank instead of 2 */
      lz2[p-1] = P - (p-1);
      CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
      CHKERRQ(DMDACreate3d(PETSC_COMM_WORLD,bx,by,bz,stencil_type,M,N,P,m,n,p,w,s,lx,ly,lz2,&da2));
      CHKERRQ(DMSetUp(da2));
      CHKERRQ(DMGetCompatibility(da,da2,&compatible,&set));
      if (!set || compatible) {
        CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject)da),"Error: DM not determined incompatible with known-incompatible DMDA (different local sizes) \n"));
      }
      CHKERRQ(DMDestroy(&da2));
      CHKERRQ(PetscFree(lz2));
    }
  }

  /* Check compatibility with a DM of different type (DMStag) */
  {
    DM        dm2;
    PetscBool compatible,set;
    CHKERRQ(DMStagCreate3d(PETSC_COMM_WORLD,bx,by,bz,M,N,P,m,n,p,1,1,1,1,DMSTAG_STENCIL_STAR,w,lx,ly,lz,&dm2));
    CHKERRQ(DMSetUp(dm2));
    CHKERRQ(DMGetCompatibility(da,dm2,&compatible,&set));
    /* Don't interpret the result, but note that one can run with -info */
    CHKERRQ(DMDestroy(&dm2));
  }

  /* Free memory */
  CHKERRQ(PetscFree(lx));
  CHKERRQ(PetscFree(ly));
  CHKERRQ(PetscFree(lz));
  CHKERRQ(DMDestroy(&da));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1

   test:
      suffix: 2
      nsize: 3
      args: distribute -m 1 -n 1 -p 3 -NZ 20

 TEST*/
