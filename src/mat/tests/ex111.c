
static char help[] ="Tests sequential and parallel MatMatMatMult() and MatPtAP(). Modified from ex96.c \n\
  -Mx <xg>, where <xg> = number of coarse grid points in the x-direction\n\
  -My <yg>, where <yg> = number of coarse grid points in the y-direction\n\
  -Mz <zg>, where <zg> = number of coarse grid points in the z-direction\n\
  -Npx <npx>, where <npx> = number of processors in the x-direction\n\
  -Npy <npy>, where <npy> = number of processors in the y-direction\n\
  -Npz <npz>, where <npz> = number of processors in the z-direction\n\n";

/*
    Example of usage: mpiexec -n 3 ./ex41 -Mx 10 -My 10 -Mz 10
*/

#include <petscdm.h>
#include <petscdmda.h>

/* User-defined application contexts */
typedef struct {
  PetscInt mx,my,mz;            /* number grid points in x, y and z direction */
  Vec      localX,localF;       /* local vectors with ghost region */
  DM       da;
  Vec      x,b,r;               /* global vectors */
  Mat      J;                   /* Jacobian on grid */
} GridCtx;
typedef struct {
  GridCtx  fine;
  GridCtx  coarse;
  PetscInt ratio;
  Mat      Ii;                  /* interpolation from coarse to fine */
} AppCtx;

#define COARSE_LEVEL 0
#define FINE_LEVEL   1

/*
      Mm_ratio - ration of grid lines between fine and coarse grids.
*/
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  AppCtx         user;
  PetscMPIInt    size,rank;
  PetscInt       m,n,M,N,i,nrows;
  PetscScalar    one = 1.0;
  PetscReal      fill=2.0;
  Mat            A,P,R,C,PtAP,D;
  PetscScalar    *array;
  PetscRandom    rdm;
  PetscBool      Test_3D=PETSC_FALSE,flg;
  const PetscInt *ia,*ja;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  /* Get size of fine grids and coarse grids */
  user.ratio     = 2;
  user.coarse.mx = 4; user.coarse.my = 4; user.coarse.mz = 4;

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-Mx",&user.coarse.mx,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-My",&user.coarse.my,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-Mz",&user.coarse.mz,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-ratio",&user.ratio,NULL));
  if (user.coarse.mz) Test_3D = PETSC_TRUE;

  user.fine.mx = user.ratio*(user.coarse.mx-1)+1;
  user.fine.my = user.ratio*(user.coarse.my-1)+1;
  user.fine.mz = user.ratio*(user.coarse.mz-1)+1;

  if (rank == 0) {
    if (!Test_3D) {
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"coarse grids: %" PetscInt_FMT " %" PetscInt_FMT "; fine grids: %" PetscInt_FMT " %" PetscInt_FMT "\n",user.coarse.mx,user.coarse.my,user.fine.mx,user.fine.my));
    } else {
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"coarse grids: %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT "; fine grids: %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT "\n",user.coarse.mx,user.coarse.my,user.coarse.mz,user.fine.mx,user.fine.my,user.fine.mz));
    }
  }

  /* Set up distributed array for fine grid */
  if (!Test_3D) {
    CHKERRQ(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,user.fine.mx,user.fine.my,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&user.fine.da));
  } else {
    ierr = DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,user.fine.mx,user.fine.my,user.fine.mz,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,
                        1,1,NULL,NULL,NULL,&user.fine.da);CHKERRQ(ierr);
  }
  CHKERRQ(DMSetFromOptions(user.fine.da));
  CHKERRQ(DMSetUp(user.fine.da));

  /* Create and set A at fine grids */
  CHKERRQ(DMSetMatType(user.fine.da,MATAIJ));
  CHKERRQ(DMCreateMatrix(user.fine.da,&A));
  CHKERRQ(MatGetLocalSize(A,&m,&n));
  CHKERRQ(MatGetSize(A,&M,&N));

  /* set val=one to A (replace with random values!) */
  CHKERRQ(PetscRandomCreate(PETSC_COMM_WORLD,&rdm));
  CHKERRQ(PetscRandomSetFromOptions(rdm));
  if (size == 1) {
    CHKERRQ(MatGetRowIJ(A,0,PETSC_FALSE,PETSC_FALSE,&nrows,&ia,&ja,&flg));
    if (flg) {
      CHKERRQ(MatSeqAIJGetArray(A,&array));
      for (i=0; i<ia[nrows]; i++) array[i] = one;
      CHKERRQ(MatSeqAIJRestoreArray(A,&array));
    }
    CHKERRQ(MatRestoreRowIJ(A,0,PETSC_FALSE,PETSC_FALSE,&nrows,&ia,&ja,&flg));
  } else {
    Mat AA,AB;
    CHKERRQ(MatMPIAIJGetSeqAIJ(A,&AA,&AB,NULL));
    CHKERRQ(MatGetRowIJ(AA,0,PETSC_FALSE,PETSC_FALSE,&nrows,&ia,&ja,&flg));
    if (flg) {
      CHKERRQ(MatSeqAIJGetArray(AA,&array));
      for (i=0; i<ia[nrows]; i++) array[i] = one;
      CHKERRQ(MatSeqAIJRestoreArray(AA,&array));
    }
    CHKERRQ(MatRestoreRowIJ(AA,0,PETSC_FALSE,PETSC_FALSE,&nrows,&ia,&ja,&flg));
    CHKERRQ(MatGetRowIJ(AB,0,PETSC_FALSE,PETSC_FALSE,&nrows,&ia,&ja,&flg));
    if (flg) {
      CHKERRQ(MatSeqAIJGetArray(AB,&array));
      for (i=0; i<ia[nrows]; i++) array[i] = one;
      CHKERRQ(MatSeqAIJRestoreArray(AB,&array));
    }
    CHKERRQ(MatRestoreRowIJ(AB,0,PETSC_FALSE,PETSC_FALSE,&nrows,&ia,&ja,&flg));
  }
  /* Set up distributed array for coarse grid */
  if (!Test_3D) {
    CHKERRQ(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,user.coarse.mx,user.coarse.my,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&user.coarse.da));
  } else {
    CHKERRQ(DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,user.coarse.mx,user.coarse.my,user.coarse.mz,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,NULL,&user.coarse.da));
  }
  CHKERRQ(DMSetFromOptions(user.coarse.da));
  CHKERRQ(DMSetUp(user.coarse.da));

  /* Create interpolation between the fine and coarse grids */
  CHKERRQ(DMCreateInterpolation(user.coarse.da,user.fine.da,&P,NULL));

  /* Get R = P^T */
  CHKERRQ(MatTranspose(P,MAT_INITIAL_MATRIX,&R));

  /* C = R*A*P */
  /* Developer's API */
  CHKERRQ(MatProductCreate(R,A,P,&D));
  CHKERRQ(MatProductSetType(D,MATPRODUCT_ABC));
  CHKERRQ(MatProductSetFromOptions(D));
  CHKERRQ(MatProductSymbolic(D));
  CHKERRQ(MatProductNumeric(D));
  CHKERRQ(MatProductNumeric(D)); /* Test reuse symbolic D */

  /* User's API */
  { /* Test MatMatMatMult_Basic() */
    Mat Adense,Cdense;
    CHKERRQ(MatConvert(A,MATDENSE,MAT_INITIAL_MATRIX,&Adense));
    CHKERRQ(MatMatMatMult(R,Adense,P,MAT_INITIAL_MATRIX,fill,&Cdense));
    CHKERRQ(MatMatMatMult(R,Adense,P,MAT_REUSE_MATRIX,fill,&Cdense));

    CHKERRQ(MatMultEqual(D,Cdense,10,&flg));
    PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"D*v != Cdense*v");
    CHKERRQ(MatDestroy(&Adense));
    CHKERRQ(MatDestroy(&Cdense));
  }

  CHKERRQ(MatMatMatMult(R,A,P,MAT_INITIAL_MATRIX,fill,&C));
  CHKERRQ(MatMatMatMult(R,A,P,MAT_REUSE_MATRIX,fill,&C));
  CHKERRQ(MatProductClear(C));

  /* Test D == C */
  CHKERRQ(MatEqual(D,C,&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"D != C");

  /* Test C == PtAP */
  CHKERRQ(MatPtAP(A,P,MAT_INITIAL_MATRIX,fill,&PtAP));
  CHKERRQ(MatPtAP(A,P,MAT_REUSE_MATRIX,fill,&PtAP));
  CHKERRQ(MatEqual(C,PtAP,&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"C != PtAP");
  CHKERRQ(MatDestroy(&PtAP));

  /* Clean up */
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(PetscRandomDestroy(&rdm));
  CHKERRQ(DMDestroy(&user.fine.da));
  CHKERRQ(DMDestroy(&user.coarse.da));
  CHKERRQ(MatDestroy(&P));
  CHKERRQ(MatDestroy(&R));
  CHKERRQ(MatDestroy(&C));
  CHKERRQ(MatDestroy(&D));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

   test:
      suffix: 2
      nsize: 2
      args: -matmatmatmult_via scalable

   test:
      suffix: 3
      nsize: 2
      args: -matmatmatmult_via nonscalable
      output_file: output/ex111_1.out

TEST*/
