
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

  PetscCall(PetscInitialize(&argc,&argv,NULL,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  /* Get size of fine grids and coarse grids */
  user.ratio     = 2;
  user.coarse.mx = 4; user.coarse.my = 4; user.coarse.mz = 4;

  PetscCall(PetscOptionsGetInt(NULL,NULL,"-Mx",&user.coarse.mx,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-My",&user.coarse.my,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-Mz",&user.coarse.mz,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-ratio",&user.ratio,NULL));
  if (user.coarse.mz) Test_3D = PETSC_TRUE;

  user.fine.mx = user.ratio*(user.coarse.mx-1)+1;
  user.fine.my = user.ratio*(user.coarse.my-1)+1;
  user.fine.mz = user.ratio*(user.coarse.mz-1)+1;

  if (rank == 0) {
    if (!Test_3D) {
      PetscCall(PetscPrintf(PETSC_COMM_SELF,"coarse grids: %" PetscInt_FMT " %" PetscInt_FMT "; fine grids: %" PetscInt_FMT " %" PetscInt_FMT "\n",user.coarse.mx,user.coarse.my,user.fine.mx,user.fine.my));
    } else {
      PetscCall(PetscPrintf(PETSC_COMM_SELF,"coarse grids: %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT "; fine grids: %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT "\n",user.coarse.mx,user.coarse.my,user.coarse.mz,user.fine.mx,user.fine.my,user.fine.mz));
    }
  }

  /* Set up distributed array for fine grid */
  if (!Test_3D) {
    PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,user.fine.mx,user.fine.my,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&user.fine.da));
  } else {
    PetscCall(DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,user.fine.mx,user.fine.my,user.fine.mz,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,
                           1,1,NULL,NULL,NULL,&user.fine.da));
  }
  PetscCall(DMSetFromOptions(user.fine.da));
  PetscCall(DMSetUp(user.fine.da));

  /* Create and set A at fine grids */
  PetscCall(DMSetMatType(user.fine.da,MATAIJ));
  PetscCall(DMCreateMatrix(user.fine.da,&A));
  PetscCall(MatGetLocalSize(A,&m,&n));
  PetscCall(MatGetSize(A,&M,&N));

  /* set val=one to A (replace with random values!) */
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD,&rdm));
  PetscCall(PetscRandomSetFromOptions(rdm));
  if (size == 1) {
    PetscCall(MatGetRowIJ(A,0,PETSC_FALSE,PETSC_FALSE,&nrows,&ia,&ja,&flg));
    if (flg) {
      PetscCall(MatSeqAIJGetArray(A,&array));
      for (i=0; i<ia[nrows]; i++) array[i] = one;
      PetscCall(MatSeqAIJRestoreArray(A,&array));
    }
    PetscCall(MatRestoreRowIJ(A,0,PETSC_FALSE,PETSC_FALSE,&nrows,&ia,&ja,&flg));
  } else {
    Mat AA,AB;
    PetscCall(MatMPIAIJGetSeqAIJ(A,&AA,&AB,NULL));
    PetscCall(MatGetRowIJ(AA,0,PETSC_FALSE,PETSC_FALSE,&nrows,&ia,&ja,&flg));
    if (flg) {
      PetscCall(MatSeqAIJGetArray(AA,&array));
      for (i=0; i<ia[nrows]; i++) array[i] = one;
      PetscCall(MatSeqAIJRestoreArray(AA,&array));
    }
    PetscCall(MatRestoreRowIJ(AA,0,PETSC_FALSE,PETSC_FALSE,&nrows,&ia,&ja,&flg));
    PetscCall(MatGetRowIJ(AB,0,PETSC_FALSE,PETSC_FALSE,&nrows,&ia,&ja,&flg));
    if (flg) {
      PetscCall(MatSeqAIJGetArray(AB,&array));
      for (i=0; i<ia[nrows]; i++) array[i] = one;
      PetscCall(MatSeqAIJRestoreArray(AB,&array));
    }
    PetscCall(MatRestoreRowIJ(AB,0,PETSC_FALSE,PETSC_FALSE,&nrows,&ia,&ja,&flg));
  }
  /* Set up distributed array for coarse grid */
  if (!Test_3D) {
    PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,user.coarse.mx,user.coarse.my,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&user.coarse.da));
  } else {
    PetscCall(DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,user.coarse.mx,user.coarse.my,user.coarse.mz,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,NULL,&user.coarse.da));
  }
  PetscCall(DMSetFromOptions(user.coarse.da));
  PetscCall(DMSetUp(user.coarse.da));

  /* Create interpolation between the fine and coarse grids */
  PetscCall(DMCreateInterpolation(user.coarse.da,user.fine.da,&P,NULL));

  /* Get R = P^T */
  PetscCall(MatTranspose(P,MAT_INITIAL_MATRIX,&R));

  /* C = R*A*P */
  /* Developer's API */
  PetscCall(MatProductCreate(R,A,P,&D));
  PetscCall(MatProductSetType(D,MATPRODUCT_ABC));
  PetscCall(MatProductSetFromOptions(D));
  PetscCall(MatProductSymbolic(D));
  PetscCall(MatProductNumeric(D));
  PetscCall(MatProductNumeric(D)); /* Test reuse symbolic D */

  /* User's API */
  { /* Test MatMatMatMult_Basic() */
    Mat Adense,Cdense;
    PetscCall(MatConvert(A,MATDENSE,MAT_INITIAL_MATRIX,&Adense));
    PetscCall(MatMatMatMult(R,Adense,P,MAT_INITIAL_MATRIX,fill,&Cdense));
    PetscCall(MatMatMatMult(R,Adense,P,MAT_REUSE_MATRIX,fill,&Cdense));

    PetscCall(MatMultEqual(D,Cdense,10,&flg));
    PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"D*v != Cdense*v");
    PetscCall(MatDestroy(&Adense));
    PetscCall(MatDestroy(&Cdense));
  }

  PetscCall(MatMatMatMult(R,A,P,MAT_INITIAL_MATRIX,fill,&C));
  PetscCall(MatMatMatMult(R,A,P,MAT_REUSE_MATRIX,fill,&C));
  PetscCall(MatProductClear(C));

  /* Test D == C */
  PetscCall(MatEqual(D,C,&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"D != C");

  /* Test C == PtAP */
  PetscCall(MatPtAP(A,P,MAT_INITIAL_MATRIX,fill,&PtAP));
  PetscCall(MatPtAP(A,P,MAT_REUSE_MATRIX,fill,&PtAP));
  PetscCall(MatEqual(C,PtAP,&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"C != PtAP");
  PetscCall(MatDestroy(&PtAP));

  /* Clean up */
  PetscCall(MatDestroy(&A));
  PetscCall(PetscRandomDestroy(&rdm));
  PetscCall(DMDestroy(&user.fine.da));
  PetscCall(DMDestroy(&user.coarse.da));
  PetscCall(MatDestroy(&P));
  PetscCall(MatDestroy(&R));
  PetscCall(MatDestroy(&C));
  PetscCall(MatDestroy(&D));
  PetscCall(PetscFinalize());
  return 0;
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
