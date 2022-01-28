
static char help[] ="Tests sequential and parallel DMCreateMatrix(), MatMatMult() and MatPtAP()\n\
  -Mx <xg>, where <xg> = number of coarse grid points in the x-direction\n\
  -My <yg>, where <yg> = number of coarse grid points in the y-direction\n\
  -Mz <zg>, where <zg> = number of coarse grid points in the z-direction\n\
  -Npx <npx>, where <npx> = number of processors in the x-direction\n\
  -Npy <npy>, where <npy> = number of processors in the y-direction\n\
  -Npz <npz>, where <npz> = number of processors in the z-direction\n\n";

/*
    This test is modified from ~src/ksp/tests/ex19.c.
    Example of usage: mpiexec -n 3 ./ex96 -Mx 10 -My 10 -Mz 10
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
  PetscInt       Npx=PETSC_DECIDE,Npy=PETSC_DECIDE,Npz=PETSC_DECIDE;
  PetscMPIInt    size,rank;
  PetscInt       m,n,M,N,i,nrows;
  PetscScalar    one = 1.0;
  PetscReal      fill=2.0;
  Mat            A,A_tmp,P,C,C1,C2;
  PetscScalar    *array,none = -1.0,alpha;
  Vec            x,v1,v2,v3,v4;
  PetscReal      norm,norm_tmp,norm_tmp1,tol=100.*PETSC_MACHINE_EPSILON;
  PetscRandom    rdm;
  PetscBool      Test_MatMatMult=PETSC_TRUE,Test_MatPtAP=PETSC_TRUE,Test_3D=PETSC_TRUE,flg;
  const PetscInt *ia,*ja;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  ierr = PetscOptionsGetReal(NULL,NULL,"-tol",&tol,NULL);CHKERRQ(ierr);

  user.ratio     = 2;
  user.coarse.mx = 20; user.coarse.my = 20; user.coarse.mz = 20;

  ierr = PetscOptionsGetInt(NULL,NULL,"-Mx",&user.coarse.mx,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-My",&user.coarse.my,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-Mz",&user.coarse.mz,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-ratio",&user.ratio,NULL);CHKERRQ(ierr);

  if (user.coarse.mz) Test_3D = PETSC_TRUE;

  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-Npx",&Npx,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-Npy",&Npy,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-Npz",&Npz,NULL);CHKERRQ(ierr);

  /* Set up distributed array for fine grid */
  if (!Test_3D) {
    ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,user.coarse.mx,user.coarse.my,Npx,Npy,1,1,NULL,NULL,&user.coarse.da);CHKERRQ(ierr);
  } else {
    ierr = DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,user.coarse.mx,user.coarse.my,user.coarse.mz,Npx,Npy,Npz,1,1,NULL,NULL,NULL,&user.coarse.da);CHKERRQ(ierr);
  }
  ierr = DMSetFromOptions(user.coarse.da);CHKERRQ(ierr);
  ierr = DMSetUp(user.coarse.da);CHKERRQ(ierr);

  /* This makes sure the coarse DMDA has the same partition as the fine DMDA */
  ierr = DMRefine(user.coarse.da,PetscObjectComm((PetscObject)user.coarse.da),&user.fine.da);CHKERRQ(ierr);

  /* Test DMCreateMatrix()                                         */
  /*------------------------------------------------------------*/
  ierr = DMSetMatType(user.fine.da,MATAIJ);CHKERRQ(ierr);
  ierr = DMCreateMatrix(user.fine.da,&A);CHKERRQ(ierr);
  ierr = DMSetMatType(user.fine.da,MATBAIJ);CHKERRQ(ierr);
  ierr = DMCreateMatrix(user.fine.da,&C);CHKERRQ(ierr);

  ierr = MatConvert(C,MATAIJ,MAT_INITIAL_MATRIX,&A_tmp);CHKERRQ(ierr); /* not work for mpisbaij matrix! */
  ierr = MatEqual(A,A_tmp,&flg);CHKERRQ(ierr);
  PetscAssertFalse(!flg,PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMETYPE,"A != C");
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&A_tmp);CHKERRQ(ierr);

  /*------------------------------------------------------------*/

  ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
  ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);
  /* if (rank == 0) printf("A %d, %d\n",M,N); */

  /* set val=one to A */
  if (size == 1) {
    ierr = MatGetRowIJ(A,0,PETSC_FALSE,PETSC_FALSE,&nrows,&ia,&ja,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = MatSeqAIJGetArray(A,&array);CHKERRQ(ierr);
      for (i=0; i<ia[nrows]; i++) array[i] = one;
      ierr = MatSeqAIJRestoreArray(A,&array);CHKERRQ(ierr);
    }
    ierr = MatRestoreRowIJ(A,0,PETSC_FALSE,PETSC_FALSE,&nrows,&ia,&ja,&flg);CHKERRQ(ierr);
  } else {
    Mat AA,AB;
    ierr = MatMPIAIJGetSeqAIJ(A,&AA,&AB,NULL);CHKERRQ(ierr);
    ierr = MatGetRowIJ(AA,0,PETSC_FALSE,PETSC_FALSE,&nrows,&ia,&ja,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = MatSeqAIJGetArray(AA,&array);CHKERRQ(ierr);
      for (i=0; i<ia[nrows]; i++) array[i] = one;
      ierr = MatSeqAIJRestoreArray(AA,&array);CHKERRQ(ierr);
    }
    ierr = MatRestoreRowIJ(AA,0,PETSC_FALSE,PETSC_FALSE,&nrows,&ia,&ja,&flg);CHKERRQ(ierr);
    ierr = MatGetRowIJ(AB,0,PETSC_FALSE,PETSC_FALSE,&nrows,&ia,&ja,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = MatSeqAIJGetArray(AB,&array);CHKERRQ(ierr);
      for (i=0; i<ia[nrows]; i++) array[i] = one;
      ierr = MatSeqAIJRestoreArray(AB,&array);CHKERRQ(ierr);
    }
    ierr = MatRestoreRowIJ(AB,0,PETSC_FALSE,PETSC_FALSE,&nrows,&ia,&ja,&flg);CHKERRQ(ierr);
  }
  /* ierr = MatView(A, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */

  /* Create interpolation between the fine and coarse grids */
  ierr = DMCreateInterpolation(user.coarse.da,user.fine.da,&P,NULL);CHKERRQ(ierr);
  ierr = MatGetLocalSize(P,&m,&n);CHKERRQ(ierr);
  ierr = MatGetSize(P,&M,&N);CHKERRQ(ierr);
  /* if (rank == 0) printf("P %d, %d\n",M,N); */

  /* Create vectors v1 and v2 that are compatible with A */
  ierr = VecCreate(PETSC_COMM_WORLD,&v1);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&m,NULL);CHKERRQ(ierr);
  ierr = VecSetSizes(v1,m,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(v1);CHKERRQ(ierr);
  ierr = VecDuplicate(v1,&v2);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rdm);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rdm);CHKERRQ(ierr);

  /* Test MatMatMult(): C = A*P */
  /*----------------------------*/
  if (Test_MatMatMult) {
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&A_tmp);CHKERRQ(ierr);
    ierr = MatMatMult(A_tmp,P,MAT_INITIAL_MATRIX,fill,&C);CHKERRQ(ierr);

    /* Test MAT_REUSE_MATRIX - reuse symbolic C */
    alpha=1.0;
    for (i=0; i<2; i++) {
      alpha -= 0.1;
      ierr   = MatScale(A_tmp,alpha);CHKERRQ(ierr);
      ierr   = MatMatMult(A_tmp,P,MAT_REUSE_MATRIX,fill,&C);CHKERRQ(ierr);
    }
    /* Free intermediate data structures created for reuse of C=Pt*A*P */
    ierr = MatProductClear(C);CHKERRQ(ierr);

    /* Test MatDuplicate()        */
    /*----------------------------*/
    ierr = MatDuplicate(C,MAT_COPY_VALUES,&C1);CHKERRQ(ierr);
    ierr = MatDuplicate(C1,MAT_COPY_VALUES,&C2);CHKERRQ(ierr);
    ierr = MatDestroy(&C1);CHKERRQ(ierr);
    ierr = MatDestroy(&C2);CHKERRQ(ierr);

    /* Create vector x that is compatible with P */
    ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
    ierr = MatGetLocalSize(P,NULL,&n);CHKERRQ(ierr);
    ierr = VecSetSizes(x,n,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = VecSetFromOptions(x);CHKERRQ(ierr);

    norm = 0.0;
    for (i=0; i<10; i++) {
      ierr      = VecSetRandom(x,rdm);CHKERRQ(ierr);
      ierr      = MatMult(P,x,v1);CHKERRQ(ierr);
      ierr      = MatMult(A_tmp,v1,v2);CHKERRQ(ierr); /* v2 = A*P*x */
      ierr      = MatMult(C,x,v1);CHKERRQ(ierr);  /* v1 = C*x   */
      ierr      = VecAXPY(v1,none,v2);CHKERRQ(ierr);
      ierr      = VecNorm(v1,NORM_1,&norm_tmp);CHKERRQ(ierr);
      ierr      = VecNorm(v2,NORM_1,&norm_tmp1);CHKERRQ(ierr);
      norm_tmp /= norm_tmp1;
      if (norm_tmp > norm) norm = norm_tmp;
    }
    if (norm >= tol && rank == 0) {
      ierr = PetscPrintf(PETSC_COMM_SELF,"Error: MatMatMult(), |v1 - v2|/|v2|: %g\n",(double)norm);CHKERRQ(ierr);
    }

    ierr = VecDestroy(&x);CHKERRQ(ierr);
    ierr = MatDestroy(&C);CHKERRQ(ierr);
    ierr = MatDestroy(&A_tmp);CHKERRQ(ierr);
  }

  /* Test P^T * A * P - MatPtAP() */
  /*------------------------------*/
  if (Test_MatPtAP) {
    ierr = MatPtAP(A,P,MAT_INITIAL_MATRIX,fill,&C);CHKERRQ(ierr);
    ierr = MatGetLocalSize(C,&m,&n);CHKERRQ(ierr);

    /* Test MAT_REUSE_MATRIX - reuse symbolic C */
    alpha=1.0;
    for (i=0; i<1; i++) {
      alpha -= 0.1;
      ierr   = MatScale(A,alpha);CHKERRQ(ierr);
      ierr   = MatPtAP(A,P,MAT_REUSE_MATRIX,fill,&C);CHKERRQ(ierr);
    }

    /* Free intermediate data structures created for reuse of C=Pt*A*P */
    ierr = MatProductClear(C);CHKERRQ(ierr);

    /* Test MatDuplicate()        */
    /*----------------------------*/
    ierr = MatDuplicate(C,MAT_COPY_VALUES,&C1);CHKERRQ(ierr);
    ierr = MatDuplicate(C1,MAT_COPY_VALUES,&C2);CHKERRQ(ierr);
    ierr = MatDestroy(&C1);CHKERRQ(ierr);
    ierr = MatDestroy(&C2);CHKERRQ(ierr);

    /* Create vector x that is compatible with P */
    ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
    ierr = MatGetLocalSize(P,&m,&n);CHKERRQ(ierr);
    ierr = VecSetSizes(x,n,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = VecSetFromOptions(x);CHKERRQ(ierr);

    ierr = VecCreate(PETSC_COMM_WORLD,&v3);CHKERRQ(ierr);
    ierr = VecSetSizes(v3,n,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = VecSetFromOptions(v3);CHKERRQ(ierr);
    ierr = VecDuplicate(v3,&v4);CHKERRQ(ierr);

    norm = 0.0;
    for (i=0; i<10; i++) {
      ierr = VecSetRandom(x,rdm);CHKERRQ(ierr);
      ierr = MatMult(P,x,v1);CHKERRQ(ierr);
      ierr = MatMult(A,v1,v2);CHKERRQ(ierr);  /* v2 = A*P*x */

      ierr = MatMultTranspose(P,v2,v3);CHKERRQ(ierr); /* v3 = Pt*A*P*x */
      ierr = MatMult(C,x,v4);CHKERRQ(ierr);           /* v3 = C*x   */
      ierr = VecAXPY(v4,none,v3);CHKERRQ(ierr);
      ierr = VecNorm(v4,NORM_1,&norm_tmp);CHKERRQ(ierr);
      ierr = VecNorm(v3,NORM_1,&norm_tmp1);CHKERRQ(ierr);

      norm_tmp /= norm_tmp1;
      if (norm_tmp > norm) norm = norm_tmp;
    }
    if (norm >= tol && rank == 0) {
      ierr = PetscPrintf(PETSC_COMM_SELF,"Error: MatPtAP(), |v3 - v4|/|v3|: %g\n",(double)norm);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&C);CHKERRQ(ierr);
    ierr = VecDestroy(&v3);CHKERRQ(ierr);
    ierr = VecDestroy(&v4);CHKERRQ(ierr);
    ierr = VecDestroy(&x);CHKERRQ(ierr);
  }

  /* Clean up */
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rdm);CHKERRQ(ierr);
  ierr = VecDestroy(&v1);CHKERRQ(ierr);
  ierr = VecDestroy(&v2);CHKERRQ(ierr);
  ierr = DMDestroy(&user.fine.da);CHKERRQ(ierr);
  ierr = DMDestroy(&user.coarse.da);CHKERRQ(ierr);
  ierr = MatDestroy(&P);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      args: -Mx 10 -My 5 -Mz 10
      output_file: output/ex96_1.out

   test:
      suffix: nonscalable
      nsize: 3
      args: -Mx 10 -My 5 -Mz 10
      output_file: output/ex96_1.out

   test:
      suffix: scalable
      nsize: 3
      args: -Mx 10 -My 5 -Mz 10 -matmatmult_via scalable -matptap_via scalable
      output_file: output/ex96_1.out

   test:
     suffix: seq_scalable
     nsize: 3
     args: -Mx 10 -My 5 -Mz 10 -matmatmult_via scalable -matptap_via scalable -inner_diag_mat_product_algorithm scalable -inner_offdiag_mat_product_algorithm scalable
     output_file: output/ex96_1.out

   test:
     suffix: seq_sorted
     nsize: 3
     args: -Mx 10 -My 5 -Mz 10 -matmatmult_via scalable -matptap_via scalable -inner_diag_mat_product_algorithm sorted -inner_offdiag_mat_product_algorithm sorted
     output_file: output/ex96_1.out

   test:
     suffix: seq_scalable_fast
     nsize: 3
     args: -Mx 10 -My 5 -Mz 10 -matmatmult_via scalable -matptap_via scalable -inner_diag_mat_product_algorithm scalable_fast -inner_offdiag_mat_product_algorithm scalable_fast
     output_file: output/ex96_1.out

   test:
     suffix: seq_heap
     nsize: 3
     args: -Mx 10 -My 5 -Mz 10 -matmatmult_via scalable -matptap_via scalable -inner_diag_mat_product_algorithm heap -inner_offdiag_mat_product_algorithm heap
     output_file: output/ex96_1.out

   test:
     suffix: seq_btheap
     nsize: 3
     args: -Mx 10 -My 5 -Mz 10 -matmatmult_via scalable -matptap_via scalable -inner_diag_mat_product_algorithm btheap -inner_offdiag_mat_product_algorithm btheap
     output_file: output/ex96_1.out

   test:
     suffix: seq_llcondensed
     nsize: 3
     args: -Mx 10 -My 5 -Mz 10 -matmatmult_via scalable -matptap_via scalable -inner_diag_mat_product_algorithm llcondensed -inner_offdiag_mat_product_algorithm llcondensed
     output_file: output/ex96_1.out

   test:
     suffix: seq_rowmerge
     nsize: 3
     args: -Mx 10 -My 5 -Mz 10 -matmatmult_via scalable -matptap_via scalable -inner_diag_mat_product_algorithm rowmerge -inner_offdiag_mat_product_algorithm rowmerge
     output_file: output/ex96_1.out

   test:
     suffix: allatonce
     nsize: 3
     args: -Mx 10 -My 5 -Mz 10 -matmatmult_via scalable -matptap_via allatonce
     output_file: output/ex96_1.out

   test:
     suffix: allatonce_merged
     nsize: 3
     args: -Mx 10 -My 5 -Mz 10 -matmatmult_via scalable -matptap_via allatonce_merged
     output_file: output/ex96_1.out

TEST*/
