
static char help[] ="Tests sequential and parallel DAGetMatrix(), MatMatMult() and MatPtAP()\n\
  -Mx <xg>, where <xg> = number of coarse grid points in the x-direction\n\
  -My <yg>, where <yg> = number of coarse grid points in the y-direction\n\
  -Mz <zg>, where <zg> = number of coarse grid points in the z-direction\n\
  -Npx <npx>, where <npx> = number of processors in the x-direction\n\
  -Npy <npy>, where <npy> = number of processors in the y-direction\n\
  -Npz <npz>, where <npz> = number of processors in the z-direction\n\n";

/*  
    This test is modified from ~src/ksp/examples/tests/ex19.c. 
    Example of usage: mpiexec -n 3 ex96 -Mx 10 -My 10 -Mz 10
*/

#include "petscksp.h"
#include "petscda.h"
#include "petscmg.h"
#include "../src/mat/impls/aij/seq/aij.h"
#include "../src/mat/impls/aij/mpi/mpiaij.h"

/* User-defined application contexts */
typedef struct {
   PetscInt   mx,my,mz;         /* number grid points in x, y and z direction */
   Vec        localX,localF;    /* local vectors with ghost region */
   DA         da;
   Vec        x,b,r;            /* global vectors */
   Mat        J;                /* Jacobian on grid */
} GridCtx;
typedef struct {
   GridCtx     fine;
   GridCtx     coarse;
   KSP         ksp_coarse;
   PetscInt    ratio;
   Mat         Ii;              /* interpolation from coarse to fine */
} AppCtx;

#define COARSE_LEVEL 0
#define FINE_LEVEL   1

/*
      Mm_ratio - ration of grid lines between fine and coarse grids.
*/
#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  AppCtx         user;                      
  PetscInt       Npx=PETSC_DECIDE,Npy=PETSC_DECIDE,Npz=PETSC_DECIDE;
  PetscMPIInt    size,rank;
  PetscInt       m,n,M,N,i,nrows,*ia,*ja; 
  PetscScalar    one = 1.0;
  PetscReal      fill=2.0;
  Mat            A,A_tmp,P,C,C1,C2;
  PetscScalar    *array,none = -1.0,alpha;
  Vec           x,v1,v2,v3,v4;
  PetscReal     norm,norm_tmp,norm_tmp1,tol=1.e-12;
  PetscRandom   rdm;
  PetscTruth    Test_MatMatMult=PETSC_TRUE,Test_MatPtAP=PETSC_TRUE,Test_3D=PETSC_FALSE,flg;

  PetscInitialize(&argc,&argv,PETSC_NULL,help);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-tol",&tol,PETSC_NULL);CHKERRQ(ierr);

  user.ratio = 2;
  user.coarse.mx = 2; user.coarse.my = 2; user.coarse.mz = 0;
  ierr = PetscOptionsGetInt(PETSC_NULL,"-Mx",&user.coarse.mx,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-My",&user.coarse.my,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-Mz",&user.coarse.mz,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-ratio",&user.ratio,PETSC_NULL);CHKERRQ(ierr);
  if (user.coarse.mz) Test_3D = PETSC_TRUE;

  user.fine.mx = user.ratio*(user.coarse.mx-1)+1; 
  user.fine.my = user.ratio*(user.coarse.my-1)+1;
  user.fine.mz = user.ratio*(user.coarse.mz-1)+1;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-Npx",&Npx,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-Npy",&Npy,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-Npz",&Npz,PETSC_NULL);CHKERRQ(ierr);

  /* Set up distributed array for fine grid */
  if (!Test_3D){
    ierr = DACreate2d(PETSC_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_STAR,user.fine.mx,
                    user.fine.my,Npx,Npy,1,1,PETSC_NULL,PETSC_NULL,&user.fine.da);CHKERRQ(ierr);
  } else {
    ierr = DACreate3d(PETSC_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_STAR,
                    user.fine.mx,user.fine.my,user.fine.mz,Npx,Npy,Npz,
                    1,1,PETSC_NULL,PETSC_NULL,PETSC_NULL,&user.fine.da);CHKERRQ(ierr);
  }

  /* Test DAGetMatrix()                                         */
  /*------------------------------------------------------------*/
  ierr = DAGetMatrix(user.fine.da,MATAIJ,&A);CHKERRQ(ierr);
  ierr = DAGetMatrix(user.fine.da,MATBAIJ,&C);CHKERRQ(ierr);
  
  ierr = MatConvert(C,MATAIJ,MAT_INITIAL_MATRIX,&A_tmp);CHKERRQ(ierr); /* not work for mpisbaij matrix! */
  ierr = MatEqual(A,A_tmp,&flg);CHKERRQ(ierr);
  if (!flg) {
    SETERRQ(PETSC_ERR_ARG_NOTSAMETYPE,"A != C"); 
  }
  ierr = MatDestroy(C);CHKERRQ(ierr);
  ierr = MatDestroy(A_tmp);CHKERRQ(ierr);
  
  /*------------------------------------------------------------*/
  
  ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
  ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);
  /* set val=one to A */
  if (size == 1){
    ierr = MatGetRowIJ(A,0,PETSC_FALSE,PETSC_FALSE,&nrows,&ia,&ja,&flg);
    if (flg){
      ierr = MatGetArray(A,&array);CHKERRQ(ierr);
      for (i=0; i<ia[nrows]; i++) array[i] = one;
      ierr = MatRestoreArray(A,&array);CHKERRQ(ierr);
    }
    ierr = MatRestoreRowIJ(A,0,PETSC_FALSE,PETSC_FALSE,&nrows,&ia,&ja,&flg);
  } else {
    Mat_MPIAIJ *aij = (Mat_MPIAIJ*)A->data;
    Mat_SeqAIJ *a=(Mat_SeqAIJ*)(aij->A)->data, *b=(Mat_SeqAIJ*)(aij->B)->data;
    /* A_part */
    for (i=0; i<a->i[m]; i++) a->a[i] = one;
    /* B_part */
    for (i=0; i<b->i[m]; i++) b->a[i] = one;
    
  }
  /* ierr = MatView(A, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */

  /* Set up distributed array for coarse grid */
  if (!Test_3D){
    ierr = DACreate2d(PETSC_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_STAR,user.coarse.mx,
                    user.coarse.my,Npx,Npy,1,1,PETSC_NULL,PETSC_NULL,&user.coarse.da);CHKERRQ(ierr);
  } else {
    ierr = DACreate3d(PETSC_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_STAR,
                    user.coarse.mx,user.coarse.my,user.coarse.mz,Npx,Npy,Npz,
                    1,1,PETSC_NULL,PETSC_NULL,PETSC_NULL,&user.coarse.da);CHKERRQ(ierr);
  }

  /* Create interpolation between the levels */
  ierr = DAGetInterpolation(user.coarse.da,user.fine.da,&P,PETSC_NULL);CHKERRQ(ierr);
  
  ierr = MatGetLocalSize(P,&m,&n);CHKERRQ(ierr);
  ierr = MatGetSize(P,&M,&N);CHKERRQ(ierr);

  /* Create vectors v1 and v2 that are compatible with A */
  ierr = VecCreate(PETSC_COMM_WORLD,&v1);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&m,PETSC_NULL);CHKERRQ(ierr);
  ierr = VecSetSizes(v1,m,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(v1);CHKERRQ(ierr);
  ierr = VecDuplicate(v1,&v2);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rdm);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rdm);CHKERRQ(ierr);

  /* Test MatMatMult(): C = A*P */
  /*----------------------------*/
  if (Test_MatMatMult){
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&A_tmp);CHKERRQ(ierr);
    ierr = MatMatMult(A_tmp,P,MAT_INITIAL_MATRIX,fill,&C);CHKERRQ(ierr);
    
    /* Test MAT_REUSE_MATRIX - reuse symbolic C */
    alpha=1.0;
    for (i=0; i<2; i++){
      alpha -=0.1;
      ierr = MatScale(A_tmp,alpha);CHKERRQ(ierr);
      ierr = MatMatMult(A_tmp,P,MAT_REUSE_MATRIX,fill,&C);CHKERRQ(ierr);
    }

    /* Test MatDuplicate()        */
    /*----------------------------*/
    ierr = MatDuplicate(C,MAT_COPY_VALUES,&C1);CHKERRQ(ierr);
    ierr = MatDuplicate(C1,MAT_COPY_VALUES,&C2);CHKERRQ(ierr);
    ierr = MatDestroy(C1);CHKERRQ(ierr);
    ierr = MatDestroy(C2);CHKERRQ(ierr);

    /* Create vector x that is compatible with P */
    ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
    ierr = MatGetLocalSize(P,PETSC_NULL,&n);CHKERRQ(ierr);
    ierr = VecSetSizes(x,n,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = VecSetFromOptions(x);CHKERRQ(ierr);

    norm = 0.0;
    for (i=0; i<10; i++) {
      ierr = VecSetRandom(x,rdm);CHKERRQ(ierr);
      ierr = MatMult(P,x,v1);CHKERRQ(ierr);  
      ierr = MatMult(A_tmp,v1,v2);CHKERRQ(ierr);  /* v2 = A*P*x */
      ierr = MatMult(C,x,v1);CHKERRQ(ierr);       /* v1 = C*x   */
      ierr = VecAXPY(v1,none,v2);CHKERRQ(ierr);
      ierr = VecNorm(v1,NORM_1,&norm_tmp);CHKERRQ(ierr);
      ierr = VecNorm(v2,NORM_1,&norm_tmp1);CHKERRQ(ierr);
      norm_tmp /= norm_tmp1;
      if (norm_tmp > norm) norm = norm_tmp;
    }
    if (norm >= tol && !rank) {
      ierr = PetscPrintf(PETSC_COMM_SELF,"Error: MatMatMult(), |v1 - v2|/|v2|: %G\n",norm);CHKERRQ(ierr);
    }
    
    ierr = VecDestroy(x);CHKERRQ(ierr);
    ierr = MatDestroy(C);CHKERRQ(ierr);
    ierr = MatDestroy(A_tmp);CHKERRQ(ierr);
  }

  /* Test P^T * A * P - MatPtAP() */ 
  /*------------------------------*/
  if (Test_MatPtAP){
    ierr = MatPtAP(A,P,MAT_INITIAL_MATRIX,fill,&C);CHKERRQ(ierr); 
    ierr = MatGetLocalSize(C,&m,&n);CHKERRQ(ierr);
    
    /* Test MAT_REUSE_MATRIX - reuse symbolic C */
    alpha=1.0;
    for (i=0; i<1; i++){
      alpha -=0.1;
      ierr = MatScale(A,alpha);CHKERRQ(ierr);
      ierr = MatPtAP(A,P,MAT_REUSE_MATRIX,fill,&C);CHKERRQ(ierr);
    }

        /* Test MatDuplicate()        */
    /*----------------------------*/
    ierr = MatDuplicate(C,MAT_COPY_VALUES,&C1);CHKERRQ(ierr);
    ierr = MatDuplicate(C1,MAT_COPY_VALUES,&C2);CHKERRQ(ierr); 
    ierr = MatDestroy(C1);CHKERRQ(ierr); 
    ierr = MatDestroy(C2);CHKERRQ(ierr); 

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
    if (norm >= tol && !rank) {
      ierr = PetscPrintf(PETSC_COMM_SELF,"Error: MatPtAP(), |v3 - v4|/|v3|: %G\n",norm);CHKERRQ(ierr);
    }
  
    ierr = MatDestroy(C);CHKERRQ(ierr);
    ierr = VecDestroy(v3);CHKERRQ(ierr);
    ierr = VecDestroy(v4);CHKERRQ(ierr);
    ierr = VecDestroy(x);CHKERRQ(ierr);
 
  }

  /* Clean up */
   ierr = MatDestroy(A);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(rdm);CHKERRQ(ierr);
  ierr = VecDestroy(v1);CHKERRQ(ierr);
  ierr = VecDestroy(v2);CHKERRQ(ierr);
  ierr = DADestroy(user.fine.da);CHKERRQ(ierr);
  ierr = DADestroy(user.coarse.da);CHKERRQ(ierr);
  ierr = MatDestroy(P);CHKERRQ(ierr); 

  ierr = PetscFinalize();CHKERRQ(ierr);

  return 0;
}
