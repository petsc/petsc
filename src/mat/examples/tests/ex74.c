
static char help[] = "Tests the various sequential routines in MatSBAIJ format.\n";

#include "petscmat.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  PetscMPIInt        size;
  PetscErrorCode     ierr;
  Vec                x,y,b,s1,s2;      
  Mat                A;             /* linear system matrix */ 
  Mat                sA,sB,sC;         /* symmetric part of the matrices */ 
  PetscInt           n,mbs=16,bs=1,nz=3,prob=1,i,j,col[3],lf,block, row,Ii,J,n1,inc; 
  PetscReal          norm1,norm2,rnorm,tol=1.e-10;
  PetscScalar        neg_one = -1.0,four=4.0,value[3];  
  IS                 perm, iscol;
  PetscRandom        rdm;
  PetscTruth         doIcc=PETSC_TRUE,equal;
  MatInfo            minfo1,minfo2;
  MatFactorInfo      factinfo;
  const MatType      type;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_ERR_SUP,"This is a uniprocessor example only!");
  ierr = PetscOptionsGetInt(PETSC_NULL,"-bs",&bs,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-mbs",&mbs,PETSC_NULL);CHKERRQ(ierr);

  n = mbs*bs;
  ierr=MatCreateSeqBAIJ(PETSC_COMM_SELF,bs,n,n,nz,PETSC_NULL, &A);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_SELF,&sA);CHKERRQ(ierr);
  ierr = MatSetSizes(sA,n,n,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetType(sA,MATSEQSBAIJ);CHKERRQ(ierr);
  ierr = MatSetFromOptions(sA);CHKERRQ(ierr); 
  ierr = MatGetType(sA,&type);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)sA,MATSEQSBAIJ,&doIcc);CHKERRQ(ierr);
  ierr = MatSeqSBAIJSetPreallocation(sA,bs,nz,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatSetOption(sA,MAT_IGNORE_LOWER_TRIANGULAR,PETSC_TRUE);CHKERRQ(ierr);

  /* Test MatGetOwnershipRange() */
  ierr = MatGetOwnershipRange(A,&Ii,&J);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(sA,&i,&j);CHKERRQ(ierr);
  if (i-Ii || j-J){
    ierr = PetscPrintf(PETSC_COMM_SELF,"Error: MatGetOwnershipRange() in MatSBAIJ format\n");CHKERRQ(ierr);
  }

  /* Assemble matrix */
  if (bs == 1){
    ierr = PetscOptionsGetInt(PETSC_NULL,"-test_problem",&prob,PETSC_NULL);CHKERRQ(ierr);
    if (prob == 1){ /* tridiagonal matrix */
      value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
      for (i=1; i<n-1; i++) {
        col[0] = i-1; col[1] = i; col[2] = i+1;
        ierr = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
        ierr = MatSetValues(sA,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
      }
      i = n - 1; col[0]=0; col[1] = n - 2; col[2] = n - 1;
      value[0]= 0.1; value[1]=-1; value[2]=2;
      ierr = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatSetValues(sA,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);

      i = 0; 
      col[0] = n-1;  col[1] = 1; col[2]=0; 
      value[0] = 0.1; value[1] = -1.0; value[2]=2;
      ierr = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatSetValues(sA,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
    }
    else if (prob ==2){ /* matrix for the five point stencil */
      n1 = (int) (sqrt((PetscReal)n) + 0.001); 
      if (n1*n1 - n) SETERRQ(PETSC_ERR_ARG_WRONG,"sqrt(n) must be a positive interger!"); 
      for (i=0; i<n1; i++) {
        for (j=0; j<n1; j++) {
          Ii = j + n1*i;
          if (i>0)   {
            J = Ii - n1; 
            ierr = MatSetValues(A,1,&Ii,1,&J,&neg_one,INSERT_VALUES);CHKERRQ(ierr); 
            ierr = MatSetValues(sA,1,&Ii,1,&J,&neg_one,INSERT_VALUES);CHKERRQ(ierr);
          }
          if (i<n1-1) {
            J = Ii + n1; 
            ierr = MatSetValues(A,1,&Ii,1,&J,&neg_one,INSERT_VALUES);CHKERRQ(ierr);
            ierr = MatSetValues(sA,1,&Ii,1,&J,&neg_one,INSERT_VALUES);CHKERRQ(ierr);
          }
          if (j>0)   {
            J = Ii - 1; 
            ierr = MatSetValues(A,1,&Ii,1,&J,&neg_one,INSERT_VALUES);CHKERRQ(ierr);
            ierr = MatSetValues(sA,1,&Ii,1,&J,&neg_one,INSERT_VALUES);CHKERRQ(ierr);
          }
          if (j<n1-1) {
            J = Ii + 1; 
            ierr = MatSetValues(A,1,&Ii,1,&J,&neg_one,INSERT_VALUES);CHKERRQ(ierr);
            ierr = MatSetValues(sA,1,&Ii,1,&J,&neg_one,INSERT_VALUES);CHKERRQ(ierr);
          }
          ierr = MatSetValues(A,1,&Ii,1,&Ii,&four,INSERT_VALUES);CHKERRQ(ierr);
          ierr = MatSetValues(sA,1,&Ii,1,&Ii,&four,INSERT_VALUES);CHKERRQ(ierr);
        }
      }                   
    }
  } 
  else { /* bs > 1 */
    for (block=0; block<n/bs; block++){
      /* diagonal blocks */
      value[0] = -1.0; value[1] = 4.0; value[2] = -1.0;
      for (i=1+block*bs; i<bs-1+block*bs; i++) {
        col[0] = i-1; col[1] = i; col[2] = i+1;
        ierr = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
        ierr = MatSetValues(sA,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);    
      }
      i = bs - 1+block*bs; col[0] = bs - 2+block*bs; col[1] = bs - 1+block*bs;
      value[0]=-1.0; value[1]=4.0;  
      ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatSetValues(sA,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr); 

      i = 0+block*bs; col[0] = 0+block*bs; col[1] = 1+block*bs; 
      value[0]=4.0; value[1] = -1.0; 
      ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatSetValues(sA,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);  
    }
    /* off-diagonal blocks */
    value[0]=-1.0;
    for (i=0; i<(n/bs-1)*bs; i++){
      col[0]=i+bs;
      ierr = MatSetValues(A,1,&i,1,col,value,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatSetValues(sA,1,&i,1,col,value,INSERT_VALUES);CHKERRQ(ierr);
      col[0]=i; row=i+bs;
      ierr = MatSetValues(A,1,&row,1,col,value,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatSetValues(sA,1,&row,1,col,value,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  /* PetscPrintf(PETSC_COMM_SELF,"\n The Matrix: \n");
  MatView(A, PETSC_VIEWER_DRAW_WORLD);
  MatView(A, PETSC_VIEWER_STDOUT_WORLD); */ 

  ierr = MatAssemblyBegin(sA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(sA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);  
  /* PetscPrintf(PETSC_COMM_SELF,"\n Symmetric Part of Matrix: \n");
  MatView(sA, PETSC_VIEWER_DRAW_WORLD); 
  MatView(sA, PETSC_VIEWER_STDOUT_WORLD); 
  */

  /* Test MatDuplicate() */
  ierr = MatNorm(A,NORM_FROBENIUS,&norm1);CHKERRQ(ierr); 
  ierr = MatDuplicate(sA,MAT_COPY_VALUES,&sB);CHKERRQ(ierr);
  ierr = MatEqual(sA,sB,&equal);CHKERRQ(ierr);
  if (!equal) SETERRQ(PETSC_ERR_ARG_NOTSAMETYPE,"Error in MatDuplicate()");

  /* Test MatNorm() */
  ierr = MatNorm(A,NORM_FROBENIUS,&norm1);CHKERRQ(ierr); 
  ierr = MatNorm(sB,NORM_FROBENIUS,&norm2);CHKERRQ(ierr);
  rnorm = PetscAbsScalar(norm1-norm2)/norm2;
  if (rnorm > tol){ 
    ierr = PetscPrintf(PETSC_COMM_SELF,"Error: MatNorm_FROBENIUS, NormA=%16.14e NormsB=%16.14e\n",norm1,norm2);CHKERRQ(ierr);
  }
  ierr = MatNorm(A,NORM_INFINITY,&norm1);CHKERRQ(ierr);
  ierr = MatNorm(sB,NORM_INFINITY,&norm2);CHKERRQ(ierr);
  rnorm = PetscAbsScalar(norm1-norm2)/norm2;
  if (rnorm > tol){ 
    ierr = PetscPrintf(PETSC_COMM_SELF,"Error: MatNorm_INFINITY(), NormA=%16.14e NormsB=%16.14e\n",norm1,norm2);CHKERRQ(ierr);
  }
  ierr = MatNorm(A,NORM_1,&norm1);CHKERRQ(ierr);
  ierr = MatNorm(sB,NORM_1,&norm2);CHKERRQ(ierr);
  rnorm = PetscAbsScalar(norm1-norm2)/norm2;
  if (rnorm > tol){ 
    ierr = PetscPrintf(PETSC_COMM_SELF,"Error: MatNorm_INFINITY(), NormA=%16.14e NormsB=%16.14e\n",norm1,norm2);CHKERRQ(ierr);
  }

  /* Test MatGetInfo(), MatGetSize(), MatGetBlockSize() */
  ierr = MatGetInfo(A,MAT_LOCAL,&minfo1);CHKERRQ(ierr);
  ierr = MatGetInfo(sB,MAT_LOCAL,&minfo2);CHKERRQ(ierr);
  /*
  printf("matrix nonzeros (BAIJ format) = %d, allocated nonzeros= %d\n", (int)minfo1.nz_used,(int)minfo1.nz_allocated); 
  printf("matrix nonzeros(SBAIJ format) = %d, allocated nonzeros= %d\n", (int)minfo2.nz_used,(int)minfo2.nz_allocated); 
  */
  i = (int) (minfo1.nz_used - minfo2.nz_used); 
  j = (int) (minfo2.nz_allocated - minfo2.nz_used);
  if (i<0 || j<0) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"Error: MatGetInfo()\n");CHKERRQ(ierr);
  }

  ierr = MatGetSize(A,&Ii,&J);CHKERRQ(ierr);
  ierr = MatGetSize(sB,&i,&j);CHKERRQ(ierr); 
  if (i-Ii || j-J) {
    PetscPrintf(PETSC_COMM_SELF,"Error: MatGetSize()\n");CHKERRQ(ierr);
  }
 
  ierr = MatGetBlockSize(A, &Ii);CHKERRQ(ierr);
  ierr = MatGetBlockSize(sB, &i);CHKERRQ(ierr);
  if (i-Ii){
    ierr = PetscPrintf(PETSC_COMM_SELF,"Error: MatGetBlockSize()\n");CHKERRQ(ierr);
  }

  ierr = PetscRandomCreate(PETSC_COMM_SELF,&rdm);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rdm);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&x);CHKERRQ(ierr);     
  ierr = VecDuplicate(x,&s1);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&s2);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&y);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&b);CHKERRQ(ierr);
  ierr = VecSetRandom(x,rdm);CHKERRQ(ierr);

  /* Test MatDiagonalScale(), MatGetDiagonal(), MatScale() */
#if !defined(PETSC_USE_COMPLEX)
  /* Scaling matrix with complex numbers results non-spd matrix, 
     causing crash of MatForwardSolve() and MatBackwardSolve() */
  ierr = MatDiagonalScale(A,x,x);CHKERRQ(ierr);
  ierr = MatDiagonalScale(sB,x,x);CHKERRQ(ierr); 
  ierr = MatMultEqual(A,sB,10,&equal);CHKERRQ(ierr);
  if (!equal) SETERRQ(PETSC_ERR_ARG_NOTSAMETYPE,"Error in MatDiagonalScale");

  ierr = MatGetDiagonal(A,s1);CHKERRQ(ierr);  
  ierr = MatGetDiagonal(sB,s2);CHKERRQ(ierr);  
  ierr = VecAXPY(s2,neg_one,s1);CHKERRQ(ierr);
  ierr = VecNorm(s2,NORM_1,&norm1);CHKERRQ(ierr);
  if ( norm1>tol) { 
    ierr = PetscPrintf(PETSC_COMM_SELF,"Error:MatGetDiagonal(), ||s1-s2||=%G\n",norm1);CHKERRQ(ierr);
  }

  {
    PetscScalar alpha=0.1;
    ierr = MatScale(A,alpha);CHKERRQ(ierr);
    ierr = MatScale(sB,alpha);CHKERRQ(ierr);
  }
#endif

  /* Test MatGetRowMaxAbs() */
  ierr = MatGetRowMaxAbs(A,s1,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatGetRowMaxAbs(sB,s2,PETSC_NULL);CHKERRQ(ierr); 
  ierr = VecNorm(s1,NORM_1,&norm1);CHKERRQ(ierr);
  ierr = VecNorm(s2,NORM_1,&norm2);CHKERRQ(ierr);
  norm1 -= norm2;
  if (norm1<-tol || norm1>tol) { 
    ierr = PetscPrintf(PETSC_COMM_SELF,"Error:MatGetRowMaxAbs() \n");CHKERRQ(ierr);
  } 

  /* Test MatMult() */
  for (i=0; i<40; i++) { 
    ierr = VecSetRandom(x,rdm);CHKERRQ(ierr);
    ierr = MatMult(A,x,s1);CHKERRQ(ierr);
    ierr = MatMult(sB,x,s2);CHKERRQ(ierr);
    ierr = VecNorm(s1,NORM_1,&norm1);CHKERRQ(ierr);
    ierr = VecNorm(s2,NORM_1,&norm2);CHKERRQ(ierr);
    norm1 -= norm2;
    if (norm1<-tol || norm1>tol) { 
      ierr = PetscPrintf(PETSC_COMM_SELF,"Error: MatMult(), norm1-norm2: %G\n",norm1);CHKERRQ(ierr);
    }
  }  

  /* MatMultAdd() */
  for (i=0; i<40; i++) {
    ierr = VecSetRandom(x,rdm);CHKERRQ(ierr);
    ierr = VecSetRandom(y,rdm);CHKERRQ(ierr);
    ierr = MatMultAdd(A,x,y,s1);CHKERRQ(ierr);
    ierr = MatMultAdd(sB,x,y,s2);CHKERRQ(ierr);
    ierr = VecNorm(s1,NORM_1,&norm1);CHKERRQ(ierr);
    ierr = VecNorm(s2,NORM_1,&norm2);CHKERRQ(ierr);
    norm1 -= norm2;
    if (norm1<-tol || norm1>tol) { 
      ierr = PetscPrintf(PETSC_COMM_SELF,"Error:MatMultAdd(),  norm1-norm2: %G\n",norm1);CHKERRQ(ierr);
    } 
  }

  /* Test MatCholeskyFactor(), MatICCFactor() with natural ordering */
  ierr = MatGetOrdering(A,MATORDERING_NATURAL,&perm,&iscol);CHKERRQ(ierr); 
  ierr = ISDestroy(iscol);CHKERRQ(ierr);
  norm1 = tol;  
  inc   = bs;

  /* initialize factinfo */
  ierr = PetscMemzero(&factinfo,sizeof(MatFactorInfo));CHKERRQ(ierr);

  for (lf=-1; lf<10; lf += inc){   
    if (lf==-1) {  /* Cholesky factor of sB (duplicate sA) */
      factinfo.fill = 5.0;   
      ierr = MatGetFactor(sB,MAT_SOLVER_PETSC,MAT_FACTOR_CHOLESKY,&sC);CHKERRQ(ierr);
      ierr = MatCholeskyFactorSymbolic(sC,sB,perm,&factinfo);CHKERRQ(ierr); 
    } else if (!doIcc){
      break;
    } else {       /* incomplete Cholesky factor */
      factinfo.fill   = 5.0;
      factinfo.levels = lf;
      ierr = MatGetFactor(sB,MAT_SOLVER_PETSC,MAT_FACTOR_ICC,&sC);CHKERRQ(ierr);
      ierr = MatICCFactorSymbolic(sC,sB,perm,&factinfo);CHKERRQ(ierr);
    }
    ierr = MatCholeskyFactorNumeric(sC,sB,&factinfo);CHKERRQ(ierr);
    /* MatView(sC, PETSC_VIEWER_DRAW_WORLD); */

    /* test MatGetDiagonal on numeric factor */
    /*
    if (lf == -1) {
      ierr = MatGetDiagonal(sC,s1);CHKERRQ(ierr);  
      printf(" in ex74.c, diag: \n");
      ierr = VecView(s1,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
    }
    */

    ierr = MatMult(sB,x,b);CHKERRQ(ierr);

    /* test MatForwardSolve() and MatBackwardSolve() */
    if (lf == -1){
      ierr = MatForwardSolve(sC,b,s1);CHKERRQ(ierr);
      ierr = MatBackwardSolve(sC,s1,s2);CHKERRQ(ierr);      
      ierr = VecAXPY(s2,neg_one,x);CHKERRQ(ierr);
      ierr = VecNorm(s2,NORM_2,&norm2);CHKERRQ(ierr);
      if (10*norm1 < norm2){
        ierr = PetscPrintf(PETSC_COMM_SELF,"MatForwardSolve and BackwardSolve: Norm of error=%G, bs=%d\n",norm2,bs);CHKERRQ(ierr); 
      }
    } 

    /* test MatSolve() */
    ierr = MatSolve(sC,b,y);CHKERRQ(ierr);
    ierr = MatDestroy(sC);CHKERRQ(ierr);
    /* Check the error */
    ierr = VecAXPY(y,neg_one,x);CHKERRQ(ierr);
    ierr = VecNorm(y,NORM_2,&norm2);CHKERRQ(ierr);
    /* printf("lf: %d, error: %G\n", lf,norm2); */
    if (10*norm1 < norm2 && lf-inc != -1){
      ierr = PetscPrintf(PETSC_COMM_SELF,"lf=%D, %D, Norm of error=%G, %G\n",lf-inc,lf,norm1,norm2);CHKERRQ(ierr); 
    } 
    norm1 = norm2;
    if (norm2 < tol && lf != -1) break;
  } 

  ierr = ISDestroy(perm);CHKERRQ(ierr);

  ierr = MatDestroy(A);CHKERRQ(ierr);
  ierr = MatDestroy(sB);CHKERRQ(ierr); 
  ierr = MatDestroy(sA);CHKERRQ(ierr);
  ierr = VecDestroy(x);CHKERRQ(ierr);
  ierr = VecDestroy(y);CHKERRQ(ierr);
  ierr = VecDestroy(s1);CHKERRQ(ierr);
  ierr = VecDestroy(s2);CHKERRQ(ierr);
  ierr = VecDestroy(b);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(rdm);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
