
static char help[] = "Tests matrix permutation for factorization and solve on matrix with MatSBAIJ format. Modified from ex74.c\n";

#include "petscmat.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Vec         x,y,b;
  Mat         A;           /* linear system matrix */ 
  Mat         sA,sC;       /* symmetric part of the matrices */ 
  int         n,mbs=16,bs=1,nz=3,prob=1;
  int         ierr,i,j,col[3],size,block, row,I,J,n1,*ip_ptr;
  int         lf;          /* level of fill for icc */
  PetscReal   norm1,norm2,tol=1.e-10;
  PetscScalar neg_one = -1.0,four=4.0,value[3];  
  IS          perm;
  PetscRandom rand;
  PetscTruth  reorder=PETSC_TRUE;
  MatFactorInfo  factinfo;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(1,"This is a uniprocessor example only!");
  ierr = PetscOptionsGetInt(PETSC_NULL,"-bs",&bs,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-mbs",&mbs,PETSC_NULL);CHKERRQ(ierr);

  n = mbs*bs;
  ierr=MatCreateSeqBAIJ(PETSC_COMM_WORLD,bs,n,n,nz,PETSC_NULL, &A);CHKERRQ(ierr);
  ierr=MatCreateSeqSBAIJ(PETSC_COMM_WORLD,bs,n,n,nz,PETSC_NULL, &sA);CHKERRQ(ierr);

  /* Test MatGetOwnershipRange() */
  ierr = MatGetOwnershipRange(A,&I,&J);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(sA,&i,&j);CHKERRQ(ierr);
  if (i-I || j-J){
    PetscPrintf(PETSC_COMM_SELF,"Error: MatGetOwnershipRange() in MatSBAIJ format\n");
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

      i = 0; col[0] = 0; col[1] = 1; col[2]=n-1;
      value[0] = 2.0; value[1] = -1.0; value[2]=0.1;
      ierr = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatSetValues(sA,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
    }
    else if (prob ==2){ /* matrix for the five point stencil */
      n1 = (int) (sqrt((PetscReal)n) + 0.001); 
      if (n1*n1 - n) SETERRQ(PETSC_ERR_ARG_WRONG,"sqrt(n) must be a positive interger!"); 
      for (i=0; i<n1; i++) {
        for (j=0; j<n1; j++) {
          I = j + n1*i;
          if (i>0)   {
            J = I - n1; 
            ierr = MatSetValues(A,1,&I,1,&J,&neg_one,INSERT_VALUES);CHKERRQ(ierr); 
            ierr = MatSetValues(sA,1,&I,1,&J,&neg_one,INSERT_VALUES);CHKERRQ(ierr);
          }
          if (i<n1-1) {
            J = I + n1; 
            ierr = MatSetValues(A,1,&I,1,&J,&neg_one,INSERT_VALUES);CHKERRQ(ierr);
            ierr = MatSetValues(sA,1,&I,1,&J,&neg_one,INSERT_VALUES);CHKERRQ(ierr);
          }
          if (j>0)   {
            J = I - 1; 
            ierr = MatSetValues(A,1,&I,1,&J,&neg_one,INSERT_VALUES);CHKERRQ(ierr);
            ierr = MatSetValues(sA,1,&I,1,&J,&neg_one,INSERT_VALUES);CHKERRQ(ierr);
          }
          if (j<n1-1) {
            J = I + 1; 
            ierr = MatSetValues(A,1,&I,1,&J,&neg_one,INSERT_VALUES);CHKERRQ(ierr);
            ierr = MatSetValues(sA,1,&I,1,&J,&neg_one,INSERT_VALUES);CHKERRQ(ierr);
          }
          ierr = MatSetValues(A,1,&I,1,&I,&four,INSERT_VALUES);CHKERRQ(ierr);
          ierr = MatSetValues(sA,1,&I,1,&I,&four,INSERT_VALUES);CHKERRQ(ierr);
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
    if (bs == 2){
      /* insert a value to off-diag blocks */
      row = 2; col[0] = 5; value[0] = 0.01;
      ierr = MatSetValues(A,1,&row,1,col,value,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatSetValues(sA,1,&row,1,col,value,INSERT_VALUES);CHKERRQ(ierr);
      row = 0; col[0] = 3; value[0] = 0.01;
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
  /* PetscPrintf(PETSC_COMM_SELF,"\n Symmetric Part of Matrix: \n"); */
  /* MatView(sA, PETSC_VIEWER_DRAW_WORLD); */
  /* MatView(sA, PETSC_VIEWER_STDOUT_WORLD); */

  /* Vectors */
  ierr = PetscRandomCreate(PETSC_COMM_SELF,RANDOM_DEFAULT,&rand);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&b);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&y);CHKERRQ(ierr);
  ierr = VecSetRandom(rand,x);CHKERRQ(ierr);

  /* Test MatReordering() */
  ierr = PetscMalloc(mbs*sizeof(int),&ip_ptr);CHKERRQ(ierr);
  for (i=0; i<mbs; i++) ip_ptr[i] = i;
  if(reorder){
    i = ip_ptr[1]; ip_ptr[1] = ip_ptr[mbs-2]; ip_ptr[mbs-2] = i; 
    /* i = ip_ptr[0]; ip_ptr[0] = ip_ptr[mbs-1]; ip_ptr[mbs-1] = i; */
    /* i = ip_ptr[2]; ip_ptr[2] = ip_ptr[mbs-3]; ip_ptr[mbs-3] = i; */
  }  
  ierr = ISCreateGeneral(PETSC_COMM_SELF,mbs,ip_ptr,&perm);CHKERRQ(ierr);
  ierr = ISSetPermutation(perm);CHKERRQ(ierr);
  
  /* Test MatCholeskyFactor(), MatICCFactor() */
  norm1 = tol;
  for (lf=-1; lf<10*bs; lf += bs){ 
    if (lf==-1) {  /* Cholesky factor */
      factinfo.fill = 5.0;
      ierr = MatCholeskyFactorSymbolic(sA,perm,&factinfo,&sC);CHKERRQ(ierr);
    } else {       /* incomplete Cholesky factor */
      factinfo.fill   = 5.0;
      factinfo.levels = lf;
      ierr = MatICCFactorSymbolic(sA,perm,&factinfo,&sC);CHKERRQ(ierr);
    }      
    ierr = MatCholeskyFactorNumeric(sA,&sC);CHKERRQ(ierr);  
    /* MatView(sC, PETSC_VIEWER_DRAW_WORLD);  */ /* view factored matrix */
    /* MatView(sC, PETSC_VIEWER_STDOUT_WORLD); */
       
    ierr = MatMult(sA,x,b);CHKERRQ(ierr);
    ierr = MatSolve(sC,b,y);CHKERRQ(ierr); 
    if (bs == 1) {
      Vecs xx,bb;
      ierr = VecsCreateSeq(PETSC_COMM_SELF,n,4,&xx);CHKERRQ(ierr);
      ierr = VecsDuplicate(xx,&bb);CHKERRQ(ierr);
      ierr = MatSolves(sC,bb,xx);CHKERRQ(ierr); 
      ierr = VecsDestroy(xx);CHKERRQ(ierr); 
      ierr = VecsDestroy(bb);CHKERRQ(ierr); 
    }
    ierr = MatDestroy(sC);CHKERRQ(ierr);

    /* Check the error */
    ierr = VecAXPY(&neg_one,x,y);CHKERRQ(ierr);
    ierr = VecNorm(y,NORM_2,&norm2);CHKERRQ(ierr);
    /* printf("lf: %d, error: %g\n", lf,norm2); */
    if (10*norm1 < norm2 && lf-bs != -1){ 
      ierr = PetscPrintf(PETSC_COMM_SELF,"lf=%d, %d, Norm of error=%g, %g\n",lf-bs,lf,norm1,norm2);CHKERRQ(ierr); 
    }
    norm1 = norm2;
    if (norm2 < tol && lf != -1) break;
  } 

  ierr = ISDestroy(perm);CHKERRQ(ierr);
  ierr = PetscFree(ip_ptr);CHKERRQ(ierr);
  ierr = MatDestroy(A);CHKERRQ(ierr);
  ierr = MatDestroy(sA);CHKERRQ(ierr);
  ierr = VecDestroy(x);CHKERRQ(ierr);
  ierr = VecDestroy(y);CHKERRQ(ierr);  
  ierr = VecDestroy(b);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(rand);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
