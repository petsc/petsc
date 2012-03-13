
static char help[] = "Tests sequential and parallel MatMatMult() and MatPtAP(), MatTransposeMatMult(), sequential MatMatTransposeMult(), MatRARt()\n\
Input arguments are:\n\
  -f0 <input_file> -f1 <input_file> -f2 <input_file> -f3 <input_file> : file to load\n\n";
/* Example of usage:
   ./ex94 -f0 <A_binary> -f1 <B_binary> -matmatmult_mat_view_info -matmatmulttr_mat_view_info
   mpiexec -n 3 ./ex94 -f0 medium -f1 medium -f2 arco1 -f3 arco1 -matmatmult_mat_view_info
*/

#include <petscmat.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat            A,A_save,B,P,R,C,C1;
  Vec            x,v1,v2,v3,v4;
  PetscViewer    viewer;
  PetscErrorCode ierr;
  PetscMPIInt    size,rank;
  PetscInt       i,m,n,j,*idxn,M,N,nzp,rstart,rend;
  PetscReal      norm,norm_abs,norm_tmp,tol=1.e-8,fill=4.0;
  PetscRandom    rdm;
  char           file[4][128];
  PetscBool      flg,preload = PETSC_TRUE;
  PetscScalar    *a,rval,alpha,none = -1.0;
  PetscBool      Test_MatMatMult=PETSC_TRUE,Test_MatMatTr=PETSC_TRUE,Test_MatPtAP=PETSC_TRUE,Test_MatRARt=PETSC_TRUE;
  PetscInt       pm,pn,pM,pN;
  MatInfo        info;
  
  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  ierr = PetscOptionsGetReal(PETSC_NULL,"-fill",&fill,PETSC_NULL);CHKERRQ(ierr);

  /*  Load the matrices A_save and B */
  ierr = PetscOptionsGetString(PETSC_NULL,"-f0",file[0],128,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"Must indicate a file name for small matrix A with the -f0 option.");
  ierr = PetscOptionsGetString(PETSC_NULL,"-f1",file[1],128,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"Must indicate a file name for small matrix B with the -f1 option.");
  ierr = PetscOptionsGetString(PETSC_NULL,"-f2",file[2],128,&flg);CHKERRQ(ierr);
  if (!flg) {
    preload = PETSC_FALSE;
  } else {
    ierr = PetscOptionsGetString(PETSC_NULL,"-f3",file[3],128,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"Must indicate a file name for test matrix B with the -f3 option."); 
  }

  PetscPreLoadBegin(preload,"Load system");
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[2*PetscPreLoadIt],FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A_save);CHKERRQ(ierr);
  ierr = MatLoad(A_save,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[2*PetscPreLoadIt+1],FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&B);CHKERRQ(ierr);
  ierr = MatLoad(B,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  ierr = MatGetSize(B,&M,&N);CHKERRQ(ierr);
  nzp  = (PetscInt)(0.1*M);
  ierr = PetscMalloc((nzp+1)*(sizeof(PetscInt)+sizeof(PetscScalar)),&idxn);CHKERRQ(ierr);
  a    = (PetscScalar*)(idxn + nzp);              
 
  /* Create vectors v1 and v2 that are compatible with A_save */
  ierr = VecCreate(PETSC_COMM_WORLD,&v1);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A_save,&m,PETSC_NULL);CHKERRQ(ierr);
  ierr = VecSetSizes(v1,m,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(v1);CHKERRQ(ierr);
  ierr = VecDuplicate(v1,&v2);CHKERRQ(ierr);

  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rdm);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rdm);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-fill",&fill,PETSC_NULL);CHKERRQ(ierr);

  /* Test MatMatMult() */
  /*-------------------*/
  if (Test_MatMatMult){
    ierr = MatDuplicate(A_save,MAT_COPY_VALUES,&A);CHKERRQ(ierr);
    ierr = MatMatMult(A,B,MAT_INITIAL_MATRIX,fill,&C);CHKERRQ(ierr);
    ierr = MatSetOptionsPrefix(C,"matmatmult_");CHKERRQ(ierr); /* enable option '-matmatmult_' for matrix C */
    ierr = MatGetInfo(C,MAT_GLOBAL_SUM,&info);CHKERRQ(ierr);
    /* ierr = PetscPrintf(PETSC_COMM_WORLD,"MatMatMult: nz_allocated = %g; nz_used = %g; nz_unneeded = %g\n",info.nz_allocated,info.nz_used, info.nz_unneeded); */

    /* Test MAT_REUSE_MATRIX - reuse symbolic C */
    alpha=1.0;
    for (i=0; i<2; i++){
      alpha -=0.1;
      ierr = MatScale(A,alpha);CHKERRQ(ierr);
      ierr = MatMatMult(A,B,MAT_REUSE_MATRIX,fill,&C);CHKERRQ(ierr);
    }

    /* Create vector x that is compatible with B */
    ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
    ierr = MatGetLocalSize(B,PETSC_NULL,&n);CHKERRQ(ierr);
    ierr = VecSetSizes(x,n,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = VecSetFromOptions(x);CHKERRQ(ierr);

    norm = 0.0;
    for (i=0; i<10; i++) {
      ierr = VecSetRandom(x,rdm);CHKERRQ(ierr);
      ierr = MatMult(B,x,v1);CHKERRQ(ierr);  
      ierr = MatMult(A,v1,v2);CHKERRQ(ierr);  /* v2 = A*B*x */
      ierr = MatMult(C,x,v1);CHKERRQ(ierr);   /* v1 = C*x   */
      ierr = VecNorm(v1,NORM_2,&norm_abs);CHKERRQ(ierr);
      ierr = VecAXPY(v1,none,v2);CHKERRQ(ierr);
      ierr = VecNorm(v1,NORM_2,&norm_tmp);CHKERRQ(ierr);
      norm_tmp /= norm_abs;
      if (norm_tmp > norm) norm = norm_tmp;
    }
    if (norm >= tol) {
      ierr = PetscPrintf(PETSC_COMM_SELF,"Error: MatMatMult(), |v1 - v2|: %G\n",norm);CHKERRQ(ierr);
    }

    ierr = VecDestroy(&x);CHKERRQ(ierr);
    ierr = MatDestroy(&A);CHKERRQ(ierr);

    /* Test MatDuplicate() of C */
    ierr = MatDuplicate(C,MAT_COPY_VALUES,&C1);CHKERRQ(ierr);
    ierr = MatDestroy(&C1);CHKERRQ(ierr);
    ierr = MatDestroy(&C);CHKERRQ(ierr); 
  } /* if (Test_MatMatMult) */

  /* Test MatTransposeMatMult() and MatMatTransposeMult() */
  /*------------------------------------------------------*/
  if (Test_MatMatTr){
    /* Create P */
    PetscInt PN,rstart,rend;
    PN   = M/2;
    nzp  = 5; /* num of nonzeros in each row of P */
    ierr = MatCreate(PETSC_COMM_WORLD,&P);CHKERRQ(ierr); 
    ierr = MatSetSizes(P,PETSC_DECIDE,PETSC_DECIDE,M,PN);CHKERRQ(ierr); 
    ierr = MatSetType(P,MATAIJ);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(P,nzp,PETSC_NULL);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(P,nzp,PETSC_NULL,nzp,PETSC_NULL);CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(P,&rstart,&rend);CHKERRQ(ierr);
    for (i=0; i<nzp; i++){
      ierr = PetscRandomGetValue(rdm,&a[i]);CHKERRQ(ierr);
    }
    for (i=rstart; i<rend; i++){
      for (j=0; j<nzp; j++){
        ierr = PetscRandomGetValue(rdm,&rval);CHKERRQ(ierr);
        idxn[j] = (PetscInt)(PetscRealPart(rval)*PN);
      }
      ierr = MatSetValues(P,1,&i,nzp,idxn,a,ADD_VALUES);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    /* Create R = P^T */
    ierr = MatTranspose(P,MAT_INITIAL_MATRIX,&R);CHKERRQ(ierr); 
    
    /* C = P^T*B */
    ierr = MatTransposeMatMult(P,B,MAT_INITIAL_MATRIX,fill,&C);CHKERRQ(ierr);
    ierr = MatGetInfo(C,MAT_GLOBAL_SUM,&info);CHKERRQ(ierr);
    
    /* Test MAT_REUSE_MATRIX - reuse symbolic C */
    ierr = MatTransposeMatMult(P,B,MAT_REUSE_MATRIX,fill,&C);CHKERRQ(ierr);
   
    /* Compare P^T*B and R*B */
    ierr = MatMatMult(R,B,MAT_INITIAL_MATRIX,fill,&C1);CHKERRQ(ierr);
    ierr = MatEqual(C,C1,&flg);CHKERRQ(ierr);
    if (!flg){
      /* Check norm of C1 = (-1.0)*C + C1 */
      PetscReal nrm;
      ierr = MatAXPY(C1,-1.0,C,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatNorm(C1,NORM_INFINITY,&nrm);CHKERRQ(ierr);
      if (nrm > 1.e-14){
        ierr = PetscPrintf(PETSC_COMM_WORLD,"Error in MatTransposeMatMult(): %g\n",nrm);
      }
    }
    ierr = MatDestroy(&C1);CHKERRQ(ierr);
    ierr = MatDestroy(&C);CHKERRQ(ierr);

    /* C = B*R^T */
    if (size == 1){
      ierr = MatMatTransposeMult(B,R,MAT_INITIAL_MATRIX,fill,&C);CHKERRQ(ierr);
      ierr = MatSetOptionsPrefix(C,"matmatmulttr_");CHKERRQ(ierr); /* enable '-matmatmulttr_' for matrix C */
      ierr = MatGetInfo(C,MAT_GLOBAL_SUM,&info);CHKERRQ(ierr);
      
      /* Test MAT_REUSE_MATRIX - reuse symbolic C */
      ierr = MatMatTransposeMult(B,R,MAT_REUSE_MATRIX,fill,&C);CHKERRQ(ierr);
    
      /* Check */
      ierr = MatMatMult(B,P,MAT_INITIAL_MATRIX,fill,&C1);CHKERRQ(ierr);
      ierr = MatEqual(C,C1,&flg);CHKERRQ(ierr);
      if (!flg){
        ierr = PetscPrintf(PETSC_COMM_WORLD,"Error in MatMatTransposeMult()\n");
      }
      ierr = MatDestroy(&C1);CHKERRQ(ierr);  
      ierr = MatDestroy(&C);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&P);CHKERRQ(ierr);
    ierr = MatDestroy(&R);CHKERRQ(ierr);
  }

  /* Test MatPtAP() */
  /*----------------------*/
  if (Test_MatPtAP){
    PetscInt PN;
    ierr = MatDuplicate(A_save,MAT_COPY_VALUES,&A);CHKERRQ(ierr);
    ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);
    ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
    /* ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] A: %d,%d, %d,%d\n",rank,m,n,M,N); */

    PN   = M/2; 
    nzp  = (PetscInt)(0.1*PN+1); /* num of nozeros in each row of P */
    ierr = MatCreate(PETSC_COMM_WORLD,&P);CHKERRQ(ierr); 
    ierr = MatSetSizes(P,PETSC_DECIDE,PETSC_DECIDE,N,PN);CHKERRQ(ierr); 
    ierr = MatSetType(P,MATAIJ);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(P,nzp,PETSC_NULL);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(P,nzp,PETSC_NULL,nzp,PETSC_NULL);CHKERRQ(ierr);
    for (i=0; i<nzp; i++){
      ierr = PetscRandomGetValue(rdm,&a[i]);CHKERRQ(ierr);
    }
    ierr = MatGetOwnershipRange(P,&rstart,&rend);CHKERRQ(ierr);
    for (i=rstart; i<rend; i++){
      for (j=0; j<nzp; j++){
        ierr = PetscRandomGetValue(rdm,&rval);CHKERRQ(ierr);
        idxn[j] = (PetscInt)(PetscRealPart(rval)*PN);
      }
      ierr = MatSetValues(P,1,&i,nzp,idxn,a,ADD_VALUES);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    /* ierr = MatView(P,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */
    ierr = MatGetSize(P,&pM,&pN);CHKERRQ(ierr);
    ierr = MatGetLocalSize(P,&pm,&pn);CHKERRQ(ierr);
    ierr = MatPtAP(A,P,MAT_INITIAL_MATRIX,fill,&C);CHKERRQ(ierr); 
    /* if (!rank){ierr = PetscPrintf(PETSC_COMM_SELF," MatPtAP() is done, P, %d, %d, %d,%d\n",pm,pn,pM,pN);} */

    /* Test MAT_REUSE_MATRIX - reuse symbolic C */
    alpha=1.0;
    for (i=0; i<2; i++){
      alpha -=0.1;
      ierr = MatScale(A,alpha);CHKERRQ(ierr);
      ierr = MatPtAP(A,P,MAT_REUSE_MATRIX,fill,&C);CHKERRQ(ierr);
    }

    /* Test MatDuplicate() */
    Mat Cdup;
    ierr = MatDuplicate(C,MAT_COPY_VALUES,&Cdup);CHKERRQ(ierr);
    ierr = MatDestroy(&Cdup);CHKERRQ(ierr);

    if (size>1) Test_MatRARt = PETSC_FALSE;
    /* Test MatRARt() */
    if (Test_MatRARt){
      Mat       R, RARt;
      PetscBool equal;
      ierr = MatTranspose(P,MAT_INITIAL_MATRIX,&R);CHKERRQ(ierr);
      ierr = MatRARt(A,R,MAT_INITIAL_MATRIX,2.0,&RARt);CHKERRQ(ierr);
      ierr = MatEqual(C,RARt,&equal);CHKERRQ(ierr);
      if (!equal) {
        PetscReal norm;
        ierr = MatAXPY(RARt,-1.0,C,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr); /* RARt = -RARt + C */
        ierr = MatNorm(RARt,NORM_FROBENIUS,&norm);CHKERRQ(ierr);
        if (norm > 1.e-12) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"|PtAP - RARt| = %g",norm);
      }
      ierr = MatDestroy(&R);CHKERRQ(ierr);
      ierr = MatDestroy(&RARt);CHKERRQ(ierr);
    }

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
      ierr = VecNorm(v4,NORM_2,&norm_abs);CHKERRQ(ierr);
      ierr = VecAXPY(v4,none,v3);CHKERRQ(ierr);
      ierr = VecNorm(v4,NORM_2,&norm_tmp);CHKERRQ(ierr);
      norm_tmp /= norm_abs;
      if (norm_tmp > norm) norm = norm_tmp;
    }
    if (norm >= tol) {
      ierr = PetscPrintf(PETSC_COMM_SELF,"Error: MatPtAP(), |v1 - v2|: %G\n",norm);CHKERRQ(ierr);
    }
  
    ierr = MatDestroy(&A);CHKERRQ(ierr);
    ierr = MatDestroy(&P);CHKERRQ(ierr);
    ierr = MatDestroy(&C);CHKERRQ(ierr);
    ierr = VecDestroy(&v3);CHKERRQ(ierr);
    ierr = VecDestroy(&v4);CHKERRQ(ierr);
    ierr = VecDestroy(&x);CHKERRQ(ierr);
  } 

  /* Destroy objects */
  ierr = VecDestroy(&v1);CHKERRQ(ierr);
  ierr = VecDestroy(&v2);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rdm);CHKERRQ(ierr);
  ierr = PetscFree(idxn);CHKERRQ(ierr);
  
  ierr = MatDestroy(&A_save);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);

  PetscPreLoadEnd();
  ierr = PetscFinalize();
  return 0;
}

