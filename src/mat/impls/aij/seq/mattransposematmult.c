
/*
  Defines matrix-matrix product routines 
          C = A^T * B
*/

#include <../src/mat/impls/aij/seq/aij.h> /*I "petscmat.h" I*/

#undef __FUNCT__
#define __FUNCT__ "MatTransposeMatMult_SeqAIJ_SeqDense"
PetscErrorCode MatTransposeMatMult_SeqAIJ_SeqDense(Mat A,Mat B,MatReuse scall,PetscReal fill,Mat *C)
{
  PetscErrorCode      ierr;
  PetscInt            i,j,k,m,n,dof=B->cmap->N;
  Mat_MatTransMatMult *atb;
  Mat                 Cdense;
  PetscScalar         *Barray,*Carray,*btarray,*ctarray;
  Vec                 bt,ct;

  PetscFunctionBegin;
  ierr = PetscNew(Mat_MatTransMatMult,&atb);CHKERRQ(ierr);

  /* create MAIJ matrix mA from A */
  ierr = MatCreateMAIJ(A,dof,&atb->mA);CHKERRQ(ierr);
  
  /* create output dense matrix C = A^T*B */
  ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
  ierr = MatCreate(PetscObjectComm((PetscObject)A),&Cdense);CHKERRQ(ierr);
  ierr = MatSetSizes(Cdense,n,dof,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetType(Cdense,MATDENSE);CHKERRQ(ierr);
  ierr = MatSeqDenseSetPreallocation(Cdense,NULL);CHKERRQ(ierr);
  //ierr = MatMPIDenseSetPreallocation(Cdense,NULL);CHKERRQ(ierr);

  /* create vectors bt and ct to hold locally transposed arrays of B and C */
  ierr = VecCreate(PETSC_COMM_SELF,&bt);CHKERRQ(ierr);
  ierr = VecSetSizes(bt,m*dof,m*dof);CHKERRQ(ierr);
  ierr = VecSetFromOptions(bt);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_SELF,&ct);CHKERRQ(ierr);
  ierr = VecSetSizes(ct,n*dof,n*dof);CHKERRQ(ierr);
  ierr = VecSetFromOptions(ct);CHKERRQ(ierr);
  atb->bt = bt;
  atb->ct = ct;

  /* transpose local arry of B, then copy it to vector bt */
  ierr = MatDenseGetArray(B,&Barray);CHKERRQ(ierr);
  ierr = VecGetArray(bt,&btarray);CHKERRQ(ierr);

  k=0;
  for (j=0; j<dof; j++) {
    for (i=0; i<m; i++) btarray[i*dof + j] = Barray[k++]; 
  }
  ierr = VecRestoreArray(bt,&btarray);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(B,&Barray);CHKERRQ(ierr);
  
  /* compute ct = mA^T * cb */
  ierr = MatMultTranspose(atb->mA,bt,ct);CHKERRQ(ierr);

  /* transpose local arry of ct to matrix C */
  ierr = MatDenseGetArray(Cdense,&Carray);CHKERRQ(ierr);
  ierr = VecGetArray(ct,&ctarray);CHKERRQ(ierr);
  k = 0;
  for (j=0; j<dof; j++) {
    for (i=0; i<n; i++) Carray[k++] = ctarray[i*dof + j];
  }
  ierr = VecRestoreArray(ct,&ctarray);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(Cdense,&Carray);CHKERRQ(ierr);

  *C = Cdense;
  
  ierr = MatDestroy(&atb->mA);CHKERRQ(ierr);
  ierr = VecDestroy(&atb->bt);CHKERRQ(ierr);
  ierr = VecDestroy(&atb->ct);CHKERRQ(ierr);
  ierr = PetscFree(atb);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatTransposeMatMultSymbolic_SeqAIJ_SeqDense"
PetscErrorCode MatTransposeMatMultSymbolic_SeqAIJ_SeqDense(Mat A,Mat B,PetscReal fill,Mat *C)
{
  //PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatTransposeMatMultNumeric_SeqAIJ_SeqDense"
PetscErrorCode MatTransposeMatMultNumeric_SeqAIJ_SeqDense(Mat A,Mat B,Mat C)
{
    //PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
