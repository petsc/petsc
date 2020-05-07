#include <petsc/private/matimpl.h>        /*I "petscmat.h" I*/

PetscErrorCode MatConvert_Shell(Mat oldmat, MatType newtype,MatReuse reuse,Mat *newmat)
{
  Mat            mat;
  Vec            in,out;
  MPI_Comm       comm;
  PetscScalar    *array;
  PetscInt       *dnnz,*onnz,*dnnzu,*onnzu;
  PetscInt       cst,Nbs,mbs,nbs,rbs,cbs;
  PetscInt       im,i,m,n,M,N,*rows,start;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)oldmat,&comm);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(oldmat,&start,NULL);CHKERRQ(ierr);
  ierr = MatGetOwnershipRangeColumn(oldmat,&cst,NULL);CHKERRQ(ierr);
  ierr = MatCreateVecs(oldmat,&in,&out);CHKERRQ(ierr);
  ierr = MatGetLocalSize(oldmat,&m,&n);CHKERRQ(ierr);
  ierr = MatGetSize(oldmat,&M,&N);CHKERRQ(ierr);
  ierr = PetscMalloc1(m,&rows);CHKERRQ(ierr);

  ierr = MatCreate(comm,&mat);CHKERRQ(ierr);
  ierr = MatSetSizes(mat,m,n,M,N);CHKERRQ(ierr);
  ierr = MatSetType(mat,newtype);CHKERRQ(ierr);
  ierr = MatSetBlockSizesFromMats(mat,oldmat,oldmat);CHKERRQ(ierr);
  ierr = MatGetBlockSizes(mat,&rbs,&cbs);CHKERRQ(ierr);
  mbs  = m/rbs;
  nbs  = n/cbs;
  Nbs  = N/cbs;
  cst  = cst/cbs;
  ierr = PetscMalloc4(mbs,&dnnz,mbs,&onnz,mbs,&dnnzu,mbs,&onnzu);CHKERRQ(ierr);
  for (i=0; i<mbs; i++) {
    dnnz[i]  = nbs;
    onnz[i]  = Nbs - nbs;
    dnnzu[i] = PetscMax(nbs - i,0);
    onnzu[i] = PetscMax(Nbs - (cst + nbs),0);
  }
  ierr = MatXAIJSetPreallocation(mat,PETSC_DECIDE,dnnz,onnz,dnnzu,onnzu);CHKERRQ(ierr);
  ierr = PetscFree4(dnnz,onnz,dnnzu,onnzu);CHKERRQ(ierr);
  ierr = VecSetOption(in,VEC_IGNORE_OFF_PROC_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatSetUp(mat);CHKERRQ(ierr);
  for (i=0; i<N; i++) {
    PetscInt j;

    ierr = VecZeroEntries(in);CHKERRQ(ierr);
    ierr = VecSetValue(in,i,1.,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(in);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(in);CHKERRQ(ierr);
    ierr = MatMult(oldmat,in,out);CHKERRQ(ierr);
    ierr = VecGetArray(out,&array);CHKERRQ(ierr);
    for (j=0, im = 0; j<m; j++) {
      if (PetscAbsScalar(array[j]) == 0.0) continue;
      rows[im]  = j+start;
      array[im] = array[j];
      im++;
    }
    ierr = MatSetValues(mat,im,rows,1,&i,array,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecRestoreArray(out,&array);CHKERRQ(ierr);
  }
  ierr = PetscFree(rows);CHKERRQ(ierr);
  ierr = VecDestroy(&in);CHKERRQ(ierr);
  ierr = VecDestroy(&out);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (reuse == MAT_INPLACE_MATRIX) {
    ierr = MatHeaderReplace(oldmat,&mat);CHKERRQ(ierr);
  } else {
    *newmat = mat;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetDiagonal_CF(Mat A,Vec X)
{
  Mat            B;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,&B);CHKERRQ(ierr);
  if (!B) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing user matrix");
  ierr = MatGetDiagonal(B,X);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_CF(Mat A,Vec X,Vec Y)
{
  Mat            B;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,&B);CHKERRQ(ierr);
  if (!B) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing user matrix");
  ierr = MatMult(B,X,Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTranspose_CF(Mat A,Vec X,Vec Y)
{
  Mat            B;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,&B);CHKERRQ(ierr);
  if (!B) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing user matrix");
  ierr = MatMultTranspose(B,X,Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_CF(Mat A)
{
  Mat            B;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,&B);CHKERRQ(ierr);
  if (!B) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing user matrix");
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatConvertFrom_Shell(Mat A, MatType newtype,MatReuse reuse,Mat *B)
{
  Mat            M;
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStrcmp(newtype,MATSHELL,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only conversion to MATSHELL");
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
    ierr = MatCreateShell(PetscObjectComm((PetscObject)A),A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N,A,&M);CHKERRQ(ierr);
    ierr = MatSetBlockSizesFromMats(M,A,A);CHKERRQ(ierr);
    ierr = MatShellSetOperation(M,MATOP_MULT,          (void (*)(void))MatMult_CF);CHKERRQ(ierr);
    ierr = MatShellSetOperation(M,MATOP_MULT_TRANSPOSE,(void (*)(void))MatMultTranspose_CF);CHKERRQ(ierr);
    ierr = MatShellSetOperation(M,MATOP_GET_DIAGONAL,  (void (*)(void))MatGetDiagonal_CF);CHKERRQ(ierr);
    ierr = MatShellSetOperation(M,MATOP_DESTROY,       (void (*)(void))MatDestroy_CF);CHKERRQ(ierr);
    ierr = PetscFree(M->defaultvectype);CHKERRQ(ierr);
    ierr = PetscStrallocpy(A->defaultvectype,&M->defaultvectype);CHKERRQ(ierr);
    *B = M;
  } else SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Not implemented");
  PetscFunctionReturn(0);
}
