#include <../src/mat/impls/aij/mpi/mpiaij.h>

#undef __FUNCT__
#define __FUNCT__ "MatGetMultiProcBlock_MPIAIJ"
PetscErrorCode  MatGetMultiProcBlock_MPIAIJ(Mat mat, MPI_Comm subComm, MatReuse scall,Mat* subMat)
{
  PetscErrorCode ierr;
  Mat_MPIAIJ     *aij = (Mat_MPIAIJ*)mat->data;
  Mat_SeqAIJ*    aijB = (Mat_SeqAIJ*)aij->B->data;
  PetscMPIInt    commRank,subCommSize,subCommRank;
  PetscMPIInt    *commRankMap,subRank,rank,commsize;
  PetscInt       *garrayCMap,col,i,j,*nnz,newRow,newCol;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(((PetscObject)mat)->comm,&commsize);CHKERRQ(ierr);
  ierr = MPI_Comm_size(subComm,&subCommSize);CHKERRQ(ierr);

  /* create subMat object with the relavent layout */
  if (scall == MAT_INITIAL_MATRIX){
    ierr = MatCreate(subComm,subMat);CHKERRQ(ierr);
    ierr = MatSetType(*subMat,MATMPIAIJ);CHKERRQ(ierr);
    ierr = MatSetSizes(*subMat,mat->rmap->n,mat->cmap->n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = MatSetBlockSizes(*subMat,mat->rmap->bs,mat->cmap->bs);CHKERRQ(ierr);

    /* need to setup rmap and cmap before Preallocation */
    ierr = PetscLayoutSetBlockSize((*subMat)->rmap,mat->rmap->bs);CHKERRQ(ierr);
    ierr = PetscLayoutSetBlockSize((*subMat)->cmap,mat->cmap->bs);CHKERRQ(ierr);
    ierr = PetscLayoutSetUp((*subMat)->rmap);CHKERRQ(ierr);
    ierr = PetscLayoutSetUp((*subMat)->cmap);CHKERRQ(ierr);
  }

  /* create a map of comm_rank from subComm to comm - should commRankMap and garrayCMap be kept for reused? */
  ierr = MPI_Comm_rank(((PetscObject)mat)->comm,&commRank);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(subComm,&subCommRank);CHKERRQ(ierr);
  ierr = PetscMalloc(subCommSize*sizeof(PetscMPIInt),&commRankMap);CHKERRQ(ierr);
  ierr = MPI_Allgather(&commRank,1,MPI_INT,commRankMap,1,MPI_INT,subComm);CHKERRQ(ierr);

  /* Traverse garray and identify column indices [of offdiag mat] that
   should be discarded. For the ones not discarded, store the newCol+1
   value in garrayCMap */
  ierr = PetscMalloc(aij->B->cmap->n*sizeof(PetscInt),&garrayCMap);CHKERRQ(ierr);
  ierr = PetscMemzero(garrayCMap,aij->B->cmap->n*sizeof(PetscInt));CHKERRQ(ierr);
  for (i=0; i<aij->B->cmap->n; i++) {
    col = aij->garray[i];
    for (subRank=0; subRank<subCommSize; subRank++) {
      rank = commRankMap[subRank];
      if ((col >= mat->cmap->range[rank]) && (col < mat->cmap->range[rank+1])) {
        garrayCMap[i] = (*subMat)->cmap->range[subRank] + col - mat->cmap->range[rank]+1;
        break;
      }
    }
  }

  if (scall == MAT_INITIAL_MATRIX){
    /* Now compute preallocation for the offdiag mat */
    ierr = PetscMalloc(aij->B->rmap->n*sizeof(PetscInt),&nnz);CHKERRQ(ierr);
    ierr = PetscMemzero(nnz,aij->B->rmap->n*sizeof(PetscInt));CHKERRQ(ierr);
    for (i=0; i<aij->B->rmap->n; i++) {
      for (j=aijB->i[i]; j<aijB->i[i+1]; j++) {
        if (garrayCMap[aijB->j[j]]) nnz[i]++;
      }
    }
    ierr = MatMPIAIJSetPreallocation(*(subMat),0,PETSC_NULL,0,nnz);CHKERRQ(ierr);

    /* reuse diag block with the new submat */
    ierr = MatDestroy(&((Mat_MPIAIJ*)((*subMat)->data))->A);CHKERRQ(ierr);
    ((Mat_MPIAIJ*)((*subMat)->data))->A = aij->A;
    ierr = PetscObjectReference((PetscObject)aij->A);CHKERRQ(ierr);
  } else if ( ((Mat_MPIAIJ*)(*subMat)->data)->A != aij->A  ){
    PetscObject obj = (PetscObject)((Mat_MPIAIJ*)((*subMat)->data))->A;
    ierr = PetscObjectReference((PetscObject)obj);CHKERRQ(ierr);
    ((Mat_MPIAIJ*)((*subMat)->data))->A = aij->A;
    ierr = PetscObjectReference((PetscObject)aij->A);CHKERRQ(ierr);
  }

  /* Now traverse aij->B and insert values into subMat */
  for (i=0; i<aij->B->rmap->n; i++) {
    newRow = (*subMat)->rmap->range[subCommRank] + i;
    for (j=aijB->i[i]; j<aijB->i[i+1]; j++) {
      newCol = garrayCMap[aijB->j[j]];
      if (newCol) {
        newCol--; /* remove the increment */
        ierr = MatSetValues(*subMat,1,&newRow,1,&newCol,(aijB->a+j),INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }

  /* assemble the submat */
  ierr = MatAssemblyBegin(*subMat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*subMat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* deallocate temporary data */
  ierr = PetscFree(commRankMap);CHKERRQ(ierr);
  ierr = PetscFree(garrayCMap);CHKERRQ(ierr);
  if (scall == MAT_INITIAL_MATRIX){
    ierr = PetscFree(nnz);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
