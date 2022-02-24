#include <../src/mat/impls/aij/mpi/mpiaij.h>

PetscErrorCode  MatGetMultiProcBlock_MPIAIJ(Mat mat, MPI_Comm subComm, MatReuse scall,Mat *subMat)
{
  Mat_MPIAIJ     *aij  = (Mat_MPIAIJ*)mat->data;
  Mat_SeqAIJ     *aijB = (Mat_SeqAIJ*)aij->B->data;
  PetscMPIInt    subCommSize,subCommRank;
  PetscMPIInt    *commRankMap,subRank,rank,commRank;
  PetscInt       *garrayCMap,col,i,j,*nnz,newRow,newCol;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_size(subComm,&subCommSize));
  CHKERRMPI(MPI_Comm_rank(subComm,&subCommRank));
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)mat),&commRank));

  /* create subMat object with the relevant layout */
  if (scall == MAT_INITIAL_MATRIX) {
    CHKERRQ(MatCreate(subComm,subMat));
    CHKERRQ(MatSetType(*subMat,MATMPIAIJ));
    CHKERRQ(MatSetSizes(*subMat,mat->rmap->n,mat->cmap->n,PETSC_DECIDE,PETSC_DECIDE));
    CHKERRQ(MatSetBlockSizesFromMats(*subMat,mat,mat));

    /* need to setup rmap and cmap before Preallocation */
    CHKERRQ(PetscLayoutSetUp((*subMat)->rmap));
    CHKERRQ(PetscLayoutSetUp((*subMat)->cmap));
  }

  /* create a map of comm_rank from subComm to comm - should commRankMap and garrayCMap be kept for reused? */
  CHKERRQ(PetscMalloc1(subCommSize,&commRankMap));
  CHKERRMPI(MPI_Allgather(&commRank,1,MPI_INT,commRankMap,1,MPI_INT,subComm));

  /* Traverse garray and identify column indices [of offdiag mat] that
   should be discarded. For the ones not discarded, store the newCol+1
   value in garrayCMap */
  CHKERRQ(PetscCalloc1(aij->B->cmap->n,&garrayCMap));
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

  if (scall == MAT_INITIAL_MATRIX) {
    /* Compute preallocation for the offdiag mat */
    CHKERRQ(PetscCalloc1(aij->B->rmap->n,&nnz));
    for (i=0; i<aij->B->rmap->n; i++) {
      for (j=aijB->i[i]; j<aijB->i[i+1]; j++) {
        if (garrayCMap[aijB->j[j]]) nnz[i]++;
      }
    }
    CHKERRQ(MatMPIAIJSetPreallocation(*(subMat),0,NULL,0,nnz));

    /* reuse diag block with the new submat */
    CHKERRQ(MatDestroy(&((Mat_MPIAIJ*)((*subMat)->data))->A));
    ((Mat_MPIAIJ*)((*subMat)->data))->A = aij->A;
    CHKERRQ(PetscObjectReference((PetscObject)aij->A));
  } else if (((Mat_MPIAIJ*)(*subMat)->data)->A != aij->A) {
    PetscObject obj = (PetscObject)((Mat_MPIAIJ*)((*subMat)->data))->A;
    CHKERRQ(PetscObjectReference((PetscObject)obj));
    ((Mat_MPIAIJ*)((*subMat)->data))->A = aij->A;
    CHKERRQ(PetscObjectReference((PetscObject)aij->A));
  }

  /* Traverse aij->B and insert values into subMat */
  if ((*subMat)->assembled) {
    (*subMat)->was_assembled = PETSC_TRUE;
    (*subMat)->assembled     = PETSC_FALSE;
  }
  for (i=0; i<aij->B->rmap->n; i++) {
    newRow = (*subMat)->rmap->range[subCommRank] + i;
    for (j=aijB->i[i]; j<aijB->i[i+1]; j++) {
      newCol = garrayCMap[aijB->j[j]];
      if (newCol) {
        newCol--; /* remove the increment */
        CHKERRQ(MatSetValues_MPIAIJ(*subMat,1,&newRow,1,&newCol,(aijB->a+j),INSERT_VALUES));
      }
    }
  }
  CHKERRQ(MatAssemblyBegin(*subMat,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(*subMat,MAT_FINAL_ASSEMBLY));

  /* deallocate temporary data */
  CHKERRQ(PetscFree(commRankMap));
  CHKERRQ(PetscFree(garrayCMap));
  if (scall == MAT_INITIAL_MATRIX) {
    CHKERRQ(PetscFree(nnz));
  }
  PetscFunctionReturn(0);
}
