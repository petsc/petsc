#include <../src/mat/impls/aij/mpi/mpiaij.h>

PetscErrorCode  MatGetMultiProcBlock_MPIAIJ(Mat mat, MPI_Comm subComm, MatReuse scall,Mat *subMat)
{
  Mat_MPIAIJ     *aij  = (Mat_MPIAIJ*)mat->data;
  Mat_SeqAIJ     *aijB = (Mat_SeqAIJ*)aij->B->data;
  PetscMPIInt    subCommSize,subCommRank;
  PetscMPIInt    *commRankMap,subRank,rank,commRank;
  PetscInt       *garrayCMap,col,i,j,*nnz,newRow,newCol;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(subComm,&subCommSize));
  PetscCallMPI(MPI_Comm_rank(subComm,&subCommRank));
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)mat),&commRank));

  /* create subMat object with the relevant layout */
  if (scall == MAT_INITIAL_MATRIX) {
    PetscCall(MatCreate(subComm,subMat));
    PetscCall(MatSetType(*subMat,MATMPIAIJ));
    PetscCall(MatSetSizes(*subMat,mat->rmap->n,mat->cmap->n,PETSC_DECIDE,PETSC_DECIDE));
    PetscCall(MatSetBlockSizesFromMats(*subMat,mat,mat));

    /* need to setup rmap and cmap before Preallocation */
    PetscCall(PetscLayoutSetUp((*subMat)->rmap));
    PetscCall(PetscLayoutSetUp((*subMat)->cmap));
  }

  /* create a map of comm_rank from subComm to comm - should commRankMap and garrayCMap be kept for reused? */
  PetscCall(PetscMalloc1(subCommSize,&commRankMap));
  PetscCallMPI(MPI_Allgather(&commRank,1,MPI_INT,commRankMap,1,MPI_INT,subComm));

  /* Traverse garray and identify column indices [of offdiag mat] that
   should be discarded. For the ones not discarded, store the newCol+1
   value in garrayCMap */
  PetscCall(PetscCalloc1(aij->B->cmap->n,&garrayCMap));
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
    PetscCall(PetscCalloc1(aij->B->rmap->n,&nnz));
    for (i=0; i<aij->B->rmap->n; i++) {
      for (j=aijB->i[i]; j<aijB->i[i+1]; j++) {
        if (garrayCMap[aijB->j[j]]) nnz[i]++;
      }
    }
    PetscCall(MatMPIAIJSetPreallocation(*(subMat),0,NULL,0,nnz));

    /* reuse diag block with the new submat */
    PetscCall(MatDestroy(&((Mat_MPIAIJ*)((*subMat)->data))->A));
    ((Mat_MPIAIJ*)((*subMat)->data))->A = aij->A;
    PetscCall(PetscObjectReference((PetscObject)aij->A));
  } else if (((Mat_MPIAIJ*)(*subMat)->data)->A != aij->A) {
    PetscObject obj = (PetscObject)((Mat_MPIAIJ*)((*subMat)->data))->A;
    PetscCall(PetscObjectReference((PetscObject)obj));
    ((Mat_MPIAIJ*)((*subMat)->data))->A = aij->A;
    PetscCall(PetscObjectReference((PetscObject)aij->A));
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
        PetscCall(MatSetValues_MPIAIJ(*subMat,1,&newRow,1,&newCol,(aijB->a+j),INSERT_VALUES));
      }
    }
  }
  PetscCall(MatAssemblyBegin(*subMat,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*subMat,MAT_FINAL_ASSEMBLY));

  /* deallocate temporary data */
  PetscCall(PetscFree(commRankMap));
  PetscCall(PetscFree(garrayCMap));
  if (scall == MAT_INITIAL_MATRIX) {
    PetscCall(PetscFree(nnz));
  }
  PetscFunctionReturn(0);
}
