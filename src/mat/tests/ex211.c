
static char help[] = "Tests MatCreateSubmatrix() in parallel.";

#include <petscmat.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>

PetscErrorCode ISGetSeqIS_SameColDist_Private(Mat mat,IS isrow,IS iscol,IS *isrow_d,IS *iscol_d,IS *iscol_o,const PetscInt *garray[])
{
  Vec            x,cmap;
  const PetscInt *is_idx;
  PetscScalar    *xarray,*cmaparray;
  PetscInt       ncols,isstart,*idx,m,rstart,count;
  Mat_MPIAIJ     *a=(Mat_MPIAIJ*)mat->data;
  Mat            B=a->B;
  Vec            lvec=a->lvec,lcmap;
  PetscInt       i,cstart,cend,Bn=B->cmap->N;
  MPI_Comm       comm;
  PetscMPIInt    rank;
  VecScatter     Mvctx;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)mat,&comm));
  PetscCallMPI(MPI_Comm_rank(comm,&rank));
  PetscCall(ISGetLocalSize(iscol,&ncols));

  /* (1) iscol is a sub-column vector of mat, pad it with '-1.' to form a full vector x */
  PetscCall(MatCreateVecs(mat,&x,NULL));
  PetscCall(VecDuplicate(x,&cmap));
  PetscCall(VecSet(x,-1.0));
  PetscCall(VecSet(cmap,-1.0));

  PetscCall(VecDuplicate(lvec,&lcmap));

  /* Get start indices */
  PetscCallMPI(MPI_Scan(&ncols,&isstart,1,MPIU_INT,MPI_SUM,comm));
  isstart -= ncols;
  PetscCall(MatGetOwnershipRangeColumn(mat,&cstart,&cend));

  PetscCall(ISGetIndices(iscol,&is_idx));
  PetscCall(VecGetArray(x,&xarray));
  PetscCall(VecGetArray(cmap,&cmaparray));
  PetscCall(PetscMalloc1(ncols,&idx));
  for (i=0; i<ncols; i++) {
    xarray[is_idx[i]-cstart]    = (PetscScalar)is_idx[i];
    cmaparray[is_idx[i]-cstart] = (PetscScalar)(i + isstart);      /* global index of iscol[i] */
    idx[i]                      = is_idx[i]-cstart; /* local index of iscol[i]  */
  }
  PetscCall(VecRestoreArray(x,&xarray));
  PetscCall(VecRestoreArray(cmap,&cmaparray));
  PetscCall(ISRestoreIndices(iscol,&is_idx));

  /* Get iscol_d */
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF,ncols,idx,PETSC_OWN_POINTER,iscol_d));
  PetscCall(ISGetBlockSize(iscol,&i));
  PetscCall(ISSetBlockSize(*iscol_d,i));

  /* Get isrow_d */
  PetscCall(ISGetLocalSize(isrow,&m));
  rstart = mat->rmap->rstart;
  PetscCall(PetscMalloc1(m,&idx));
  PetscCall(ISGetIndices(isrow,&is_idx));
  for (i=0; i<m; i++) idx[i] = is_idx[i]-rstart;
  PetscCall(ISRestoreIndices(isrow,&is_idx));

  PetscCall(ISCreateGeneral(PETSC_COMM_SELF,m,idx,PETSC_OWN_POINTER,isrow_d));
  PetscCall(ISGetBlockSize(isrow,&i));
  PetscCall(ISSetBlockSize(*isrow_d,i));

  /* (2) Scatter x and cmap using aij->Mvctx to get their off-process portions (see MatMult_MPIAIJ) */
#if 0
  if (!a->Mvctx_mpi1) {
    /* a->Mvctx causes random 'count' in o-build? See src/mat/tests/runex59_2 */
    a->Mvctx_mpi1_flg = PETSC_TRUE;
    PetscCall(MatSetUpMultiply_MPIAIJ(mat));
  }
  Mvctx = a->Mvctx_mpi1;
#endif
  Mvctx = a->Mvctx;
  PetscCall(VecScatterBegin(Mvctx,x,lvec,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(Mvctx,x,lvec,INSERT_VALUES,SCATTER_FORWARD));

  PetscCall(VecScatterBegin(Mvctx,cmap,lcmap,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(Mvctx,cmap,lcmap,INSERT_VALUES,SCATTER_FORWARD));

  /* (3) create sequential iscol_o (a subset of iscol) and isgarray */
  /* off-process column indices */
  count = 0;
  PetscInt *cmap1;
  PetscCall(PetscMalloc1(Bn,&idx));
  PetscCall(PetscMalloc1(Bn,&cmap1));

  PetscCall(VecGetArray(lvec,&xarray));
  PetscCall(VecGetArray(lcmap,&cmaparray));
  for (i=0; i<Bn; i++) {
    if (PetscRealPart(xarray[i]) > -1.0) {
      idx[count]   = i;                   /* local column index in off-diagonal part B */
      cmap1[count] = (PetscInt)(PetscRealPart(cmaparray[i]));  /* column index in submat */
      count++;
    }
  }
  printf("[%d] Bn %d, count %d\n",rank,Bn,count);
  PetscCall(VecRestoreArray(lvec,&xarray));
  PetscCall(VecRestoreArray(lcmap,&cmaparray));
  if (count != 6) {
    printf("[%d] count %d != 6 lvec:\n",rank,count);
    PetscCall(VecView(lvec,0));

    printf("[%d] count %d != 6 lcmap:\n",rank,count);
    PetscCall(VecView(lcmap,0));
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"count %d != 6",count);
  }

  PetscCall(ISCreateGeneral(PETSC_COMM_SELF,count,idx,PETSC_COPY_VALUES,iscol_o));
  /* cannot ensure iscol_o has same blocksize as iscol! */

  PetscCall(PetscFree(idx));

  *garray = cmap1;

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&cmap));
  PetscCall(VecDestroy(&lcmap));
  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  Mat            C,A;
  PetscInt       i,j,m = 3,n = 2,rstart,rend;
  PetscMPIInt    size,rank;
  PetscScalar    v;
  IS             isrow,iscol;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  n    = 2*size;

  PetscCall(MatCreate(PETSC_COMM_WORLD,&C));
  PetscCall(MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n));
  PetscCall(MatSetFromOptions(C));
  PetscCall(MatSetUp(C));

  /*
        This is JUST to generate a nice test matrix, all processors fill up
    the entire matrix. This is not something one would ever do in practice.
  */
  PetscCall(MatGetOwnershipRange(C,&rstart,&rend));
  for (i=rstart; i<rend; i++) {
    for (j=0; j<m*n; j++) {
      v    = i + j + 1;
      PetscCall(MatSetValues(C,1,&i,1,&j,&v,INSERT_VALUES));
    }
  }

  PetscCall(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));

  /*
     Generate a new matrix consisting of every second row and column of
   the original matrix
  */
  PetscCall(MatGetOwnershipRange(C,&rstart,&rend));
  /* Create parallel IS with the rows we want on THIS processor */
  PetscCall(ISCreateStride(PETSC_COMM_WORLD,(rend-rstart)/2,rstart,2,&isrow));
  /* Create parallel IS with the rows we want on THIS processor (same as rows for now) */
  PetscCall(ISCreateStride(PETSC_COMM_WORLD,(rend-rstart)/2,rstart,2,&iscol));

  IS             iscol_d,isrow_d,iscol_o;
  const PetscInt *garray;
  PetscCall(ISGetSeqIS_SameColDist_Private(C,isrow,iscol,&isrow_d,&iscol_d,&iscol_o,&garray));

  PetscCall(ISDestroy(&isrow_d));
  PetscCall(ISDestroy(&iscol_d));
  PetscCall(ISDestroy(&iscol_o));
  PetscCall(PetscFree(garray));

  PetscCall(MatCreateSubMatrix(C,isrow,iscol,MAT_INITIAL_MATRIX,&A));
  PetscCall(MatCreateSubMatrix(C,isrow,iscol,MAT_REUSE_MATRIX,&A));

  PetscCall(ISDestroy(&isrow));
  PetscCall(ISDestroy(&iscol));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&C));
  PetscCall(PetscFinalize());
  return 0;
}
