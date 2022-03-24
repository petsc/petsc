
static char help[] = "Tests MatCreateSubmatrix() in parallel.";

#include <petscmat.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>

PetscErrorCode ISGetSeqIS_SameColDist_Private(Mat mat,IS isrow,IS iscol,IS *isrow_d,IS *iscol_d,IS *iscol_o,const PetscInt *garray[])
{
  PetscErrorCode ierr;
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
  CHKERRQ(PetscObjectGetComm((PetscObject)mat,&comm));
  CHKERRMPI(MPI_Comm_rank(comm,&rank));
  CHKERRQ(ISGetLocalSize(iscol,&ncols));

  /* (1) iscol is a sub-column vector of mat, pad it with '-1.' to form a full vector x */
  CHKERRQ(MatCreateVecs(mat,&x,NULL));
  CHKERRQ(VecDuplicate(x,&cmap));
  CHKERRQ(VecSet(x,-1.0));
  CHKERRQ(VecSet(cmap,-1.0));

  CHKERRQ(VecDuplicate(lvec,&lcmap));

  /* Get start indices */
  CHKERRMPI(MPI_Scan(&ncols,&isstart,1,MPIU_INT,MPI_SUM,comm));
  isstart -= ncols;
  CHKERRQ(MatGetOwnershipRangeColumn(mat,&cstart,&cend));

  CHKERRQ(ISGetIndices(iscol,&is_idx));
  CHKERRQ(VecGetArray(x,&xarray));
  CHKERRQ(VecGetArray(cmap,&cmaparray));
  CHKERRQ(PetscMalloc1(ncols,&idx));
  for (i=0; i<ncols; i++) {
    xarray[is_idx[i]-cstart]    = (PetscScalar)is_idx[i];
    cmaparray[is_idx[i]-cstart] = (PetscScalar)(i + isstart);      /* global index of iscol[i] */
    idx[i]                      = is_idx[i]-cstart; /* local index of iscol[i]  */
  }
  CHKERRQ(VecRestoreArray(x,&xarray));
  CHKERRQ(VecRestoreArray(cmap,&cmaparray));
  CHKERRQ(ISRestoreIndices(iscol,&is_idx));

  /* Get iscol_d */
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,ncols,idx,PETSC_OWN_POINTER,iscol_d));
  CHKERRQ(ISGetBlockSize(iscol,&i));
  CHKERRQ(ISSetBlockSize(*iscol_d,i));

  /* Get isrow_d */
  CHKERRQ(ISGetLocalSize(isrow,&m));
  rstart = mat->rmap->rstart;
  CHKERRQ(PetscMalloc1(m,&idx));
  CHKERRQ(ISGetIndices(isrow,&is_idx));
  for (i=0; i<m; i++) idx[i] = is_idx[i]-rstart;
  CHKERRQ(ISRestoreIndices(isrow,&is_idx));

  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,m,idx,PETSC_OWN_POINTER,isrow_d));
  CHKERRQ(ISGetBlockSize(isrow,&i));
  CHKERRQ(ISSetBlockSize(*isrow_d,i));

  /* (2) Scatter x and cmap using aij->Mvctx to get their off-process portions (see MatMult_MPIAIJ) */
#if 0
  if (!a->Mvctx_mpi1) {
    /* a->Mvctx causes random 'count' in o-build? See src/mat/tests/runex59_2 */
    a->Mvctx_mpi1_flg = PETSC_TRUE;
    CHKERRQ(MatSetUpMultiply_MPIAIJ(mat));
  }
  Mvctx = a->Mvctx_mpi1;
#endif
  Mvctx = a->Mvctx;
  CHKERRQ(VecScatterBegin(Mvctx,x,lvec,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(Mvctx,x,lvec,INSERT_VALUES,SCATTER_FORWARD));

  CHKERRQ(VecScatterBegin(Mvctx,cmap,lcmap,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(Mvctx,cmap,lcmap,INSERT_VALUES,SCATTER_FORWARD));

  /* (3) create sequential iscol_o (a subset of iscol) and isgarray */
  /* off-process column indices */
  count = 0;
  PetscInt *cmap1;
  CHKERRQ(PetscMalloc1(Bn,&idx));
  CHKERRQ(PetscMalloc1(Bn,&cmap1));

  CHKERRQ(VecGetArray(lvec,&xarray));
  CHKERRQ(VecGetArray(lcmap,&cmaparray));
  for (i=0; i<Bn; i++) {
    if (PetscRealPart(xarray[i]) > -1.0) {
      idx[count]   = i;                   /* local column index in off-diagonal part B */
      cmap1[count] = (PetscInt)(PetscRealPart(cmaparray[i]));  /* column index in submat */
      count++;
    }
  }
  printf("[%d] Bn %d, count %d\n",rank,Bn,count);
  CHKERRQ(VecRestoreArray(lvec,&xarray));
  CHKERRQ(VecRestoreArray(lcmap,&cmaparray));
  if (count != 6) {
    printf("[%d] count %d != 6 lvec:\n",rank,count);
    CHKERRQ(VecView(lvec,0));

    printf("[%d] count %d != 6 lcmap:\n",rank,count);
    CHKERRQ(VecView(lcmap,0));
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"count %d != 6",count);
  }

  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,count,idx,PETSC_COPY_VALUES,iscol_o));
  /* cannot ensure iscol_o has same blocksize as iscol! */

  CHKERRQ(PetscFree(idx));

  *garray = cmap1;

  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&cmap));
  CHKERRQ(VecDestroy(&lcmap));
  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  Mat            C,A;
  PetscInt       i,j,m = 3,n = 2,rstart,rend;
  PetscMPIInt    size,rank;
  PetscErrorCode ierr;
  PetscScalar    v;
  IS             isrow,iscol;

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  n    = 2*size;

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&C));
  CHKERRQ(MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n));
  CHKERRQ(MatSetFromOptions(C));
  CHKERRQ(MatSetUp(C));

  /*
        This is JUST to generate a nice test matrix, all processors fill up
    the entire matrix. This is not something one would ever do in practice.
  */
  CHKERRQ(MatGetOwnershipRange(C,&rstart,&rend));
  for (i=rstart; i<rend; i++) {
    for (j=0; j<m*n; j++) {
      v    = i + j + 1;
      CHKERRQ(MatSetValues(C,1,&i,1,&j,&v,INSERT_VALUES));
    }
  }

  CHKERRQ(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));

  /*
     Generate a new matrix consisting of every second row and column of
   the original matrix
  */
  CHKERRQ(MatGetOwnershipRange(C,&rstart,&rend));
  /* Create parallel IS with the rows we want on THIS processor */
  CHKERRQ(ISCreateStride(PETSC_COMM_WORLD,(rend-rstart)/2,rstart,2,&isrow));
  /* Create parallel IS with the rows we want on THIS processor (same as rows for now) */
  CHKERRQ(ISCreateStride(PETSC_COMM_WORLD,(rend-rstart)/2,rstart,2,&iscol));

  IS             iscol_d,isrow_d,iscol_o;
  const PetscInt *garray;
  CHKERRQ(ISGetSeqIS_SameColDist_Private(C,isrow,iscol,&isrow_d,&iscol_d,&iscol_o,&garray));

  CHKERRQ(ISDestroy(&isrow_d));
  CHKERRQ(ISDestroy(&iscol_d));
  CHKERRQ(ISDestroy(&iscol_o));
  CHKERRQ(PetscFree(garray));

  CHKERRQ(MatCreateSubMatrix(C,isrow,iscol,MAT_INITIAL_MATRIX,&A));
  CHKERRQ(MatCreateSubMatrix(C,isrow,iscol,MAT_REUSE_MATRIX,&A));

  CHKERRQ(ISDestroy(&isrow));
  CHKERRQ(ISDestroy(&iscol));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&C));
  CHKERRQ(PetscFinalize());
  return 0;
}
