
#include <../src/mat/impls/aij/seq/aij.h>
#include <petsc/private/isimpl.h>
#include <petsc/private/vecimpl.h>
#include <petsclayouthdf5.h>

#if defined(PETSC_HAVE_HDF5)
PetscErrorCode MatLoad_AIJ_HDF5(Mat mat, PetscViewer viewer)
{
  PetscViewerFormat format;
  const PetscInt  *i_glob = NULL;
  PetscInt        *i = NULL;
  const PetscInt  *j = NULL;
  const PetscScalar *a = NULL;
  char            *a_name = NULL, *i_name = NULL, *j_name = NULL, *c_name = NULL;
  const char      *mat_name = NULL;
  PetscInt        p, m, M, N;
  PetscInt        bs = mat->rmap->bs;
  PetscInt        *range;
  PetscBool       flg;
  IS              is_i = NULL, is_j = NULL;
  Vec             vec_a = NULL;
  PetscLayout     jmap = NULL;
  MPI_Comm        comm;
  PetscMPIInt     rank, size;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscCall(PetscViewerGetFormat(viewer, &format));
  switch (format) {
    case PETSC_VIEWER_HDF5_PETSC:
    case PETSC_VIEWER_DEFAULT:
    case PETSC_VIEWER_NATIVE:
    case PETSC_VIEWER_HDF5_MAT:
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"PetscViewerFormat %s not supported for HDF5 input.",PetscViewerFormats[format]);
  }

  PetscCall(PetscObjectGetComm((PetscObject)mat,&comm));
  PetscCallMPI(MPI_Comm_rank(comm,&rank));
  PetscCallMPI(MPI_Comm_size(comm,&size));
  PetscCall(PetscObjectGetName((PetscObject)mat,&mat_name));
  if (format==PETSC_VIEWER_HDF5_MAT) {
    PetscCall(PetscStrallocpy("jc",&i_name));
    PetscCall(PetscStrallocpy("ir",&j_name));
    PetscCall(PetscStrallocpy("data",&a_name));
    PetscCall(PetscStrallocpy("MATLAB_sparse",&c_name));
  } else {
    /* TODO Once corresponding MatView is implemented, change the names to i,j,a */
    /* TODO Maybe there could be both namings in the file, using "symbolic link" features of HDF5. */
    PetscCall(PetscStrallocpy("jc",&i_name));
    PetscCall(PetscStrallocpy("ir",&j_name));
    PetscCall(PetscStrallocpy("data",&a_name));
    PetscCall(PetscStrallocpy("MATLAB_sparse",&c_name));
  }

  ierr = PetscOptionsBegin(comm,NULL,"Options for loading matrix from HDF5","Mat");PetscCall(ierr);
  PetscCall(PetscOptionsInt("-matload_block_size","Set the blocksize used to store the matrix","MatLoad",bs,&bs,&flg));
  ierr = PetscOptionsEnd();PetscCall(ierr);
  if (flg) {
    PetscCall(MatSetBlockSize(mat, bs));
  }

  PetscCall(PetscViewerHDF5PushGroup(viewer,mat_name));
  PetscCall(PetscViewerHDF5ReadAttribute(viewer,NULL,c_name,PETSC_INT,NULL,&N));
  PetscCall(PetscViewerHDF5ReadSizes(viewer, i_name, NULL, &M));
  --M;  /* i has size M+1 as there is global number of nonzeros stored at the end */

  if (format==PETSC_VIEWER_HDF5_MAT && !mat->symmetric) {
    /* Swap row and columns layout for unallocated matrix. I want to avoid calling MatTranspose() just to transpose sparsity pattern and layout. */
    if (!mat->preallocated) {
      PetscLayout tmp;
      tmp = mat->rmap; mat->rmap = mat->cmap; mat->cmap = tmp;
    } else SETERRQ(comm,PETSC_ERR_SUP,"Not for preallocated matrix - we would need to transpose it here which we want to avoid");
  }

  /* If global sizes are set, check if they are consistent with that given in the file */
  PetscCheckFalse(mat->rmap->N >= 0 && mat->rmap->N != M,PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Inconsistent # of rows: Matrix in file has (%" PetscInt_FMT ") and input matrix has (%" PetscInt_FMT ")",mat->rmap->N,M);
  PetscCheckFalse(mat->cmap->N >= 0 && mat->cmap->N != N,PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Inconsistent # of cols: Matrix in file has (%" PetscInt_FMT ") and input matrix has (%" PetscInt_FMT ")",mat->cmap->N,N);

  /* Determine ownership of all (block) rows and columns */
  mat->rmap->N = M;
  mat->cmap->N = N;
  PetscCall(PetscLayoutSetUp(mat->rmap));
  PetscCall(PetscLayoutSetUp(mat->cmap));
  m = mat->rmap->n;

  /* Read array i (array of row indices) */
  PetscCall(PetscMalloc1(m+1, &i)); /* allocate i with one more position for local number of nonzeros on each rank */
  i[0] = i[m] = 0; /* make the last entry always defined - the code block below overwrites it just on last rank */
  if (rank == size-1) m++; /* in the loaded array i_glob, only the last rank has one more position with the global number of nonzeros */
  M++;
  PetscCall(ISCreate(comm,&is_i));
  PetscCall(PetscObjectSetName((PetscObject)is_i,i_name));
  PetscCall(PetscLayoutSetLocalSize(is_i->map,m));
  PetscCall(PetscLayoutSetSize(is_i->map,M));
  PetscCall(ISLoad(is_i,viewer));
  PetscCall(ISGetIndices(is_i,&i_glob));
  PetscCall(PetscArraycpy(i,i_glob,m));

  /* Reset m and M to the matrix sizes */
  m = mat->rmap->n;
  M--;

  /* Create PetscLayout for j and a vectors; construct ranges first */
  PetscCall(PetscMalloc1(size+1, &range));
  PetscCallMPI(MPI_Allgather(i, 1, MPIU_INT, range, 1, MPIU_INT, comm));
  /* Last rank has global number of nonzeros (= length of j and a arrays) in i[m] (last i entry) so broadcast it */
  range[size] = i[m];
  PetscCallMPI(MPI_Bcast(&range[size], 1, MPIU_INT, size-1, comm));
  for (p=size-1; p>0; p--) {
    if (!range[p]) range[p] = range[p+1]; /* for ranks with 0 rows, take the value from the next processor */
  }
  i[m] = range[rank+1]; /* i[m] (last i entry) is equal to next rank's offset */
  /* Deduce rstart, rend, n and N from the ranges */
  PetscCall(PetscLayoutCreateFromRanges(comm,range,PETSC_OWN_POINTER,1,&jmap));

  /* Convert global to local indexing of rows */
  for (p=1; p<m+1; ++p) i[p] -= i[0];
  i[0] = 0;

  /* Read array j (array of column indices) */
  PetscCall(ISCreate(comm,&is_j));
  PetscCall(PetscObjectSetName((PetscObject)is_j,j_name));
  PetscCall(PetscLayoutDuplicate(jmap,&is_j->map));
  PetscCall(ISLoad(is_j,viewer));
  PetscCall(ISGetIndices(is_j,&j));

  /* Read array a (array of values) */
  PetscCall(VecCreate(comm,&vec_a));
  PetscCall(PetscObjectSetName((PetscObject)vec_a,a_name));
  PetscCall(PetscLayoutDuplicate(jmap,&vec_a->map));
  PetscCall(VecLoad(vec_a,viewer));
  PetscCall(VecGetArrayRead(vec_a,&a));

  /* populate matrix */
  if (!((PetscObject)mat)->type_name) {
    PetscCall(MatSetType(mat,MATAIJ));
  }
  PetscCall(MatSeqAIJSetPreallocationCSR(mat,i,j,a));
  PetscCall(MatMPIAIJSetPreallocationCSR(mat,i,j,a));
  /*
  PetscCall(MatSeqBAIJSetPreallocationCSR(mat,bs,i,j,a));
  PetscCall(MatMPIBAIJSetPreallocationCSR(mat,bs,i,j,a));
  */

  if (format==PETSC_VIEWER_HDF5_MAT && !mat->symmetric) {
    /* Transpose the input matrix back */
    PetscCall(MatTranspose(mat,MAT_INPLACE_MATRIX,&mat));
  }

  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscCall(PetscFree(i_name));
  PetscCall(PetscFree(j_name));
  PetscCall(PetscFree(a_name));
  PetscCall(PetscFree(c_name));
  PetscCall(PetscLayoutDestroy(&jmap));
  PetscCall(PetscFree(i));
  PetscCall(ISRestoreIndices(is_i,&i_glob));
  PetscCall(ISRestoreIndices(is_j,&j));
  PetscCall(VecRestoreArrayRead(vec_a,&a));
  PetscCall(ISDestroy(&is_i));
  PetscCall(ISDestroy(&is_j));
  PetscCall(VecDestroy(&vec_a));
  PetscFunctionReturn(0);
}
#endif
