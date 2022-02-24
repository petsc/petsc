
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
  CHKERRQ(PetscViewerGetFormat(viewer, &format));
  switch (format) {
    case PETSC_VIEWER_HDF5_PETSC:
    case PETSC_VIEWER_DEFAULT:
    case PETSC_VIEWER_NATIVE:
    case PETSC_VIEWER_HDF5_MAT:
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"PetscViewerFormat %s not supported for HDF5 input.",PetscViewerFormats[format]);
  }

  CHKERRQ(PetscObjectGetComm((PetscObject)mat,&comm));
  CHKERRMPI(MPI_Comm_rank(comm,&rank));
  CHKERRMPI(MPI_Comm_size(comm,&size));
  CHKERRQ(PetscObjectGetName((PetscObject)mat,&mat_name));
  if (format==PETSC_VIEWER_HDF5_MAT) {
    CHKERRQ(PetscStrallocpy("jc",&i_name));
    CHKERRQ(PetscStrallocpy("ir",&j_name));
    CHKERRQ(PetscStrallocpy("data",&a_name));
    CHKERRQ(PetscStrallocpy("MATLAB_sparse",&c_name));
  } else {
    /* TODO Once corresponding MatView is implemented, change the names to i,j,a */
    /* TODO Maybe there could be both namings in the file, using "symbolic link" features of HDF5. */
    CHKERRQ(PetscStrallocpy("jc",&i_name));
    CHKERRQ(PetscStrallocpy("ir",&j_name));
    CHKERRQ(PetscStrallocpy("data",&a_name));
    CHKERRQ(PetscStrallocpy("MATLAB_sparse",&c_name));
  }

  ierr = PetscOptionsBegin(comm,NULL,"Options for loading matrix from HDF5","Mat");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsInt("-matload_block_size","Set the blocksize used to store the matrix","MatLoad",bs,&bs,&flg));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (flg) {
    CHKERRQ(MatSetBlockSize(mat, bs));
  }

  CHKERRQ(PetscViewerHDF5PushGroup(viewer,mat_name));
  CHKERRQ(PetscViewerHDF5ReadAttribute(viewer,NULL,c_name,PETSC_INT,NULL,&N));
  CHKERRQ(PetscViewerHDF5ReadSizes(viewer, i_name, NULL, &M));
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
  CHKERRQ(PetscLayoutSetUp(mat->rmap));
  CHKERRQ(PetscLayoutSetUp(mat->cmap));
  m = mat->rmap->n;

  /* Read array i (array of row indices) */
  CHKERRQ(PetscMalloc1(m+1, &i)); /* allocate i with one more position for local number of nonzeros on each rank */
  i[0] = i[m] = 0; /* make the last entry always defined - the code block below overwrites it just on last rank */
  if (rank == size-1) m++; /* in the loaded array i_glob, only the last rank has one more position with the global number of nonzeros */
  M++;
  CHKERRQ(ISCreate(comm,&is_i));
  CHKERRQ(PetscObjectSetName((PetscObject)is_i,i_name));
  CHKERRQ(PetscLayoutSetLocalSize(is_i->map,m));
  CHKERRQ(PetscLayoutSetSize(is_i->map,M));
  CHKERRQ(ISLoad(is_i,viewer));
  CHKERRQ(ISGetIndices(is_i,&i_glob));
  CHKERRQ(PetscArraycpy(i,i_glob,m));

  /* Reset m and M to the matrix sizes */
  m = mat->rmap->n;
  M--;

  /* Create PetscLayout for j and a vectors; construct ranges first */
  CHKERRQ(PetscMalloc1(size+1, &range));
  CHKERRMPI(MPI_Allgather(i, 1, MPIU_INT, range, 1, MPIU_INT, comm));
  /* Last rank has global number of nonzeros (= length of j and a arrays) in i[m] (last i entry) so broadcast it */
  range[size] = i[m];
  CHKERRMPI(MPI_Bcast(&range[size], 1, MPIU_INT, size-1, comm));
  for (p=size-1; p>0; p--) {
    if (!range[p]) range[p] = range[p+1]; /* for ranks with 0 rows, take the value from the next processor */
  }
  i[m] = range[rank+1]; /* i[m] (last i entry) is equal to next rank's offset */
  /* Deduce rstart, rend, n and N from the ranges */
  CHKERRQ(PetscLayoutCreateFromRanges(comm,range,PETSC_OWN_POINTER,1,&jmap));

  /* Convert global to local indexing of rows */
  for (p=1; p<m+1; ++p) i[p] -= i[0];
  i[0] = 0;

  /* Read array j (array of column indices) */
  CHKERRQ(ISCreate(comm,&is_j));
  CHKERRQ(PetscObjectSetName((PetscObject)is_j,j_name));
  CHKERRQ(PetscLayoutDuplicate(jmap,&is_j->map));
  CHKERRQ(ISLoad(is_j,viewer));
  CHKERRQ(ISGetIndices(is_j,&j));

  /* Read array a (array of values) */
  CHKERRQ(VecCreate(comm,&vec_a));
  CHKERRQ(PetscObjectSetName((PetscObject)vec_a,a_name));
  CHKERRQ(PetscLayoutDuplicate(jmap,&vec_a->map));
  CHKERRQ(VecLoad(vec_a,viewer));
  CHKERRQ(VecGetArrayRead(vec_a,&a));

  /* populate matrix */
  if (!((PetscObject)mat)->type_name) {
    CHKERRQ(MatSetType(mat,MATAIJ));
  }
  CHKERRQ(MatSeqAIJSetPreallocationCSR(mat,i,j,a));
  CHKERRQ(MatMPIAIJSetPreallocationCSR(mat,i,j,a));
  /*
  CHKERRQ(MatSeqBAIJSetPreallocationCSR(mat,bs,i,j,a));
  CHKERRQ(MatMPIBAIJSetPreallocationCSR(mat,bs,i,j,a));
  */

  if (format==PETSC_VIEWER_HDF5_MAT && !mat->symmetric) {
    /* Transpose the input matrix back */
    CHKERRQ(MatTranspose(mat,MAT_INPLACE_MATRIX,&mat));
  }

  CHKERRQ(PetscViewerHDF5PopGroup(viewer));
  CHKERRQ(PetscFree(i_name));
  CHKERRQ(PetscFree(j_name));
  CHKERRQ(PetscFree(a_name));
  CHKERRQ(PetscFree(c_name));
  CHKERRQ(PetscLayoutDestroy(&jmap));
  CHKERRQ(PetscFree(i));
  CHKERRQ(ISRestoreIndices(is_i,&i_glob));
  CHKERRQ(ISRestoreIndices(is_j,&j));
  CHKERRQ(VecRestoreArrayRead(vec_a,&a));
  CHKERRQ(ISDestroy(&is_i));
  CHKERRQ(ISDestroy(&is_j));
  CHKERRQ(VecDestroy(&vec_a));
  PetscFunctionReturn(0);
}
#endif
