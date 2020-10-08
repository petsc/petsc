
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
  ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
  switch (format) {
    case PETSC_VIEWER_HDF5_PETSC:
    case PETSC_VIEWER_DEFAULT:
    case PETSC_VIEWER_NATIVE:
    case PETSC_VIEWER_HDF5_MAT:
      break;
    default:
      SETERRQ1(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"PetscViewerFormat %s not supported for HDF5 input.",PetscViewerFormats[format]);
  }

  ierr = PetscObjectGetComm((PetscObject)mat,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  ierr = PetscObjectGetName((PetscObject)mat,&mat_name);CHKERRQ(ierr);
  if (format==PETSC_VIEWER_HDF5_MAT) {
    ierr = PetscStrallocpy("jc",&i_name);CHKERRQ(ierr);
    ierr = PetscStrallocpy("ir",&j_name);CHKERRQ(ierr);
    ierr = PetscStrallocpy("data",&a_name);CHKERRQ(ierr);
    ierr = PetscStrallocpy("MATLAB_sparse",&c_name);CHKERRQ(ierr);
  } else {
    /* TODO Once corresponding MatView is implemented, change the names to i,j,a */
    /* TODO Maybe there could be both namings in the file, using "symbolic link" features of HDF5. */
    ierr = PetscStrallocpy("jc",&i_name);CHKERRQ(ierr);
    ierr = PetscStrallocpy("ir",&j_name);CHKERRQ(ierr);
    ierr = PetscStrallocpy("data",&a_name);CHKERRQ(ierr);
    ierr = PetscStrallocpy("MATLAB_sparse",&c_name);CHKERRQ(ierr);
  }

  ierr = PetscOptionsBegin(comm,NULL,"Options for loading matrix from HDF5","Mat");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-matload_block_size","Set the blocksize used to store the matrix","MatLoad",bs,&bs,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (flg) {
    ierr = MatSetBlockSize(mat, bs);CHKERRQ(ierr);
  }

  ierr = PetscViewerHDF5PushGroup(viewer,mat_name);CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadAttribute(viewer,NULL,c_name,PETSC_INT,&N);CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadSizes(viewer, i_name, NULL, &M);CHKERRQ(ierr);
  --M;  /* i has size M+1 as there is global number of nonzeros stored at the end */

  if (format==PETSC_VIEWER_HDF5_MAT && !mat->symmetric) {
    /* Swap row and columns layout for unallocated matrix. I want to avoid calling MatTranspose() just to transpose sparsity pattern and layout. */
    if (!mat->preallocated) {
      PetscLayout tmp;
      tmp = mat->rmap; mat->rmap = mat->cmap; mat->cmap = tmp;
    } else SETERRQ(comm,PETSC_ERR_SUP,"Not for preallocated matrix - we would need to transpose it here which we want to avoid");
  }

  /* If global sizes are set, check if they are consistent with that given in the file */
  if (mat->rmap->N >= 0 && mat->rmap->N != M) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Inconsistent # of rows: Matrix in file has (%D) and input matrix has (%D)",mat->rmap->N,M);
  if (mat->cmap->N >= 0 && mat->cmap->N != N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Inconsistent # of cols: Matrix in file has (%D) and input matrix has (%D)",mat->cmap->N,N);

  /* Determine ownership of all (block) rows and columns */
  mat->rmap->N = M;
  mat->cmap->N = N;
  ierr = PetscLayoutSetUp(mat->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(mat->cmap);CHKERRQ(ierr);
  m = mat->rmap->n;

  /* Read array i (array of row indices) */
  ierr = PetscMalloc1(m+1, &i);CHKERRQ(ierr); /* allocate i with one more position for local number of nonzeros on each rank */
  i[0] = i[m] = 0; /* make the last entry always defined - the code block below overwrites it just on last rank */
  if (rank == size-1) m++; /* in the loaded array i_glob, only the last rank has one more position with the global number of nonzeros */
  M++;
  ierr = ISCreate(comm,&is_i);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)is_i,i_name);CHKERRQ(ierr);
  ierr = PetscLayoutSetLocalSize(is_i->map,m);CHKERRQ(ierr);
  ierr = PetscLayoutSetSize(is_i->map,M);CHKERRQ(ierr);
  ierr = ISLoad(is_i,viewer);CHKERRQ(ierr);
  ierr = ISGetIndices(is_i,&i_glob);CHKERRQ(ierr);
  ierr = PetscArraycpy(i,i_glob,m);CHKERRQ(ierr);

  /* Reset m and M to the matrix sizes */
  m = mat->rmap->n;
  M--;

  /* Create PetscLayout for j and a vectors; construct ranges first */
  ierr = PetscMalloc1(size+1, &range);CHKERRQ(ierr);
  ierr = MPI_Allgather(i, 1, MPIU_INT, range, 1, MPIU_INT, comm);CHKERRMPI(ierr);
  /* Last rank has global number of nonzeros (= length of j and a arrays) in i[m] (last i entry) so broadcast it */
  range[size] = i[m];
  ierr = MPI_Bcast(&range[size], 1, MPIU_INT, size-1, comm);CHKERRMPI(ierr);
  for (p=size-1; p>0; p--) {
    if (!range[p]) range[p] = range[p+1]; /* for ranks with 0 rows, take the value from the next processor */
  }
  i[m] = range[rank+1]; /* i[m] (last i entry) is equal to next rank's offset */
  /* Deduce rstart, rend, n and N from the ranges */
  ierr = PetscLayoutCreateFromRanges(comm,range,PETSC_OWN_POINTER,1,&jmap);CHKERRQ(ierr);

  /* Convert global to local indexing of rows */
  for (p=1; p<m+1; ++p) i[p] -= i[0];
  i[0] = 0;

  /* Read array j (array of column indices) */
  ierr = ISCreate(comm,&is_j);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)is_j,j_name);CHKERRQ(ierr);
  ierr = PetscLayoutDuplicate(jmap,&is_j->map);CHKERRQ(ierr);
  ierr = ISLoad(is_j,viewer);CHKERRQ(ierr);
  ierr = ISGetIndices(is_j,&j);CHKERRQ(ierr);

  /* Read array a (array of values) */
  ierr = VecCreate(comm,&vec_a);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)vec_a,a_name);CHKERRQ(ierr);
  ierr = PetscLayoutDuplicate(jmap,&vec_a->map);CHKERRQ(ierr);
  ierr = VecLoad(vec_a,viewer);CHKERRQ(ierr);
  ierr = VecGetArrayRead(vec_a,&a);CHKERRQ(ierr);

  /* populate matrix */
  if (!((PetscObject)mat)->type_name) {
    ierr = MatSetType(mat,MATAIJ);CHKERRQ(ierr);
  }
  ierr = MatSeqAIJSetPreallocationCSR(mat,i,j,a);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocationCSR(mat,i,j,a);CHKERRQ(ierr);
  /*
  ierr = MatSeqBAIJSetPreallocationCSR(mat,bs,i,j,a);CHKERRQ(ierr);
  ierr = MatMPIBAIJSetPreallocationCSR(mat,bs,i,j,a);CHKERRQ(ierr);
  */

  if (format==PETSC_VIEWER_HDF5_MAT && !mat->symmetric) {
    /* Transpose the input matrix back */
    ierr = MatTranspose(mat,MAT_INPLACE_MATRIX,&mat);CHKERRQ(ierr);
  }

  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = PetscFree(i_name);CHKERRQ(ierr);
  ierr = PetscFree(j_name);CHKERRQ(ierr);
  ierr = PetscFree(a_name);CHKERRQ(ierr);
  ierr = PetscFree(c_name);CHKERRQ(ierr);
  ierr = PetscLayoutDestroy(&jmap);CHKERRQ(ierr);
  ierr = PetscFree(i);CHKERRQ(ierr);
  ierr = ISRestoreIndices(is_i,&i_glob);CHKERRQ(ierr);
  ierr = ISRestoreIndices(is_j,&j);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(vec_a,&a);CHKERRQ(ierr);
  ierr = ISDestroy(&is_i);CHKERRQ(ierr);
  ierr = ISDestroy(&is_j);CHKERRQ(ierr);
  ierr = VecDestroy(&vec_a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

