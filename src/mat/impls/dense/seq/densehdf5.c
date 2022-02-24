
/* TODO change to
#include <../src/mat/impls/dense/seq/dense.h>
*/
#include <../src/mat/impls/dense/mpi/mpidense.h>
#include <petsc/private/isimpl.h>
#include <petsc/private/vecimpl.h>
#include <petsc/private/viewerhdf5impl.h>
#include <petsclayouthdf5.h>

#if defined(PETSC_HAVE_HDF5)
PetscErrorCode MatLoad_Dense_HDF5(Mat mat, PetscViewer viewer)
{
  PetscViewer_HDF5    *hdf5;
  hid_t               scalartype; /* scalar type (H5T_NATIVE_FLOAT or H5T_NATIVE_DOUBLE) */
  PetscLayout         vmap;
  PetscViewerFormat   format;
  PetscScalar         *a = NULL;
  const char          *mat_name = NULL;
  MPI_Comm            comm;
  PetscMPIInt         rank, size;

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
  hdf5 = (PetscViewer_HDF5*) viewer->data;
  /* we store dense matrix columns as blocks, like MATLAB save(filename,variables,'-v7.3') does */
  hdf5->horizontal = PETSC_TRUE;

  PetscCheckFalse(!((PetscObject)mat)->name,PetscObjectComm((PetscObject)mat), PETSC_ERR_SUP, "Mat name must be set with PetscObjectSetName() before MatLoad()");
#if defined(PETSC_USE_REAL_SINGLE)
  scalartype = H5T_NATIVE_FLOAT;
#elif defined(PETSC_USE_REAL___FLOAT128)
#error "HDF5 output with 128 bit floats not supported."
#elif defined(PETSC_USE_REAL___FP16)
#error "HDF5 output with 16 bit floats not supported."
#else
  scalartype = H5T_NATIVE_DOUBLE;
#endif

  CHKERRQ(PetscObjectGetComm((PetscObject)mat,&comm));
  CHKERRMPI(MPI_Comm_rank(comm,&rank));
  CHKERRMPI(MPI_Comm_size(comm,&size));
  CHKERRQ(PetscObjectGetName((PetscObject)mat,&mat_name));

  /* Convert user-defined rmap and cmap to the dataset layout */
  CHKERRQ(PetscLayoutCreate(PetscObjectComm((PetscObject)mat),&vmap));
  if (mat->rmap->n >= 0 && mat->cmap->N < 0) {
    /* We need to know mat->cmap->N if user specifies custom mat->rmap->n, otherwise the latter would get ignored below */
    CHKERRQ(PetscViewerHDF5ReadSizes(viewer, mat_name, &mat->cmap->N, NULL));
  }
  vmap->bs = mat->cmap->N;
  vmap->n = (mat->rmap->n < 0 || mat->cmap->N < 0) ? -1 : mat->rmap->n * mat->cmap->N;
  vmap->N = (mat->rmap->N < 0 || mat->cmap->N < 0) ? -1 : mat->rmap->N * mat->cmap->N;

  /* Read the dataset and setup its layout */
  /* Note: PetscViewerHDF5ReadSizes_Private takes into account that the dataset is transposed for MATLAB MAT files */
  CHKERRQ(PetscViewerHDF5Load(viewer, mat_name, vmap, scalartype, (void**)&a));

  /* Convert the dataset layout back to rmap and cmap */
  mat->cmap->N = vmap->bs;
  mat->rmap->n = vmap->n / mat->cmap->N;
  mat->rmap->N = vmap->N / mat->cmap->N;
  CHKERRQ(PetscLayoutSetUp(mat->rmap));
  CHKERRQ(PetscLayoutSetUp(mat->cmap));
  CHKERRQ(PetscLayoutDestroy(&vmap));

  /* TODO adding PetscCopyMode flag to MatSeqDenseSetPreallocation would make this code cleaner and simpler */
  {
    PetscBool flg;
    Mat_SeqDense *impl;
    CHKERRQ(PetscObjectTypeCompare((PetscObject)mat,MATSEQDENSE,&flg));
    if (flg) {
      impl = (Mat_SeqDense*)mat->data;
      CHKERRQ(MatSeqDenseSetPreallocation(mat,a));
    } else {
      Mat_MPIDense *implm = (Mat_MPIDense*)mat->data;
      CHKERRQ(MatMPIDenseSetPreallocation(mat,a));
      impl = (Mat_SeqDense*)implm->A->data;
    }
    impl->user_alloc = PETSC_FALSE;
  }

  CHKERRQ(MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}
#endif
