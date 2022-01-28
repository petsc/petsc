#include <petscmat.h>

#define NNORMS 6

static PetscErrorCode MatLoadComputeNorms(Mat data_mat, PetscViewer inp_viewer, PetscReal norms[])
{
  Mat            corr_mat;
  PetscInt       M,N;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatLoad(data_mat, inp_viewer);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(data_mat, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(data_mat, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatViewFromOptions(data_mat, NULL, "-view_mat");CHKERRQ(ierr);

  ierr = MatGetSize(data_mat, &M, &N);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Data matrix size: %" PetscInt_FMT " %" PetscInt_FMT "\n", M,N);CHKERRQ(ierr);

  /* compute matrix norms */
  ierr = MatNorm(data_mat, NORM_1, &norms[0]);CHKERRQ(ierr);
  ierr = MatNorm(data_mat, NORM_INFINITY, &norms[1]);CHKERRQ(ierr);
  ierr = MatNorm(data_mat, NORM_FROBENIUS, &norms[2]);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Data matrix norms: %g %g %g\n", (double)norms[0],(double)norms[1],(double)norms[2]);CHKERRQ(ierr);

  /* compute autocorrelation matrix */
  ierr = MatMatTransposeMult(data_mat, data_mat, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &corr_mat);CHKERRQ(ierr);

  /* compute autocorrelation matrix norms */
  ierr = MatNorm(corr_mat, NORM_1, &norms[3]);CHKERRQ(ierr);
  ierr = MatNorm(corr_mat, NORM_INFINITY, &norms[4]);CHKERRQ(ierr);
  ierr = MatNorm(corr_mat, NORM_FROBENIUS, &norms[5]);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Autocorrelation matrix norms: %g %g %g\n", (double)norms[3],(double)norms[4],(double)norms[5]);CHKERRQ(ierr);

  ierr = MatDestroy(&corr_mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode GetReader(MPI_Comm comm, const char option[], PetscViewer *r, PetscViewerFormat *fmt)
{
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsGetViewer(PETSC_COMM_SELF, NULL, NULL, option, r, fmt, &flg);CHKERRQ(ierr);
  if (flg) {
    PetscFileMode mode;
    ierr = PetscViewerFileGetMode(*r, &mode);CHKERRQ(ierr);
    flg = (PetscBool) (mode == FILE_MODE_READ);
  }
  PetscAssertFalse(!flg,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Need to specify %s viewer_type:file:format:read", option);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  PetscErrorCode    ierr;
  PetscInt          i;
  PetscReal         norms0[NNORMS], norms1[NNORMS];
  PetscViewer       inp_viewer;
  PetscViewerFormat fmt;
  Mat               data_mat;
  char              mat_name[PETSC_MAX_PATH_LEN]="dmatrix";

  ierr = PetscInitialize(&argc, &argv, NULL, NULL);if (ierr) return ierr;
  ierr = PetscOptionsGetString(NULL,NULL,"-mat_name",mat_name,sizeof(mat_name),NULL);CHKERRQ(ierr);

  /* load matrix sequentially */
  ierr = MatCreate(PETSC_COMM_SELF, &data_mat);CHKERRQ(ierr);
  ierr = MatSetType(data_mat,MATDENSE);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)data_mat, mat_name);CHKERRQ(ierr);
  ierr = GetReader(PETSC_COMM_SELF, "-serial_reader", &inp_viewer, &fmt);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(inp_viewer, fmt);CHKERRQ(ierr);
  ierr = MatLoadComputeNorms(data_mat, inp_viewer, norms0);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(inp_viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&inp_viewer);CHKERRQ(ierr);
  ierr = MatViewFromOptions(data_mat, NULL, "-view_serial_mat");CHKERRQ(ierr);
  ierr = MatDestroy(&data_mat);CHKERRQ(ierr);

  /* load matrix in parallel */
  ierr = MatCreate(PETSC_COMM_WORLD, &data_mat);CHKERRQ(ierr);
  ierr = MatSetType(data_mat,MATDENSE);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)data_mat, mat_name);CHKERRQ(ierr);
  ierr = GetReader(PETSC_COMM_WORLD, "-parallel_reader", &inp_viewer, &fmt);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(inp_viewer, fmt);CHKERRQ(ierr);
  ierr = MatLoadComputeNorms(data_mat, inp_viewer, norms1);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(inp_viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&inp_viewer);CHKERRQ(ierr);
  ierr = MatViewFromOptions(data_mat, NULL, "-view_parallel_mat");CHKERRQ(ierr);
  ierr = MatDestroy(&data_mat);CHKERRQ(ierr);

  for (i=0; i<NNORMS; i++) {
    PetscAssertFalse(PetscAbs(norms0[i] - norms1[i]) > PETSC_SMALL,PETSC_COMM_SELF, PETSC_ERR_PLIB, "norm0[%" PetscInt_FMT "] = %g != %g = norms1[%" PetscInt_FMT "]", i, (double)norms0[i], (double)norms1[i], i);
  }

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 1
    requires: hdf5 datafilespath complex
    args:  -serial_reader hdf5:${DATAFILESPATH}/matrices/hdf5/sample_data.h5::read -parallel_reader hdf5:${DATAFILESPATH}/matrices/hdf5/sample_data.h5::read
    nsize: {{1 2 4}}

  test:
    requires: hdf5 datafilespath
    args:  -serial_reader hdf5:${DATAFILESPATH}/matrices/hdf5/tiny_rectangular_mat.h5::read -parallel_reader hdf5:${DATAFILESPATH}/matrices/hdf5/tiny_rectangular_mat.h5::read
    nsize: {{1 2}}
    test:
      suffix: 2-complex
      requires: complex
      args: -mat_name ComplexMat
    test:
      suffix: 2-real
      requires: !complex
      args: -mat_name RealMat

TEST*/
