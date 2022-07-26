#include <petscmat.h>

#define NNORMS 6

static PetscErrorCode MatLoadComputeNorms(Mat data_mat, PetscViewer inp_viewer, PetscReal norms[])
{
  Mat            corr_mat;
  PetscInt       M,N;

  PetscFunctionBegin;
  PetscCall(MatLoad(data_mat, inp_viewer));
  PetscCall(MatAssemblyBegin(data_mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(data_mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatViewFromOptions(data_mat, NULL, "-view_mat"));

  PetscCall(MatGetSize(data_mat, &M, &N));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Data matrix size: %" PetscInt_FMT " %" PetscInt_FMT "\n", M,N));

  /* compute matrix norms */
  PetscCall(MatNorm(data_mat, NORM_1, &norms[0]));
  PetscCall(MatNorm(data_mat, NORM_INFINITY, &norms[1]));
  PetscCall(MatNorm(data_mat, NORM_FROBENIUS, &norms[2]));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Data matrix norms: %g %g %g\n", (double)norms[0],(double)norms[1],(double)norms[2]));

  /* compute autocorrelation matrix */
  PetscCall(MatMatTransposeMult(data_mat, data_mat, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &corr_mat));

  /* compute autocorrelation matrix norms */
  PetscCall(MatNorm(corr_mat, NORM_1, &norms[3]));
  PetscCall(MatNorm(corr_mat, NORM_INFINITY, &norms[4]));
  PetscCall(MatNorm(corr_mat, NORM_FROBENIUS, &norms[5]));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Autocorrelation matrix norms: %g %g %g\n", (double)norms[3],(double)norms[4],(double)norms[5]));

  PetscCall(MatDestroy(&corr_mat));
  PetscFunctionReturn(0);
}

static PetscErrorCode GetReader(MPI_Comm comm, const char option[], PetscViewer *r, PetscViewerFormat *fmt)
{
  PetscBool      flg;

  PetscFunctionBegin;
  PetscCall(PetscOptionsGetViewer(PETSC_COMM_SELF, NULL, NULL, option, r, fmt, &flg));
  if (flg) {
    PetscFileMode mode;
    PetscCall(PetscViewerFileGetMode(*r, &mode));
    flg = (PetscBool) (mode == FILE_MODE_READ);
  }
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Need to specify %s viewer_type:file:format:read", option);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  PetscInt          i;
  PetscReal         norms0[NNORMS], norms1[NNORMS];
  PetscViewer       inp_viewer;
  PetscViewerFormat fmt;
  Mat               data_mat;
  char              mat_name[PETSC_MAX_PATH_LEN]="dmatrix";

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  PetscCall(PetscOptionsGetString(NULL,NULL,"-mat_name",mat_name,sizeof(mat_name),NULL));

  /* load matrix sequentially */
  PetscCall(MatCreate(PETSC_COMM_SELF, &data_mat));
  PetscCall(MatSetType(data_mat,MATDENSE));
  PetscCall(PetscObjectSetName((PetscObject)data_mat, mat_name));
  PetscCall(GetReader(PETSC_COMM_SELF, "-serial_reader", &inp_viewer, &fmt));
  PetscCall(PetscViewerPushFormat(inp_viewer, fmt));
  PetscCall(MatLoadComputeNorms(data_mat, inp_viewer, norms0));
  PetscCall(PetscViewerPopFormat(inp_viewer));
  PetscCall(PetscViewerDestroy(&inp_viewer));
  PetscCall(MatViewFromOptions(data_mat, NULL, "-view_serial_mat"));
  PetscCall(MatDestroy(&data_mat));

  /* load matrix in parallel */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &data_mat));
  PetscCall(MatSetType(data_mat,MATDENSE));
  PetscCall(PetscObjectSetName((PetscObject)data_mat, mat_name));
  PetscCall(GetReader(PETSC_COMM_WORLD, "-parallel_reader", &inp_viewer, &fmt));
  PetscCall(PetscViewerPushFormat(inp_viewer, fmt));
  PetscCall(MatLoadComputeNorms(data_mat, inp_viewer, norms1));
  PetscCall(PetscViewerPopFormat(inp_viewer));
  PetscCall(PetscViewerDestroy(&inp_viewer));
  PetscCall(MatViewFromOptions(data_mat, NULL, "-view_parallel_mat"));
  PetscCall(MatDestroy(&data_mat));

  for (i=0; i<NNORMS; i++) {
    PetscCheck(PetscAbs(norms0[i] - norms1[i]) <= PETSC_SMALL,PETSC_COMM_SELF, PETSC_ERR_PLIB, "norm0[%" PetscInt_FMT "] = %g != %g = norms1[%" PetscInt_FMT "]", i, (double)norms0[i], (double)norms1[i], i);
  }

  PetscCall(PetscFinalize());
  return 0;
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
