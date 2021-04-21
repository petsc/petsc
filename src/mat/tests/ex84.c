#include <petscmat.h>
#if defined(PETSC_HAVE_HDF5)
#include <petscviewerhdf5.h>
#endif

int main(int argc, char **argv)
{
  PetscErrorCode ierr;
  PetscReal      dmat_norm[3], cmat_norm[3];
  PetscViewer    inp_viewer;
  Mat            data_mat, corr_mat;
  char           file[PETSC_MAX_PATH_LEN],hdf5_name[PETSC_MAX_PATH_LEN];
  PetscBool      flg;

  ierr = PetscInitialize(&argc, &argv, NULL, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Must indicate binary file with the -f option");
  ierr = PetscOptionsGetString(NULL,NULL,"-hdf5_name",hdf5_name,sizeof(hdf5_name),&flg);CHKERRQ(ierr);
  /* Set up data matrix */
  ierr = MatCreate(PETSC_COMM_WORLD, &data_mat); CHKERRQ(ierr);
  ierr = MatSetType(data_mat,MATDENSE); CHKERRQ(ierr);
  if (flg) {
    ierr = PetscObjectSetName((PetscObject)data_mat, hdf5_name); CHKERRQ(ierr);
    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, file, FILE_MODE_READ, &inp_viewer); CHKERRQ(ierr);
  } else {
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, file, FILE_MODE_READ, &inp_viewer); CHKERRQ(ierr);
  }
  ierr = MatLoad(data_mat, inp_viewer); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&inp_viewer); CHKERRQ(ierr);
  ierr = MatAssemblyBegin(data_mat, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(data_mat, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatViewFromOptions(data_mat, NULL, "-view_mat");CHKERRQ(ierr);

  ierr = MatNorm(data_mat, NORM_1, &dmat_norm[0]); CHKERRQ(ierr);
  ierr = MatNorm(data_mat, NORM_INFINITY, &dmat_norm[1]); CHKERRQ(ierr);
  ierr = MatNorm(data_mat, NORM_FROBENIUS, &dmat_norm[2]); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Data matrix norms: %g %g %g\n", (double)dmat_norm[0],(double)dmat_norm[1],(double)dmat_norm[2]); CHKERRQ(ierr);

  /* compute autocorrelation matrix */
  ierr = MatMatTransposeMult(data_mat, data_mat, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &corr_mat); CHKERRQ(ierr);

  ierr = MatNorm(corr_mat, NORM_1, &cmat_norm[0]); CHKERRQ(ierr);
  ierr = MatNorm(corr_mat, NORM_INFINITY, &cmat_norm[1]); CHKERRQ(ierr);
  ierr = MatNorm(corr_mat, NORM_FROBENIUS, &cmat_norm[2]); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Autocorrelation matrix norms: %g %g %g\n", (double)cmat_norm[0],(double)cmat_norm[1],(double)cmat_norm[2]); CHKERRQ(ierr);

  ierr = MatDestroy(&data_mat); CHKERRQ(ierr);
  ierr = MatDestroy(&corr_mat); CHKERRQ(ierr);
  ierr = PetscFinalize(); CHKERRQ(ierr);
}
