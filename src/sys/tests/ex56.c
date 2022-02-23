
static char help[] = "Tests PetscHasExternalPackage().\n\n";

#include <petscsys.h>

int main(int argc,char **argv)
{
  char              pkg[128] = "hdf5";
  PetscBool         has,flg;
  PetscErrorCode    ierr;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  ierr = PetscOptionsGetString(NULL,NULL,"-pkg",pkg,sizeof(pkg),NULL);CHKERRQ(ierr);
  ierr = PetscHasExternalPackage(pkg, &has);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "PETSc has %s? %s\n", pkg, PetscBools[has]);CHKERRQ(ierr);
  ierr = PetscStrcmp(pkg, "hdf5", &flg);CHKERRQ(ierr);
#if defined(PETSC_HAVE_HDF5)
  PetscCheckFalse(flg && !has,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "PetscHasExternalPackage() says HDF5 is not configured but PETSC_HAVE_HDF5 is defined");
#else
  PetscCheckFalse(flg && has,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "PetscHasExternalPackage() says HDF5 is configured but PETSC_HAVE_HDF5 is undefined");
#endif
  ierr = PetscStrcmp(pkg, "parmetis", &flg);CHKERRQ(ierr);
#if defined(PETSC_HAVE_PARMETIS)
  PetscCheckFalse(flg && !has,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "PetscHasExternalPackage() says PARMETIS is not configured but PETSC_HAVE_PARMETIS is defined");
#else
  PetscCheckFalse(flg && has,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "PetscHasExternalPackage() says PARMETIS is configured but PETSC_HAVE_PARMETIS is undefined");
#endif
  ierr = PetscStrcmp(pkg, "yaml", &flg);CHKERRQ(ierr);
#if defined(PETSC_HAVE_YAML)
  PetscCheckFalse(flg && !has,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "PetscHasExternalPackage() says YAML is not configured but PETSC_HAVE_YAML is defined");
#else
  PetscCheckFalse(flg && has,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "PetscHasExternalPackage() says YAML is configured but PETSC_HAVE_YAML is undefined");
#endif
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: blaslapack
      args: -pkg blaslapack
   test:
      suffix: hdf5
      requires: hdf5
      args: -pkg hdf5
   test:
      suffix: no-hdf5
      requires: !hdf5
      args: -pkg hdf5
   test:
      suffix: parmetis
      requires: parmetis
      args: -pkg parmetis
   test:
      suffix: no-parmetis
      requires: !parmetis
      args: -pkg parmetis
   test:
      suffix: yaml
      requires: yaml
      args: -pkg yaml
   test:
      suffix: no-yaml
      requires: !yaml
      args: -pkg yaml

TEST*/
