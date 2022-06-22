
static char help[] = "Tests PetscHasExternalPackage().\n\n";

#include <petscsys.h>

int main(int argc,char **argv)
{
  char              pkg[128] = "hdf5";
  PetscBool         has,flg;

  PetscCall(PetscInitialize(&argc,&argv,NULL,help));
  PetscCall(PetscOptionsGetString(NULL,NULL,"-pkg",pkg,sizeof(pkg),NULL));
  PetscCall(PetscHasExternalPackage(pkg, &has));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "PETSc has %s? %s\n", pkg, PetscBools[has]));
  PetscCall(PetscStrcmp(pkg, "hdf5", &flg));
#if defined(PETSC_HAVE_HDF5)
  PetscCheck(!flg || has,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "PetscHasExternalPackage() says HDF5 is not configured but PETSC_HAVE_HDF5 is defined");
#else
  PetscCheck(!flg || !has,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "PetscHasExternalPackage() says HDF5 is configured but PETSC_HAVE_HDF5 is undefined");
#endif
  PetscCall(PetscStrcmp(pkg, "parmetis", &flg));
#if defined(PETSC_HAVE_PARMETIS)
  PetscCheck(!flg || has,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "PetscHasExternalPackage() says PARMETIS is not configured but PETSC_HAVE_PARMETIS is defined");
#else
  PetscCheck(!flg || !has,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "PetscHasExternalPackage() says PARMETIS is configured but PETSC_HAVE_PARMETIS is undefined");
#endif
  PetscCall(PetscStrcmp(pkg, "yaml", &flg));
#if defined(PETSC_HAVE_YAML)
  PetscCheck(!flg || has,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "PetscHasExternalPackage() says YAML is not configured but PETSC_HAVE_YAML is defined");
#else
  PetscCheck(!flg || !has,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "PetscHasExternalPackage() says YAML is configured but PETSC_HAVE_YAML is undefined");
#endif
  PetscCall(PetscFinalize());
  return 0;
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
