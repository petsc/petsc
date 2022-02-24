#include "petsc.h"
#include "petscviennacl.h"
#include <viennacl/vector.hpp>
typedef viennacl::vector<PetscScalar> ViennaclVector;

int main(int argc,char *argv[])
{
  PetscErrorCode ierr;
  Vec            x,y;
  PetscInt       n = 5;
  ViennaclVector *x_vcl;

  ierr = PetscInitialize(&argc,&argv,(char*)0,NULL);if (ierr) return ierr;
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&x));
  CHKERRQ(VecSetSizes(x,n,PETSC_DECIDE));
  CHKERRQ(VecSetType(x,VECVIENNACL));
  CHKERRQ(VecSet(x,42.0));

  CHKERRQ(VecViennaCLGetArray(x,&x_vcl));

  CHKERRQ(VecCreateSeqViennaCLWithArray(PETSC_COMM_WORLD,1,n,(const ViennaclVector *)x_vcl,&y));

  // Operated on 'y', but 'x' would also be changed since both
  // 'x' and 'y' share the same viennacl vector.
  CHKERRQ(VecScale(y,2.0));

  CHKERRQ(VecViennaCLRestoreArray(x,&x_vcl));

  // Expected output: 'x' is a 5-vector with all entries as '84'.
  CHKERRQ(VecView(x,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(VecDestroy(&y));
  CHKERRQ(VecDestroy(&x));

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
      requires: viennacl defined(PETSC_HAVE_VIENNACL_NO_CUDA)

   test:
      nsize: 1
      suffix: 1
      args: -viennacl_backend opencl -viennacl_opencl_device_type gpu

TEST*/
