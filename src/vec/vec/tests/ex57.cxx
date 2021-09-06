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
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,n,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetType(x,VECVIENNACL);CHKERRQ(ierr);
  ierr = VecSet(x,42.0);CHKERRQ(ierr);

  ierr = VecViennaCLGetArray(x,&x_vcl);CHKERRQ(ierr);

  ierr = VecCreateSeqViennaCLWithArray(PETSC_COMM_WORLD,1,n,(const ViennaclVector *)x_vcl,&y);CHKERRQ(ierr);

  // Operated on 'y', but 'x' would also be changed since both
  // 'x' and 'y' share the same viennacl vector.
  ierr = VecScale(y,2.0);CHKERRQ(ierr);

  ierr = VecViennaCLRestoreArray(x,&x_vcl);CHKERRQ(ierr);

  // Expected output: 'x' is a 5-vector with all entries as '84'.
  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);

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
