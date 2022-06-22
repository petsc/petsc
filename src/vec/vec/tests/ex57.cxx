#include "petsc.h"
#include "petscviennacl.h"
#include <viennacl/vector.hpp>
typedef viennacl::vector<PetscScalar> ViennaclVector;

int main(int argc,char *argv[])
{
  Vec            x,y;
  PetscInt       n = 5;
  ViennaclVector *x_vcl;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,NULL));
  PetscCall(VecCreate(PETSC_COMM_WORLD,&x));
  PetscCall(VecSetSizes(x,n,PETSC_DECIDE));
  PetscCall(VecSetType(x,VECVIENNACL));
  PetscCall(VecSet(x,42.0));

  PetscCall(VecViennaCLGetArray(x,&x_vcl));

  PetscCall(VecCreateSeqViennaCLWithArray(PETSC_COMM_WORLD,1,n,(const ViennaclVector *)x_vcl,&y));

  // Operated on 'y', but 'x' would also be changed since both
  // 'x' and 'y' share the same viennacl vector.
  PetscCall(VecScale(y,2.0));

  PetscCall(VecViennaCLRestoreArray(x,&x_vcl));

  // Expected output: 'x' is a 5-vector with all entries as '84'.
  PetscCall(VecView(x,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&x));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: viennacl defined(PETSC_HAVE_VIENNACL_NO_CUDA)

   test:
      nsize: 1
      suffix: 1
      args: -viennacl_backend opencl -viennacl_opencl_device_type gpu

TEST*/
