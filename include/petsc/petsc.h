//
//    Include file for C++ class based interface to PETSc
//
#if !defined(__cplusplus_petsc_h)
#define __cplusplus_petsc_h

#include "include/petsc.h"

namespace PETSc {

  typedef PetscScalar Scalar;
  typedef PetscInt    Int;
  typedef PetscReal   Real;
  typedef PetscTruth  Truth;
  
  const ::PETSc::Truth TRUE  = PETSC_TRUE;
  const ::PETSc::Truth FALSE = PETSC_FALSE;

  class PETSc {
    public:
      PETSc(int *argc,char ***argv) {
        PetscInitialize(argc,argv);
      }
      ~PETSc() {
        PetscFinalize();
      }     
  };

  class Object {
    public:
      Object() {this->obj = 0;};
      ~Object() {if (this->obj) PetscObjectDestroy(this->obj);}
      PetscObject obj;
  };
}

#endif
