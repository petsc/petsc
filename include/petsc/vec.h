//
//    Include file for C++ class based interface to PETSc
//
#if !defined(__cplusplus_vec_h)
#define __cplusplus_vec_h

#include "petsc/petsc.h"
#include "petscvec.h"

namespace PETSc {
  class Vec : public ::PETSc::Object{
    public:
      Vec() {this->vec = 0;};
      Vec(MPI_Comm comm) {VecCreate(comm,&this->vec);this->obj = 0;};
      ~Vec() {if (this->vec) VecDestroy(this->vec);}
      ::Vec vec;
      void set(::PETSc::Scalar s) {VecSet(&s,this->vec);}
  };

  namespace sVec {
    class Seq : public ::PETSc::Vec {
      public:
        Seq(PetscInt l) {VecCreateSeq(l,&this->vec);}
        ~Seq() {;};
    };

    class MPI : public ::PETSc::Vec {
      public:
        MPI(MPI_Comm comm,PetscInt l,PetscInt g) {VecCreateMPI(comm,l,g,&this->vec);}
        MPI(MPI_Comm comm,PetscInt l) {VecCreateMPI(comm,l,PETSC_DETERMINE,&this->vec);}
        ~MPI() {;};
    };
  }
}

#endif
