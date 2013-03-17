#include "petscconf.h"

#include "../src/mat/impls/aij/seq/aij.h"          /*I "petscmat.h" I*/
#include "petscbt.h"
#include "../src/vec/vec/impls/dvecimpl.h"
#include "petsc-private/vecimpl.h"


#include "../src/mat/impls/aij/seq/seqviennacl/viennaclmatimpl.h"
#include "viennacl/linalg/prod.hpp"



// Ne: Number of elements
// Nl: Number of dof per element
#undef __FUNCT__
#define __FUNCT__ "MatSetValuesBatch_SeqAIJViennaCL"
PetscErrorCode MatSetValuesBatch_SeqAIJViennaCL(Mat J, PetscInt Ne, PetscInt Nl, PetscInt *elemRows, const PetscScalar *elemMats)
{
  //TODO
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: Implementation of MatSetValuesBatch_SeqAIJViennaCL() missing.");
}
