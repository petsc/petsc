#include <petsc/private/matimpl.h>      /*I "petscmat.h"  I*/

static PetscErrorCode MatColoringApply_Power(MatColoring mc,ISColoring *iscoloring)
{
  Mat             m = mc->mat,mp,ms;
  MatColoring     imc;
  PetscInt        i;
  const char      *optionsprefix;

  PetscFunctionBegin;
  /* square the matrix repeatedly if necessary */
  if (mc->dist == 1) {
    mp = m;
  } else {
    PetscCall(MatMatMult(m,m,MAT_INITIAL_MATRIX,2.0,&mp));
    for (i=2;i<mc->dist;i++) {
      ms = mp;
      PetscCall(MatMatMult(m,ms,MAT_INITIAL_MATRIX,2.0,&mp));
      PetscCall(MatDestroy(&ms));
    }
  }
  PetscCall(MatColoringCreate(mp,&imc));
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)mc,&optionsprefix));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)imc,optionsprefix));
  PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)imc,"power_"));
  PetscCall(MatColoringSetType(imc,MATCOLORINGGREEDY));
  PetscCall(MatColoringSetDistance(imc,1));
  PetscCall(MatColoringSetWeightType(imc,mc->weight_type));
  PetscCall(MatColoringSetFromOptions(imc));
  PetscCall(MatColoringApply(imc,iscoloring));
  PetscCall(MatColoringDestroy(&imc));
  if (mp != m) PetscCall(MatDestroy(&mp));
  PetscFunctionReturn(0);
}

/*MC
  MATCOLORINGPOWER - Take the matrix's nth power, then do one-coloring on it.

   Level: beginner

   Notes:
   This is merely a trivial test algorithm.

   Supports any distance coloring.

.seealso: `MatColoringCreate()`, `MatColoring`, `MatColoringSetType()`
M*/
PETSC_EXTERN PetscErrorCode MatColoringCreate_Power(MatColoring mc)
{
  PetscFunctionBegin;
  mc->ops->apply          = MatColoringApply_Power;
  mc->ops->view           = NULL;
  mc->ops->destroy        = NULL;
  mc->ops->setfromoptions = NULL;
  PetscFunctionReturn(0);
}
