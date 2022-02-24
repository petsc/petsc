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
    CHKERRQ(MatMatMult(m,m,MAT_INITIAL_MATRIX,2.0,&mp));
    for (i=2;i<mc->dist;i++) {
      ms = mp;
      CHKERRQ(MatMatMult(m,ms,MAT_INITIAL_MATRIX,2.0,&mp));
      CHKERRQ(MatDestroy(&ms));
    }
  }
  CHKERRQ(MatColoringCreate(mp,&imc));
  CHKERRQ(PetscObjectGetOptionsPrefix((PetscObject)mc,&optionsprefix));
  CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject)imc,optionsprefix));
  CHKERRQ(PetscObjectAppendOptionsPrefix((PetscObject)imc,"power_"));
  CHKERRQ(MatColoringSetType(imc,MATCOLORINGGREEDY));
  CHKERRQ(MatColoringSetDistance(imc,1));
  CHKERRQ(MatColoringSetWeightType(imc,mc->weight_type));
  CHKERRQ(MatColoringSetFromOptions(imc));
  CHKERRQ(MatColoringApply(imc,iscoloring));
  CHKERRQ(MatColoringDestroy(&imc));
  if (mp != m) CHKERRQ(MatDestroy(&mp));
  PetscFunctionReturn(0);
}

/*MC
  MATCOLORINGPOWER - Take the matrix's nth power, then do one-coloring on it.

   Level: beginner

   Notes:
   This is merely a trivial test algorithm.

   Supports any distance coloring.

.seealso: MatColoringCreate(), MatColoring, MatColoringSetType()
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
