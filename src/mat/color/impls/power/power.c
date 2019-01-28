#include <petsc/private/matimpl.h>      /*I "petscmat.h"  I*/

static PetscErrorCode MatColoringApply_Power(MatColoring mc,ISColoring *iscoloring)
{
  PetscErrorCode  ierr;
  Mat             m = mc->mat,mp,ms;
  MatColoring     imc;
  PetscInt        i;
  const char      *optionsprefix;

  PetscFunctionBegin;
  /* square the matrix repeatedly if necessary */
  if (mc->dist == 1) {
    mp = m;
  } else {
    ierr = MatMatMult(m,m,MAT_INITIAL_MATRIX,2.0,&mp);CHKERRQ(ierr);
    for (i=2;i<mc->dist;i++) {
      ms = mp;
      ierr = MatMatMult(m,ms,MAT_INITIAL_MATRIX,2.0,&mp);CHKERRQ(ierr);
      ierr = MatDestroy(&ms);CHKERRQ(ierr);
    }
  }
  ierr = MatColoringCreate(mp,&imc);CHKERRQ(ierr);
  ierr = PetscObjectGetOptionsPrefix((PetscObject)mc,&optionsprefix);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)imc,optionsprefix);CHKERRQ(ierr);
  ierr = PetscObjectAppendOptionsPrefix((PetscObject)imc,"power_");CHKERRQ(ierr);
  ierr = MatColoringSetType(imc,MATCOLORINGGREEDY);CHKERRQ(ierr);
  ierr = MatColoringSetDistance(imc,1);CHKERRQ(ierr);
  ierr = MatColoringSetWeightType(imc,mc->weight_type);CHKERRQ(ierr);
  ierr = MatColoringSetFromOptions(imc);CHKERRQ(ierr);
  ierr = MatColoringApply(imc,iscoloring);CHKERRQ(ierr);
  ierr = MatColoringDestroy(&imc);CHKERRQ(ierr);
  if (mp != m) {ierr = MatDestroy(&mp);CHKERRQ(ierr);}
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
