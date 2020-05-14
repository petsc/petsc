
#include <petsc/private/pcimpl.h>   /*I "petscpc.h" I*/
#include <../src/mat/impls/aij/seq/aij.h>

/*
   Private context (data structure) for the CP preconditioner.
*/
typedef struct {
  PetscInt    n,m;
  Vec         work;
  PetscScalar *d;       /* sum of squares of each column */
  PetscScalar *a;       /* non-zeros by column */
  PetscInt    *i,*j;    /* offsets of nonzeros by column, non-zero indices by column */
} PC_CP;


static PetscErrorCode PCSetUp_CP(PC pc)
{
  PC_CP          *cp = (PC_CP*)pc->data;
  PetscInt       i,j,*colcnt;
  PetscErrorCode ierr;
  PetscBool      flg;
  Mat_SeqAIJ     *aij = (Mat_SeqAIJ*)pc->pmat->data;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)pc->pmat,MATSEQAIJ,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Currently only handles SeqAIJ matrices");

  ierr = MatGetLocalSize(pc->pmat,&cp->m,&cp->n);CHKERRQ(ierr);
  if (cp->m != cp->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Currently only for square matrices");

  if (!cp->work) {ierr = MatCreateVecs(pc->pmat,&cp->work,NULL);CHKERRQ(ierr);}
  if (!cp->d) {ierr = PetscMalloc1(cp->n,&cp->d);CHKERRQ(ierr);}
  if (cp->a && pc->flag != SAME_NONZERO_PATTERN) {
    ierr  = PetscFree3(cp->a,cp->i,cp->j);CHKERRQ(ierr);
    cp->a = NULL;
  }

  /* convert to column format */
  if (!cp->a) {
    ierr = PetscMalloc3(aij->nz,&cp->a,cp->n+1,&cp->i,aij->nz,&cp->j);CHKERRQ(ierr);
  }
  ierr = PetscCalloc1(cp->n,&colcnt);CHKERRQ(ierr);

  for (i=0; i<aij->nz; i++) colcnt[aij->j[i]]++;
  cp->i[0] = 0;
  for (i=0; i<cp->n; i++) cp->i[i+1] = cp->i[i] + colcnt[i];
  ierr = PetscArrayzero(colcnt,cp->n);CHKERRQ(ierr);
  for (i=0; i<cp->m; i++) {  /* over rows */
    for (j=aij->i[i]; j<aij->i[i+1]; j++) {  /* over columns in row */
      cp->j[cp->i[aij->j[j]]+colcnt[aij->j[j]]]   = i;
      cp->a[cp->i[aij->j[j]]+colcnt[aij->j[j]]++] = aij->a[j];
    }
  }
  ierr = PetscFree(colcnt);CHKERRQ(ierr);

  /* compute sum of squares of each column d[] */
  for (i=0; i<cp->n; i++) {  /* over columns */
    cp->d[i] = 0.;
    for (j=cp->i[i]; j<cp->i[i+1]; j++) cp->d[i] += cp->a[j]*cp->a[j]; /* over rows in column */
    cp->d[i] = 1.0/cp->d[i];
  }
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
static PetscErrorCode PCApply_CP(PC pc,Vec bb,Vec xx)
{
  PC_CP          *cp = (PC_CP*)pc->data;
  PetscErrorCode ierr;
  PetscScalar    *b,*x,xt;
  PetscInt       i,j;

  PetscFunctionBegin;
  ierr = VecCopy(bb,cp->work);CHKERRQ(ierr);
  ierr = VecGetArray(cp->work,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);

  for (i=0; i<cp->n; i++) {  /* over columns */
    xt = 0.;
    for (j=cp->i[i]; j<cp->i[i+1]; j++) xt += cp->a[j]*b[cp->j[j]]; /* over rows in column */
    xt  *= cp->d[i];
    x[i] = xt;
    for (j=cp->i[i]; j<cp->i[i+1]; j++) b[cp->j[j]] -= xt*cp->a[j]; /* over rows in column updating b*/
  }
  for (i=cp->n-1; i>-1; i--) {  /* over columns */
    xt = 0.;
    for (j=cp->i[i]; j<cp->i[i+1]; j++) xt += cp->a[j]*b[cp->j[j]]; /* over rows in column */
    xt  *= cp->d[i];
    x[i] = xt;
    for (j=cp->i[i]; j<cp->i[i+1]; j++) b[cp->j[j]] -= xt*cp->a[j]; /* over rows in column updating b*/
  }

  ierr = VecRestoreArray(cp->work,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
static PetscErrorCode PCReset_CP(PC pc)
{
  PC_CP          *cp = (PC_CP*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(cp->d);CHKERRQ(ierr);
  ierr = VecDestroy(&cp->work);CHKERRQ(ierr);
  ierr = PetscFree3(cp->a,cp->i,cp->j);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_CP(PC pc)
{
  PC_CP          *cp = (PC_CP*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCReset_CP(pc);CHKERRQ(ierr);
  ierr = PetscFree(cp->d);CHKERRQ(ierr);
  ierr = PetscFree3(cp->a,cp->i,cp->j);CHKERRQ(ierr);
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetFromOptions_CP(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/*MC
     PCCP - a "column-projection" preconditioner

     This is a terrible preconditioner and is not recommended, ever!

     Loops over the entries of x computing dx_i (e_i is the unit vector in the ith direction) to
$
$        min || b - A(x + dx_i e_i ||_2
$        dx_i
$
$    That is, it changes a single entry of x to minimize the new residual norm.
$   Let A_i represent the ith column of A, then the minimization can be written as
$
$       min || r - (dx_i) A e_i ||_2
$       dx_i
$   or   min || r - (dx_i) A_i ||_2
$        dx_i
$
$    take the derivative with respect to dx_i to obtain
$        dx_i = (A_i^T A_i)^(-1) A_i^T r
$
$    This algorithm can be thought of as Gauss-Seidel on the normal equations

    Notes:
    This proceedure can also be done with block columns or any groups of columns
        but this is not coded.

      These "projections" can be done simultaneously for all columns (similar to Jacobi)
         or sequentially (similar to Gauss-Seidel/SOR). This is only coded for SOR type.

      This is related to, but not the same as "row projection" methods.

      This is currently coded only for SeqAIJ matrices in sequential (SOR) form.

  Level: intermediate

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PCJACOBI, PCSOR

M*/

PETSC_EXTERN PetscErrorCode PCCreate_CP(PC pc)
{
  PC_CP          *cp;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr     = PetscNewLog(pc,&cp);CHKERRQ(ierr);
  pc->data = (void*)cp;

  pc->ops->apply           = PCApply_CP;
  pc->ops->applytranspose  = PCApply_CP;
  pc->ops->setup           = PCSetUp_CP;
  pc->ops->reset           = PCReset_CP;
  pc->ops->destroy         = PCDestroy_CP;
  pc->ops->setfromoptions  = PCSetFromOptions_CP;
  pc->ops->view            = NULL;
  pc->ops->applyrichardson = NULL;
  PetscFunctionReturn(0);
}


