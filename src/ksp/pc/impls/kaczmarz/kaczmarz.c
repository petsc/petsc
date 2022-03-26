#include <petsc/private/pcimpl.h>               /*I "petscpc.h" I*/

typedef struct {
  PetscReal  lambda; /* damping parameter */
  PetscBool  symmetric; /* apply the projections symmetrically */
} PC_Kaczmarz;

static PetscErrorCode PCDestroy_Kaczmarz(PC pc)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(pc->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_Kaczmarz(PC pc,Vec x,Vec y)
{
  PC_Kaczmarz       *jac = (PC_Kaczmarz*)pc->data;
  PetscInt          xs,xe,ys,ye,ncols,i,j;
  const PetscInt    *cols;
  const PetscScalar *vals,*xarray;
  PetscScalar       r;
  PetscReal         anrm;
  PetscScalar       *yarray;
  PetscReal         lambda=jac->lambda;

  PetscFunctionBegin;
  PetscCall(MatGetOwnershipRange(pc->pmat,&xs,&xe));
  PetscCall(MatGetOwnershipRangeColumn(pc->pmat,&ys,&ye));
  PetscCall(VecSet(y,0.));
  PetscCall(VecGetArrayRead(x,&xarray));
  PetscCall(VecGetArray(y,&yarray));
  for (i=xs;i<xe;i++) {
    /* get the maximum row width and row norms */
    PetscCall(MatGetRow(pc->pmat,i,&ncols,&cols,&vals));
    r = xarray[i-xs];
    anrm = 0.;
    for (j=0;j<ncols;j++) {
      if (cols[j] >= ys && cols[j] < ye) {
        r -= yarray[cols[j]-ys]*vals[j];
      }
      anrm += PetscRealPart(PetscSqr(vals[j]));
    }
    if (anrm > 0.) {
      for (j=0;j<ncols;j++) {
        if (cols[j] >= ys && cols[j] < ye) {
          yarray[cols[j]-ys] += vals[j]*lambda*r/anrm;
        }
      }
    }
    PetscCall(MatRestoreRow(pc->pmat,i,&ncols,&cols,&vals));
  }
  if (jac->symmetric) {
    for (i=xe-1;i>=xs;i--) {
      PetscCall(MatGetRow(pc->pmat,i,&ncols,&cols,&vals));
      r = xarray[i-xs];
      anrm = 0.;
      for (j=0;j<ncols;j++) {
        if (cols[j] >= ys && cols[j] < ye) {
          r -= yarray[cols[j]-ys]*vals[j];
        }
        anrm += PetscRealPart(PetscSqr(vals[j]));
      }
      if (anrm > 0.) {
        for (j=0;j<ncols;j++) {
          if (cols[j] >= ys && cols[j] < ye) {
            yarray[cols[j]-ys] += vals[j]*lambda*r/anrm;
          }
        }
      }
      PetscCall(MatRestoreRow(pc->pmat,i,&ncols,&cols,&vals));
    }
  }
  PetscCall(VecRestoreArray(y,&yarray));
  PetscCall(VecRestoreArrayRead(x,&xarray));
  PetscFunctionReturn(0);
}

PetscErrorCode PCSetFromOptions_Kaczmarz(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PC_Kaczmarz    *jac = (PC_Kaczmarz*)pc->data;

  PetscFunctionBegin;
  PetscCall(PetscOptionsHead(PetscOptionsObject,"Kaczmarz options"));
  PetscCall(PetscOptionsReal("-pc_kaczmarz_lambda","relaxation factor (0 < lambda)","",jac->lambda,&jac->lambda,NULL));
  PetscCall(PetscOptionsBool("-pc_kaczmarz_symmetric","apply row projections symmetrically","",jac->symmetric,&jac->symmetric,NULL));
  PetscCall(PetscOptionsTail());
  PetscFunctionReturn(0);
}

PetscErrorCode PCView_Kaczmarz(PC pc,PetscViewer viewer)
{
  PC_Kaczmarz    *jac = (PC_Kaczmarz*)pc->data;
  PetscBool      iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer,"  lambda = %g\n",(double)jac->lambda));
  }
  PetscFunctionReturn(0);
}

/*MC
     PCKaczmarz - Kaczmarz iteration

   Options Database Keys:
.  -pc_sor_lambda <1.0> - Sets damping parameter lambda

   Level: beginner

   Notes:
    In parallel this is block-Jacobi with Kaczmarz inner solve.

   References:
.  * - S. Kaczmarz, "Angenaherte Auflosing von Systemen Linearer Gleichungen",
   Bull. Internat. Acad. Polon. Sci. C1. A, 1937.

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC

M*/

PETSC_EXTERN PetscErrorCode PCCreate_Kaczmarz(PC pc)
{
  PC_Kaczmarz    *jac;

  PetscFunctionBegin;
  PetscCall(PetscNewLog(pc,&jac));

  pc->ops->apply           = PCApply_Kaczmarz;
  pc->ops->setfromoptions  = PCSetFromOptions_Kaczmarz;
  pc->ops->setup           = NULL;
  pc->ops->view            = PCView_Kaczmarz;
  pc->ops->destroy         = PCDestroy_Kaczmarz;
  pc->data                 = (void*)jac;
  jac->lambda              = 1.0;
  jac->symmetric           = PETSC_FALSE;
  PetscFunctionReturn(0);
}
