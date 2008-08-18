#include "petscvec.h"
#include "taosolver.h"
#include "private/taolinesearch_impl.h"

#undef __FUNCT__
#define __FUNCT__ "TaoLineSearchDestroy_MT"
static PetscErrorCode TaoLineSearchDestroy_MT(TaoLineSearch ls)
{
    PetscErrorCode ierr;
    PetscFunctionBegin;
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoLineSearchSetFromOptions_MT"
static PetscErrorCode TaoLineSearchSetFromOptions_MT(TaoLineSearch ls)
{
    PetscErrorCode ierr;
    PetscFunctionBegin;
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TaoLineSearchView_MT"
static PetscErrorCode TaoLineSearchView_MT(TaoLineSearch ls, PetscViewer pv)
{
    PetscErrorCode ierr;
    PetscTruth isascii;
    PetscFunctionBegin;
    ierr = PetscTypeCompare((PetscObject)pv, PETSC_VIEWER_ASCII, &isascii); CHKERRQ(ierr);
    if (isascii) {
	ierr = PetscViewerASCIIPrintf(pv,"  Line Search: MoreThuente.\n"); CHKERRQ(ierr);
    } else {
	SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported for MoreThuente TaoLineSearch",((PetscObject)pv)->type_name);
    }
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoLineSearchApply_MT"
/* @ TaoApply_LineSearch - This routine takes step length of 1.0.

   Input Parameters:
+  tao - TaoSolver context
.  X - current iterate (on output X contains new iterate, X + step*S)
.  f - objective function evaluated at X
.  G - gradient evaluated at X
-  D - search direction


   Info is set to 0.

@ */

static PetscErrorCode TaoLineSearchApply_MT(TaoLineSearch ls, Vec start_x, PetscReal start_f, Vec start_g, Vec step_direction) 
{
    PetscErrorCode ierr;
    PetscFunctionBegin;
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TaoLineSearchCreate_MT"
PetscErrorCode TaoLineSearchCreate_MT(TaoLineSearch ls)
{
    PetscErrorCode ierr;
    PetscFunctionBegin;
    PetscFunctionReturn(0);
}

