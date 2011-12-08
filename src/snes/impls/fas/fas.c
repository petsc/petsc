/* Defines the basic SNES object */
#include <../src/snes/impls/fas/fasimpls.h>    /*I  "petscsnesfas.h"  I*/

const char *SNESFASTypes[] = {"MULTIPLICATIVE","ADDITIVE","SNESFASType","SNES_FAS",0};

/*MC
Full Approximation Scheme nonlinear multigrid solver.

The nonlinear problem is solved via the repeated application of nonlinear preconditioners and coarse-grid corrections

.seealso: SNESCreate(), SNES, SNESSetType(), SNESType (for list of available types)
M*/

extern PetscErrorCode SNESDestroy_FAS(SNES snes);
extern PetscErrorCode SNESSetUp_FAS(SNES snes);
extern PetscErrorCode SNESSetFromOptions_FAS(SNES snes);
extern PetscErrorCode SNESView_FAS(SNES snes, PetscViewer viewer);
extern PetscErrorCode SNESSolve_FAS(SNES snes);
extern PetscErrorCode SNESReset_FAS(SNES snes);
extern PetscErrorCode SNESFASGalerkinDefaultFunction(SNES, Vec, Vec, void *);

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchSetType_FAS"
PetscErrorCode  SNESLineSearchSetType_FAS(SNES snes, SNESLineSearchType type)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  switch (type) {
  case SNES_LS_BASIC:
    ierr = SNESLineSearchSet(snes,SNESLineSearchNo,PETSC_NULL);CHKERRQ(ierr);
    break;
  case SNES_LS_BASIC_NONORMS:
    ierr = SNESLineSearchSet(snes,SNESLineSearchNoNorms,PETSC_NULL);CHKERRQ(ierr);
    break;
  case SNES_LS_QUADRATIC:
    ierr = SNESLineSearchSet(snes,SNESLineSearchQuadraticSecant,PETSC_NULL);CHKERRQ(ierr);
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP,"Unknown line search type.");
    break;
  }
  snes->ls_type = type;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN

#undef __FUNCT__
#define __FUNCT__ "SNESCreate_FAS"
PetscErrorCode SNESCreate_FAS(SNES snes)
{
  SNES_FAS * fas;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  snes->ops->destroy        = SNESDestroy_FAS;
  snes->ops->setup          = SNESSetUp_FAS;
  snes->ops->setfromoptions = SNESSetFromOptions_FAS;
  snes->ops->view           = SNESView_FAS;
  snes->ops->solve          = SNESSolve_FAS;
  snes->ops->reset          = SNESReset_FAS;

  snes->usesksp             = PETSC_FALSE;
  snes->usespc              = PETSC_FALSE;

  ierr = PetscNewLog(snes, SNES_FAS, &fas);CHKERRQ(ierr);
  snes->data                  = (void*) fas;
  fas->level                  = 0;
  fas->levels                 = 1;
  fas->n_cycles               = 1;
  fas->max_up_it              = 1;
  fas->max_down_it            = 1;
  fas->upsmooth               = PETSC_NULL;
  fas->downsmooth             = PETSC_NULL;
  fas->next                   = PETSC_NULL;
  fas->previous               = PETSC_NULL;
  fas->interpolate            = PETSC_NULL;
  fas->restrct                = PETSC_NULL;
  fas->inject                 = PETSC_NULL;
  fas->monitor                = PETSC_NULL;
  fas->usedmfornumberoflevels = PETSC_FALSE;
  fas->fastype                = SNES_FAS_MULTIPLICATIVE;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)snes,"SNESLineSearchSetType_C","SNESLineSearchSetType_FAS",SNESLineSearchSetType_FAS);CHKERRQ(ierr);
  ierr = SNESLineSearchSetType(snes, SNES_LS_QUADRATIC);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "SNESFASGetLevels"
/*@
   SNESFASGetLevels - Gets the number of levels in a FAS.

   Input Parameter:
.  snes - the preconditioner context

   Output parameter:
.  levels - the number of levels

   Level: advanced

.keywords: MG, get, levels, multigrid

.seealso: SNESFASSetLevels(), PCMGGetLevels()
@*/
PetscErrorCode SNESFASGetLevels(SNES snes, PetscInt * levels) {
  SNES_FAS * fas = (SNES_FAS *)snes->data;
  PetscFunctionBegin;
  *levels = fas->levels;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESFASSetCycles"
/*@
   SNESFASSetCycles - Sets the type cycles to use.  Use SNESFASSetCyclesOnLevel() for more
   complicated cycling.

   Logically Collective on SNES

   Input Parameters:
+  snes   - the multigrid context
-  cycles - the number of cycles -- 1 for V-cycle, 2 for W-cycle

   Options Database Key:
$  -snes_fas_cycles 1 or 2

   Level: advanced

.keywords: MG, set, cycles, V-cycle, W-cycle, multigrid

.seealso: SNESFASSetCyclesOnLevel()
@*/
PetscErrorCode SNESFASSetCycles(SNES snes, PetscInt cycles) {
  SNES_FAS * fas = (SNES_FAS *)snes->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  fas->n_cycles = cycles;
  if (fas->next) {
    ierr = SNESFASSetCycles(fas->next, cycles);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESFASSetCyclesOnLevel"
/*@
   SNESFASSetCyclesOnLevel - Sets the type cycles to use on a particular level.

   Logically Collective on SNES

   Input Parameters:
+  snes   - the multigrid context
.  level  - the level to set the number of cycles on
-  cycles - the number of cycles -- 1 for V-cycle, 2 for W-cycle

   Level: advanced

.keywords: MG, set, cycles, V-cycle, W-cycle, multigrid

.seealso: SNESFASSetCycles()
@*/
PetscErrorCode SNESFASSetCyclesOnLevel(SNES snes, PetscInt level, PetscInt cycles) {
  SNES_FAS * fas =  (SNES_FAS *)snes->data;
  PetscInt top_level = fas->level,i;

  PetscFunctionBegin;
  if (level > top_level)
    SETERRQ1(((PetscObject)snes)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Bad level number %d in SNESFASSetCyclesOnLevel", level);
  /* get to the correct level */
  for (i = fas->level; i > level; i--) {
    fas = (SNES_FAS *)fas->next->data;
  }
  if (fas->level != level)
    SETERRQ(((PetscObject)snes)->comm, PETSC_ERR_ARG_WRONG, "Inconsistent level labelling in SNESFASSetCyclesOnLevel");
  fas->n_cycles = cycles;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESFASSetGS"
/*@C
   SNESFASSetGS - Sets a nonlinear GS smoother and if it should be used.
   Use SNESFASSetGSOnLevel() for more complicated staging of smoothers
   and nonlinear preconditioners.

   Logically Collective on SNES

   Input Parameters:
+  snes    - the multigrid context
.  gsfunc  - the nonlinear smoother function
.  ctx     - the user context for the nonlinear smoother
-  use_gs  - whether to use the nonlinear smoother or not

   Level: advanced

.keywords: FAS, MG, set, cycles, gauss-seidel, multigrid

.seealso: SNESSetGS(), SNESFASSetGSOnLevel()
@*/
PetscErrorCode SNESFASSetGS(SNES snes, PetscErrorCode (*gsfunc)(SNES,Vec,Vec,void *), void * ctx, PetscBool use_gs) {
  PetscErrorCode ierr = 0;
  SNES_FAS       *fas = (SNES_FAS *)snes->data;
  PetscFunctionBegin;

  /* use or don't use it according to user wishes*/
  snes->usegs = use_gs;
  if (gsfunc) {
    ierr = SNESSetGS(snes, gsfunc, ctx);CHKERRQ(ierr);
    /* push the provided GS up the tree */
    if (fas->next) ierr = SNESFASSetGS(fas->next, gsfunc, ctx, use_gs);CHKERRQ(ierr);
  } else if (snes->ops->computegs) {
    /* assume that the user has set the GS solver at this level */
    if (fas->next) ierr = SNESFASSetGS(fas->next, PETSC_NULL, PETSC_NULL, use_gs);CHKERRQ(ierr);
  } else if (use_gs) {
    SETERRQ1(((PetscObject)snes)->comm, PETSC_ERR_ARG_WRONG, "No user Gauss-Seidel function provided in SNESFASSetGS on level %d", fas->level);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESFASSetGSOnLevel"
/*@C
   SNESFASSetGSOnLevel - Sets the nonlinear smoother on a particular level.

   Logically Collective on SNES

   Input Parameters:
+  snes    - the multigrid context
.  level   - the level to set the nonlinear smoother on
.  gsfunc  - the nonlinear smoother function
.  ctx     - the user context for the nonlinear smoother
-  use_gs  - whether to use the nonlinear smoother or not

   Level: advanced

.keywords: FAS, MG, set, cycles, Gauss-Seidel, multigrid

.seealso: SNESSetGS(), SNESFASSetGS()
@*/
PetscErrorCode SNESFASSetGSOnLevel(SNES snes, PetscInt level, PetscErrorCode (*gsfunc)(SNES,Vec,Vec,void *), void * ctx, PetscBool use_gs) {
  SNES_FAS       *fas =  (SNES_FAS *)snes->data;
  PetscErrorCode ierr;
  PetscInt       top_level = fas->level,i;
  SNES           cur_snes = snes;
  PetscFunctionBegin;
  if (level > top_level)
    SETERRQ1(((PetscObject)snes)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Bad level number %d in SNESFASSetCyclesOnLevel", level);
  /* get to the correct level */
  for (i = fas->level; i > level; i--) {
    fas = (SNES_FAS *)fas->next->data;
    cur_snes = fas->next;
  }
  if (fas->level != level)
    SETERRQ(((PetscObject)snes)->comm, PETSC_ERR_ARG_WRONG, "Inconsistent level labelling in SNESFASSetCyclesOnLevel");
  snes->usegs = use_gs;
  if (gsfunc) {
    ierr = SNESSetGS(cur_snes, gsfunc, ctx);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESFASGetSNES"
/*@
   SNESFASGetSNES - Gets the SNES corresponding to a particular
   level of the FAS hierarchy.

   Input Parameters:
+  snes    - the multigrid context
   level   - the level to get
-  lsnes   - whether to use the nonlinear smoother or not

   Level: advanced

.keywords: FAS, MG, set, cycles, Gauss-Seidel, multigrid

.seealso: SNESFASSetLevels(), SNESFASGetLevels()
@*/
PetscErrorCode SNESFASGetSNES(SNES snes, PetscInt level, SNES * lsnes) {
  SNES_FAS * fas = (SNES_FAS *)snes->data;
  PetscInt levels = fas->level;
  PetscInt i;
  PetscFunctionBegin;
  *lsnes = snes;
  if (fas->level < level) {
    SETERRQ(((PetscObject)snes)->comm, PETSC_ERR_ARG_OUTOFRANGE, "SNESFASGetSNES should only be called on a finer SNESFAS instance than the level.");
  }
  if (level > levels - 1) {
    SETERRQ(((PetscObject)snes)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Level %d doesn't exist in the SNESFAS.");
  }
  for (i = fas->level; i > level; i--) {
    *lsnes = fas->next;
    fas = (SNES_FAS *)(*lsnes)->data;
  }
  if (fas->level != level) SETERRQ(((PetscObject)snes)->comm, PETSC_ERR_ARG_OUTOFRANGE, "SNESFASGetSNES didn't return the right level!");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESFASSetType"
/*@
SNESFASSetType - Sets the update and correction type used for FAS.
e


@*/
PetscErrorCode  SNESFASSetType(SNES snes,SNESFASType fastype)
{
  SNES_FAS                   *fas = (SNES_FAS*)snes->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidLogicalCollectiveEnum(snes,fastype,2);
  fas->fastype = fastype;
  PetscFunctionReturn(0);
}



#undef __FUNCT__
#define __FUNCT__ "SNESFASSetLevels"
/*@C
   SNESFASSetLevels - Sets the number of levels to use with FAS.
   Must be called before any other FAS routine.

   Input Parameters:
+  snes   - the snes context
.  levels - the number of levels
-  comms  - optional communicators for each level; this is to allow solving the coarser
            problems on smaller sets of processors. Use PETSC_NULL_OBJECT for default in
            Fortran.

   Level: intermediate

   Notes:
     If the number of levels is one then the multigrid uses the -fas_levels prefix
  for setting the level options rather than the -fas_coarse prefix.

.keywords: FAS, MG, set, levels, multigrid

.seealso: SNESFASGetLevels()
@*/
PetscErrorCode SNESFASSetLevels(SNES snes, PetscInt levels, MPI_Comm * comms) {
  PetscErrorCode ierr;
  PetscInt i;
  SNES_FAS * fas = (SNES_FAS *)snes->data;
  SNES prevsnes;
  MPI_Comm comm;
  PetscFunctionBegin;
  comm = ((PetscObject)snes)->comm;
  if (levels == fas->levels) {
    if (!comms)
      PetscFunctionReturn(0);
  }
  /* user has changed the number of levels; reset */
  ierr = SNESReset(snes);CHKERRQ(ierr);
  /* destroy any coarser levels if necessary */
  if (fas->next) SNESDestroy(&fas->next);CHKERRQ(ierr);
  fas->next = PETSC_NULL;
  fas->previous = PETSC_NULL;
  prevsnes = snes;
  /* setup the finest level */
  for (i = levels - 1; i >= 0; i--) {
    if (comms) comm = comms[i];
    fas->level = i;
    fas->levels = levels;
    fas->next = PETSC_NULL;
    if (i > 0) {
      ierr = SNESCreate(comm, &fas->next);CHKERRQ(ierr);
      ierr = SNESSetOptionsPrefix(fas->next,((PetscObject)snes)->prefix);CHKERRQ(ierr);
      ierr = SNESSetType(fas->next, SNESFAS);CHKERRQ(ierr);
      ierr = PetscObjectIncrementTabLevel((PetscObject)fas->next, (PetscObject)snes, levels - i);CHKERRQ(ierr);
      ((SNES_FAS *)fas->next->data)->previous = prevsnes;
      prevsnes = fas->next;
      fas = (SNES_FAS *)prevsnes->data;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESFASSetNumberSmoothUp"
/*@
   SNESFASSetNumberSmoothUp - Sets the number of post-smoothing steps to
   use on all levels.

   Logically Collective on PC

   Input Parameters:
+  snes - the multigrid context
-  n    - the number of smoothing steps

   Options Database Key:
.  -snes_fas_smoothup <n> - Sets number of pre-smoothing steps

   Level: advanced

.keywords: FAS, MG, smooth, down, pre-smoothing, steps, multigrid

.seealso: SNESFASSetNumberSmoothDown()
@*/
PetscErrorCode SNESFASSetNumberSmoothUp(SNES snes, PetscInt n) {
  SNES_FAS * fas =  (SNES_FAS *)snes->data;
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  fas->max_up_it = n;
  if (fas->next) {
    ierr = SNESFASSetNumberSmoothUp(fas->next, n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESFASSetNumberSmoothDown"
/*@
   SNESFASSetNumberSmoothDown - Sets the number of pre-smoothing steps to
   use on all levels.

   Logically Collective on PC

   Input Parameters:
+  snes - the multigrid context
-  n    - the number of smoothing steps

   Options Database Key:
.  -snes_fas_smoothdown <n> - Sets number of pre-smoothing steps

   Level: advanced

.keywords: FAS, MG, smooth, down, pre-smoothing, steps, multigrid

.seealso: SNESFASSetNumberSmoothUp()
@*/
PetscErrorCode SNESFASSetNumberSmoothDown(SNES snes, PetscInt n) {
  SNES_FAS * fas =  (SNES_FAS *)snes->data;
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  fas->max_down_it = n;
  if (fas->next) {
    ierr = SNESFASSetNumberSmoothDown(fas->next, n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESFASSetInterpolation"
/*@
   SNESFASSetInterpolation - Sets the function to be used to calculate the
   interpolation from l-1 to the lth level

   Input Parameters:
+  snes      - the multigrid context
.  mat       - the interpolation operator
-  level     - the level (0 is coarsest) to supply [do not supply 0]

   Level: advanced

   Notes:
          Usually this is the same matrix used also to set the restriction
    for the same level.

          One can pass in the interpolation matrix or its transpose; PETSc figures
    out from the matrix size which one it is.

.keywords:  FAS, multigrid, set, interpolate, level

.seealso: SNESFASSetInjection(), SNESFASSetRestriction(), SNESFASSetRscale()
@*/
PetscErrorCode SNESFASSetInterpolation(SNES snes, PetscInt level, Mat mat) {
  SNES_FAS * fas =  (SNES_FAS *)snes->data;
  PetscInt top_level = fas->level,i;

  PetscFunctionBegin;
  if (level > top_level)
    SETERRQ1(((PetscObject)snes)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Bad level number %d in SNESFASSetInterpolation", level);
  /* get to the correct level */
  for (i = fas->level; i > level; i--) {
    fas = (SNES_FAS *)fas->next->data;
  }
  if (fas->level != level)
    SETERRQ(((PetscObject)snes)->comm, PETSC_ERR_ARG_WRONG, "Inconsistent level labelling in SNESFASSetInterpolation");
  fas->interpolate = mat;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESFASSetRestriction"
/*@
   SNESFASSetRestriction - Sets the function to be used to restrict the defect
   from level l to l-1.

   Input Parameters:
+  snes  - the multigrid context
.  mat   - the restriction matrix
-  level - the level (0 is coarsest) to supply [Do not supply 0]

   Level: advanced

   Notes:
          Usually this is the same matrix used also to set the interpolation
    for the same level.

          One can pass in the interpolation matrix or its transpose; PETSc figures
    out from the matrix size which one it is.

         If you do not set this, the transpose of the Mat set with PCMGSetInterpolation()
    is used.

.keywords: FAS, MG, set, multigrid, restriction, level

.seealso: SNESFASSetInterpolation(), SNESFASSetInjection()
@*/
PetscErrorCode SNESFASSetRestriction(SNES snes, PetscInt level, Mat mat) {
  SNES_FAS * fas =  (SNES_FAS *)snes->data;
  PetscInt top_level = fas->level,i;

  PetscFunctionBegin;
  if (level > top_level)
    SETERRQ1(((PetscObject)snes)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Bad level number %d in SNESFASSetRestriction", level);
  /* get to the correct level */
  for (i = fas->level; i > level; i--) {
    fas = (SNES_FAS *)fas->next->data;
  }
  if (fas->level != level)
    SETERRQ(((PetscObject)snes)->comm, PETSC_ERR_ARG_WRONG, "Inconsistent level labelling in SNESFASSetRestriction");
  fas->restrct = mat;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SNESFASSetRScale"
/*@
   SNESFASSetRScale - Sets the scaling factor of the restriction
   operator from level l to l-1.

   Input Parameters:
+  snes   - the multigrid context
.  rscale - the restriction scaling
-  level  - the level (0 is coarsest) to supply [Do not supply 0]

   Level: advanced

   Notes:
         This is only used in the case that the injection is not set.

.keywords: FAS, MG, set, multigrid, restriction, level

.seealso: SNESFASSetInjection(), SNESFASSetRestriction()
@*/
PetscErrorCode SNESFASSetRScale(SNES snes, PetscInt level, Vec rscale) {
  SNES_FAS * fas =  (SNES_FAS *)snes->data;
  PetscInt top_level = fas->level,i;

  PetscFunctionBegin;
  if (level > top_level)
    SETERRQ1(((PetscObject)snes)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Bad level number %d in SNESFASSetRestriction", level);
  /* get to the correct level */
  for (i = fas->level; i > level; i--) {
    fas = (SNES_FAS *)fas->next->data;
  }
  if (fas->level != level)
    SETERRQ(((PetscObject)snes)->comm, PETSC_ERR_ARG_WRONG, "Inconsistent level labelling in SNESFASSetRestriction");
  fas->rscale = rscale;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SNESFASSetInjection"
/*@
   SNESFASSetInjection - Sets the function to be used to inject the solution
   from level l to l-1.

   Input Parameters:
+  snes  - the multigrid context
.  mat   - the restriction matrix
-  level - the level (0 is coarsest) to supply [Do not supply 0]

   Level: advanced

   Notes:
         If you do not set this, the restriction and rscale is used to
   project the solution instead.

.keywords: FAS, MG, set, multigrid, restriction, level

.seealso: SNESFASSetInterpolation(), SNESFASSetRestriction()
@*/
PetscErrorCode SNESFASSetInjection(SNES snes, PetscInt level, Mat mat) {
  SNES_FAS * fas =  (SNES_FAS *)snes->data;
  PetscInt top_level = fas->level,i;

  PetscFunctionBegin;
  if (level > top_level)
    SETERRQ1(((PetscObject)snes)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Bad level number %d in SNESFASSetInjection", level);
  /* get to the correct level */
  for (i = fas->level; i > level; i--) {
    fas = (SNES_FAS *)fas->next->data;
  }
  if (fas->level != level)
    SETERRQ(((PetscObject)snes)->comm, PETSC_ERR_ARG_WRONG, "Inconsistent level labelling in SNESFASSetInjection");
  fas->inject = mat;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESReset_FAS"
PetscErrorCode SNESReset_FAS(SNES snes)
{
  PetscErrorCode ierr = 0;
  SNES_FAS * fas = (SNES_FAS *)snes->data;

  PetscFunctionBegin;
  if (fas->upsmooth)   ierr = SNESReset(fas->upsmooth);CHKERRQ(ierr);
  if (fas->downsmooth) ierr = SNESReset(fas->downsmooth);CHKERRQ(ierr);
  if (fas->next)       ierr = SNESReset(fas->next);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESDestroy_FAS"
PetscErrorCode SNESDestroy_FAS(SNES snes)
{
  SNES_FAS * fas = (SNES_FAS *)snes->data;
  PetscErrorCode ierr = 0;

  PetscFunctionBegin;
  /* recursively resets and then destroys */
  if (fas->upsmooth)     ierr = SNESDestroy(&fas->upsmooth);CHKERRQ(ierr);
  if (fas->downsmooth)   ierr = SNESDestroy(&fas->downsmooth);CHKERRQ(ierr);
  if (fas->inject) {
    ierr = MatDestroy(&fas->inject);CHKERRQ(ierr);
  }
  if (fas->interpolate == fas->restrct) {
    if (fas->interpolate)  ierr = MatDestroy(&fas->interpolate);CHKERRQ(ierr);
    fas->restrct = PETSC_NULL;
  } else {
    if (fas->interpolate)  ierr = MatDestroy(&fas->interpolate);CHKERRQ(ierr);
    if (fas->restrct)      ierr = MatDestroy(&fas->restrct);CHKERRQ(ierr);
  }
  if (fas->rscale)       ierr = VecDestroy(&fas->rscale);CHKERRQ(ierr);
  if (fas->next)         ierr = SNESDestroy(&fas->next);CHKERRQ(ierr);
  ierr = PetscFree(fas);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSetUp_FAS"
PetscErrorCode SNESSetUp_FAS(SNES snes)
{
  SNES_FAS       *fas = (SNES_FAS *) snes->data;
  PetscErrorCode ierr;
  VecScatter     injscatter;
  PetscInt       dm_levels;

  PetscFunctionBegin;

  if (fas->usedmfornumberoflevels && (fas->level == fas->levels - 1)) {
    ierr = DMGetRefineLevel(snes->dm,&dm_levels);CHKERRQ(ierr);
    dm_levels++;
    if (dm_levels > fas->levels) {
      ierr = SNESFASSetLevels(snes,dm_levels,PETSC_NULL);CHKERRQ(ierr);
      ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
    }
  }

  if (fas->fastype == SNES_FAS_MULTIPLICATIVE) {
    ierr = SNESDefaultGetWork(snes, 1);CHKERRQ(ierr); /* work vectors used for intergrid transfers */
  } else {
    ierr = SNESDefaultGetWork(snes, 4);CHKERRQ(ierr); /* work vectors used for intergrid transfers */
  }

  /* setup the pre and post smoothers and set their function, jacobian, and gs evaluation routines if the user has neglected this */
  if (fas->upsmooth) {
    ierr = SNESSetFromOptions(fas->upsmooth);CHKERRQ(ierr);
    if (snes->ops->computefunction && !fas->upsmooth->ops->computefunction) {
      ierr = SNESSetFunction(fas->upsmooth, PETSC_NULL, snes->ops->computefunction, snes->funP);CHKERRQ(ierr);
    }
    if (snes->ops->computejacobian && !fas->upsmooth->ops->computejacobian) {
      ierr = SNESSetJacobian(fas->upsmooth, fas->upsmooth->jacobian, fas->upsmooth->jacobian_pre, snes->ops->computejacobian, snes->jacP);CHKERRQ(ierr);
    }
   if (snes->ops->computegs && !fas->upsmooth->ops->computegs) {
      ierr = SNESSetGS(fas->upsmooth, snes->ops->computegs, snes->gsP);CHKERRQ(ierr);
    }
  }
  if (fas->downsmooth) {
    ierr = SNESSetFromOptions(fas->downsmooth);CHKERRQ(ierr);
    if (snes->ops->computefunction && !fas->downsmooth->ops->computefunction) {
      ierr = SNESSetFunction(fas->downsmooth, PETSC_NULL, snes->ops->computefunction, snes->funP);CHKERRQ(ierr);
    }
    if (snes->ops->computejacobian && !fas->downsmooth->ops->computejacobian) {
      ierr = SNESSetJacobian(fas->downsmooth, fas->downsmooth->jacobian, fas->downsmooth->jacobian_pre, snes->ops->computejacobian, snes->jacP);CHKERRQ(ierr);
    }
   if (snes->ops->computegs && !fas->downsmooth->ops->computegs) {
      ierr = SNESSetGS(fas->downsmooth, snes->ops->computegs, snes->gsP);CHKERRQ(ierr);
    }
  }
  /*pass the smoother, function, and jacobian up to the next level if it's not user set already */
  if (fas->next) {
    if (fas->galerkin) {
      ierr = SNESSetFunction(fas->next, PETSC_NULL, SNESFASGalerkinDefaultFunction, fas->next);CHKERRQ(ierr);
    } else {
      if (snes->ops->computefunction && !fas->next->ops->computefunction) {
        ierr = SNESSetFunction(fas->next, PETSC_NULL, snes->ops->computefunction, snes->funP);CHKERRQ(ierr);
      }
      if (snes->ops->computejacobian && !fas->next->ops->computejacobian) {
        ierr = SNESSetJacobian(fas->next, fas->next->jacobian, fas->next->jacobian_pre, snes->ops->computejacobian, snes->jacP);CHKERRQ(ierr);
      }
      if (snes->ops->computegs && !fas->next->ops->computegs) {
        ierr = SNESSetGS(fas->next, snes->ops->computegs, snes->gsP);CHKERRQ(ierr);
      }
    }
  }
  if (snes->dm) {
    /* construct EVERYTHING from the DM -- including the progressive set of smoothers */
    if (fas->next) {
      /* for now -- assume the DM and the evaluation functions have been set externally */
      if (!fas->next->dm) {
        ierr = DMCoarsen(snes->dm, ((PetscObject)fas->next)->comm, &fas->next->dm);CHKERRQ(ierr);
        ierr = SNESSetDM(fas->next, fas->next->dm);CHKERRQ(ierr);
      }
      /* set the interpolation and restriction from the DM */
      if (!fas->interpolate) {
        ierr = DMCreateInterpolation(fas->next->dm, snes->dm, &fas->interpolate, &fas->rscale);CHKERRQ(ierr);
        fas->restrct = fas->interpolate;
      }
      /* set the injection from the DM */
      if (!fas->inject) {
        ierr = DMCreateInjection(fas->next->dm, snes->dm, &injscatter);CHKERRQ(ierr);
        ierr = MatCreateScatter(((PetscObject)snes)->comm, injscatter, &fas->inject);CHKERRQ(ierr);
        ierr = VecScatterDestroy(&injscatter);CHKERRQ(ierr);
      }
    }
    /* set the DMs of the pre and post-smoothers here */
    if (fas->upsmooth)  {ierr = SNESSetDM(fas->upsmooth,   snes->dm);CHKERRQ(ierr);}
    if (fas->downsmooth){ierr = SNESSetDM(fas->downsmooth, snes->dm);CHKERRQ(ierr);}
  }

  /* setup FAS work vectors */
  if (fas->galerkin) {
    ierr = VecDuplicate(snes->vec_sol, &fas->Xg);CHKERRQ(ierr);
    ierr = VecDuplicate(snes->vec_sol, &fas->Fg);CHKERRQ(ierr);
  }

  if (fas->next) {
   /* gotta set up the solution vector for this to work */
    if (!fas->next->vec_sol) {ierr = VecDuplicate(fas->rscale, &fas->next->vec_sol);CHKERRQ(ierr);}
    if (!fas->next->vec_rhs) {ierr = VecDuplicate(fas->rscale, &fas->next->vec_rhs);CHKERRQ(ierr);}
    ierr = SNESSetUp(fas->next);CHKERRQ(ierr);
  }
  /* got to set them all up at once */
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSetFromOptions_FAS"
PetscErrorCode SNESSetFromOptions_FAS(SNES snes)
{
  SNES_FAS       *fas = (SNES_FAS *) snes->data;
  PetscInt       levels = 1;
  PetscBool      flg, smoothflg, smoothupflg, smoothdownflg, monflg;
  PetscErrorCode ierr;
  const char     *def_smooth = SNESNRICHARDSON;
  char           pre_type[256];
  char           post_type[256];
  char           monfilename[PETSC_MAX_PATH_LEN];
  SNESFASType    fastype;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("SNESFAS Options-----------------------------------");CHKERRQ(ierr);

  /* number of levels -- only process on the finest level */
  if (fas->levels - 1 == fas->level) {
    ierr = PetscOptionsInt("-snes_fas_levels", "Number of Levels", "SNESFASSetLevels", levels, &levels, &flg);CHKERRQ(ierr);
    if (!flg && snes->dm) {
      ierr = DMGetRefineLevel(snes->dm,&levels);CHKERRQ(ierr);
      levels++;
      fas->usedmfornumberoflevels = PETSC_TRUE;
    }
    ierr = SNESFASSetLevels(snes, levels, PETSC_NULL);CHKERRQ(ierr);
  }

  /* type of pre and/or post smoothers -- set both at once */
  ierr = PetscMemcpy(post_type, def_smooth, 256);CHKERRQ(ierr);
  ierr = PetscMemcpy(pre_type, def_smooth, 256);CHKERRQ(ierr);
  fastype = fas->fastype;
  ierr = PetscOptionsEnum("-snes_fas_type","FAS correction type","SNESFASSetType",SNESFASTypes,(PetscEnum)fastype,(PetscEnum*)&fastype,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = SNESFASSetType(snes, fastype);CHKERRQ(ierr);
  }
  ierr = PetscOptionsList("-snes_fas_smoother_type","Nonlinear smoother method","SNESSetType",SNESList,def_smooth,pre_type,256,&smoothflg);CHKERRQ(ierr);
  if (smoothflg) {
    ierr = PetscMemcpy(post_type, pre_type, 256);CHKERRQ(ierr);
  } else {
    ierr = PetscOptionsList("-snes_fas_smoothup_type",  "Nonlinear smoother method","SNESSetType",SNESList,def_smooth,pre_type, 256,&smoothupflg);CHKERRQ(ierr);
    ierr = PetscOptionsList("-snes_fas_smoothdown_type","Nonlinear smoother method","SNESSetType",SNESList,def_smooth,post_type,256,&smoothdownflg);CHKERRQ(ierr);
  }

  /* options for the number of preconditioning cycles and cycle type */
  ierr = PetscOptionsInt("-snes_fas_smoothup","Number of post-smooth iterations","SNESFASSetNumberSmoothUp",fas->max_up_it,&fas->max_up_it,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-snes_fas_smoothdown","Number of pre-smooth iterations","SNESFASSetNumberSmoothUp",fas->max_down_it,&fas->max_down_it,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-snes_fas_cycles","Number of cycles","SNESFASSetCycles",fas->n_cycles,&fas->n_cycles,&flg);CHKERRQ(ierr);

  ierr = PetscOptionsBool("-snes_fas_galerkin", "Form coarse problems with Galerkin","SNESFAS",fas->galerkin,&fas->galerkin,&flg);CHKERRQ(ierr);

  ierr = PetscOptionsString("-snes_fas_monitor","Monitor FAS progress","SNESMonitorSet","stdout",monfilename,PETSC_MAX_PATH_LEN,&monflg);CHKERRQ(ierr);

  /* other options for the coarsest level */
  if (fas->level == 0) {
    ierr = PetscOptionsList("-snes_fas_coarse_smoother_type","Coarsest smoother method","SNESSetType",SNESList,def_smooth,pre_type,256,&smoothflg);CHKERRQ(ierr);
  }

  ierr = PetscOptionsTail();CHKERRQ(ierr);
  /* setup from the determined types if there is no pointwise procedure or smoother defined */

  if ((!fas->downsmooth) && ((smoothdownflg || smoothflg) || !snes->usegs)) {
    const char     *prefix;
    ierr = SNESGetOptionsPrefix(snes,&prefix);CHKERRQ(ierr);
    ierr = SNESCreate(((PetscObject)snes)->comm, &fas->downsmooth);CHKERRQ(ierr);
    ierr = SNESSetOptionsPrefix(fas->downsmooth,prefix);CHKERRQ(ierr);
    if (fas->level || (fas->levels == 1)) {
      ierr = SNESAppendOptionsPrefix(fas->downsmooth,"fas_levels_down_");CHKERRQ(ierr);
    } else {
      ierr = SNESAppendOptionsPrefix(fas->downsmooth,"fas_coarse_");CHKERRQ(ierr);
    }
    ierr = PetscObjectIncrementTabLevel((PetscObject)fas->downsmooth, (PetscObject)snes, 1);CHKERRQ(ierr);
    ierr = SNESSetType(fas->downsmooth, pre_type);CHKERRQ(ierr);
  }

  if ((!fas->upsmooth) && (fas->level != 0) && ((smoothupflg || smoothflg) || !snes->usegs)) {
    const char     *prefix;
    ierr = SNESGetOptionsPrefix(snes,&prefix);CHKERRQ(ierr);
    ierr = SNESCreate(((PetscObject)snes)->comm, &fas->upsmooth);CHKERRQ(ierr);
    ierr = SNESSetOptionsPrefix(fas->upsmooth,prefix);CHKERRQ(ierr);
    ierr = SNESAppendOptionsPrefix(fas->upsmooth,"fas_levels_up_");CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)fas->upsmooth, (PetscObject)snes, 1);CHKERRQ(ierr);
    ierr = SNESSetType(fas->upsmooth, pre_type);CHKERRQ(ierr);
  }
  if (fas->upsmooth) {
    ierr = SNESSetTolerances(fas->upsmooth, 0.0, 0.0, 0.0, fas->max_up_it, 1000);CHKERRQ(ierr);
  }

  if (fas->downsmooth) {
    ierr = SNESSetTolerances(fas->downsmooth, 0.0, 0.0, 0.0, fas->max_down_it, 1000);CHKERRQ(ierr);
  }

  if (fas->level != fas->levels - 1) {
    ierr = SNESSetTolerances(snes, 0.0, 0.0, 0.0, fas->n_cycles, 1000);CHKERRQ(ierr);
  }

  if (monflg) {
    fas->monitor = PETSC_VIEWER_STDOUT_(((PetscObject)snes)->comm);CHKERRQ(ierr);
    /* set the monitors for the upsmoother and downsmoother */
    ierr = SNESMonitorCancel(snes);CHKERRQ(ierr);
    ierr = SNESMonitorSet(snes,SNESMonitorDefault,PETSC_NULL,(PetscErrorCode (*)(void**))PetscViewerDestroy);CHKERRQ(ierr);
    if (fas->upsmooth)   ierr = SNESMonitorSet(fas->upsmooth,SNESMonitorDefault,PETSC_NULL,(PetscErrorCode (*)(void**))PetscViewerDestroy);CHKERRQ(ierr);
    if (fas->downsmooth) ierr = SNESMonitorSet(fas->downsmooth,SNESMonitorDefault,PETSC_NULL,(PetscErrorCode (*)(void**))PetscViewerDestroy);CHKERRQ(ierr);
  } else {
    /* unset the monitors on the coarse levels */
    if (fas->level != fas->levels - 1) {
      ierr = SNESMonitorCancel(snes);CHKERRQ(ierr);
    }
  }

  /* recursive option setting for the smoothers */
  if (fas->next) {ierr = SNESSetFromOptions(fas->next);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESView_FAS"
PetscErrorCode SNESView_FAS(SNES snes, PetscViewer viewer)
{
  SNES_FAS   *fas = (SNES_FAS *) snes->data;
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (fas->level != fas->levels - 1) PetscFunctionReturn(0);
  ierr = PetscTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer, "FAS, levels = %d\n",  fas->levels);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);
    ierr = PetscViewerASCIIPrintf(viewer, "level: %d\n",  fas->level);CHKERRQ(ierr);
    if (fas->upsmooth) {
      ierr = PetscViewerASCIIPrintf(viewer, "up-smoother on level %D:\n",  fas->level);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushTab(viewer);
      ierr = SNESView(fas->upsmooth, viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer, "no up-smoother on level %D\n",  fas->level);CHKERRQ(ierr);
    }
    if (fas->downsmooth) {
      ierr = PetscViewerASCIIPrintf(viewer, "down-smoother on level %D:\n",  fas->level);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushTab(viewer);
      ierr = SNESView(fas->downsmooth, viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer, "no down-smoother on level %D\n",  fas->level);CHKERRQ(ierr);
    }
    if (snes->usegs) {
      ierr = PetscViewerASCIIPrintf(viewer, "Using user Gauss-Seidel on level %D -- smoothdown=%D, smoothup=%D\n",
                                    fas->level, fas->max_down_it, fas->max_up_it);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPopTab(viewer);
  } else {
    SETERRQ1(((PetscObject)snes)->comm,PETSC_ERR_SUP,"Viewer type %s not supported for SNESFAS",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FASDownSmooth"
/*
Defines the action of the downsmoother
 */
PetscErrorCode FASDownSmooth(SNES snes, Vec B, Vec X, Vec F){
  PetscErrorCode      ierr = 0;
  PetscReal           fnorm;
  SNESConvergedReason reason;
  SNES_FAS            *fas = (SNES_FAS *)snes->data;
  PetscInt            k;
  PetscFunctionBegin;
  if (fas->downsmooth) {
    ierr = SNESSolve(fas->downsmooth, B, X);CHKERRQ(ierr);
    /* check convergence reason for the smoother */
    ierr = SNESGetConvergedReason(fas->downsmooth,&reason);CHKERRQ(ierr);
    if (reason < 0 && reason != SNES_DIVERGED_MAX_IT) {
      snes->reason = SNES_DIVERGED_INNER;
      PetscFunctionReturn(0);
    }
  } else if (snes->usegs && snes->ops->computegs) {
    if (fas->monitor) {
      ierr = SNESComputeFunction(snes, X, F);CHKERRQ(ierr);
      ierr = VecNorm(F, NORM_2, &fnorm);CHKERRQ(ierr);
      ierr = PetscViewerASCIIAddTab(fas->monitor,((PetscObject)snes)->tablevel + 2);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(fas->monitor, "%d SNES GS Function norm %14.12e\n", 0, fnorm);CHKERRQ(ierr);
      ierr = PetscViewerASCIISubtractTab(fas->monitor,((PetscObject)snes)->tablevel + 2);CHKERRQ(ierr);
    }
    for (k = 0; k < fas->max_down_it; k++) {
      ierr = SNESComputeGS(snes, B, X);CHKERRQ(ierr);
      if (fas->monitor) {
        ierr = SNESComputeFunction(snes, X, F);CHKERRQ(ierr);
        ierr = VecNorm(F, NORM_2, &fnorm);CHKERRQ(ierr);
        ierr = PetscViewerASCIIAddTab(fas->monitor,((PetscObject)snes)->tablevel + 2);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(fas->monitor, "%d SNES GS Function norm %14.12e\n", k+1, fnorm);CHKERRQ(ierr);
        ierr = PetscViewerASCIISubtractTab(fas->monitor,((PetscObject)snes)->tablevel + 2);CHKERRQ(ierr);
      }
    }
  } else if (snes->pc) {
    ierr = SNESSolve(snes->pc, B, X);CHKERRQ(ierr);
    ierr = SNESGetConvergedReason(fas->downsmooth,&reason);CHKERRQ(ierr);
    if (reason < 0 && reason != SNES_DIVERGED_MAX_IT) {
      snes->reason = SNES_DIVERGED_INNER;
      PetscFunctionReturn(0);
    }
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FASUpSmooth"
/*
Defines the action of the upsmoother
 */
PetscErrorCode FASUpSmooth (SNES snes, Vec B, Vec X, Vec F) {
  PetscErrorCode      ierr = 0;
  PetscReal           fnorm;
  SNESConvergedReason reason;
  SNES_FAS            *fas = (SNES_FAS *)snes->data;
  PetscInt            k;
  PetscFunctionBegin;
  if (fas->upsmooth) {
    ierr = SNESSolve(fas->downsmooth, B, X);CHKERRQ(ierr);
    /* check convergence reason for the smoother */
    ierr = SNESGetConvergedReason(fas->downsmooth,&reason);CHKERRQ(ierr);
    if (reason < 0 && reason != SNES_DIVERGED_MAX_IT) {
      snes->reason = SNES_DIVERGED_INNER;
      PetscFunctionReturn(0);
    }
  } else if (snes->usegs && snes->ops->computegs) {
    if (fas->monitor) {
      ierr = SNESComputeFunction(snes, X, F);CHKERRQ(ierr);
      ierr = VecNorm(F, NORM_2, &fnorm);CHKERRQ(ierr);
      ierr = PetscViewerASCIIAddTab(fas->monitor,((PetscObject)snes)->tablevel + 2);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(fas->monitor, "%d SNES GS Function norm %14.12e\n", 0, fnorm);CHKERRQ(ierr);
      ierr = PetscViewerASCIISubtractTab(fas->monitor,((PetscObject)snes)->tablevel + 2);CHKERRQ(ierr);
    }
    for (k = 0; k < fas->max_up_it; k++) {
      ierr = SNESComputeGS(snes, B, X);CHKERRQ(ierr);
      if (fas->monitor) {
        ierr = SNESComputeFunction(snes, X, F);CHKERRQ(ierr);
        ierr = VecNorm(F, NORM_2, &fnorm);CHKERRQ(ierr);
        ierr = PetscViewerASCIIAddTab(fas->monitor,((PetscObject)snes)->tablevel + 2);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(fas->monitor, "%d SNES GS Function norm %14.12e\n", k+1, fnorm);CHKERRQ(ierr);
        ierr = PetscViewerASCIISubtractTab(fas->monitor,((PetscObject)snes)->tablevel + 2);CHKERRQ(ierr);
      }
    }
  } else if (snes->pc) {
    ierr = SNESSolve(snes->pc, B, X);CHKERRQ(ierr);
    ierr = SNESGetConvergedReason(fas->downsmooth,&reason);CHKERRQ(ierr);
    if (reason < 0 && reason != SNES_DIVERGED_MAX_IT) {
      snes->reason = SNES_DIVERGED_INNER;
      PetscFunctionReturn(0);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FASCoarseCorrection"
/*

Performs the FAS coarse correction as:

fine problem: F(x) = 0
coarse problem: F^c(x) = b^c

b^c = F^c(I^c_fx^f - I^c_fF(x))

with correction:



 */
PetscErrorCode FASCoarseCorrection(SNES snes, Vec X, Vec F, Vec X_new) {
  PetscErrorCode      ierr;
  Vec                 X_c, Xo_c, F_c, B_c;
  SNES_FAS            *fas = (SNES_FAS *)snes->data;
  SNESConvergedReason reason;
  PetscFunctionBegin;
  if (fas->next) {
    ierr = SNESComputeFunction(snes, X, F);CHKERRQ(ierr);

    X_c  = fas->next->vec_sol;
    Xo_c = fas->next->work[0];
    F_c  = fas->next->vec_func;
    B_c  = fas->next->vec_rhs;

    /* inject the solution */
    if (fas->inject) {
      ierr = MatRestrict(fas->inject, X, Xo_c);CHKERRQ(ierr);
    } else {
      ierr = MatRestrict(fas->restrct, X, Xo_c);CHKERRQ(ierr);
      ierr = VecPointwiseMult(Xo_c, fas->rscale, Xo_c);CHKERRQ(ierr);
    }
    ierr = VecScale(F, -1.0);CHKERRQ(ierr);

    /* restrict the defect */
    ierr = MatRestrict(fas->restrct, F, B_c);CHKERRQ(ierr);

    /* solve the coarse problem corresponding to F^c(x^c) = b^c = Rb + F^c(Rx) - RF(x) */
    fas->next->vec_rhs = PETSC_NULL;                                           /*unset the RHS to evaluate function instead of residual*/
    ierr = SNESComputeFunction(fas->next, Xo_c, F_c);CHKERRQ(ierr);

    ierr = VecAXPY(B_c, 1.0, F_c);CHKERRQ(ierr);                               /* add F_c(X) to the RHS */

    /* set initial guess of the coarse problem to the projected fine solution */
    ierr = VecCopy(Xo_c, X_c);CHKERRQ(ierr);

    /* recurse to the next level */
    fas->next->vec_rhs = B_c;
    /* ierr = FASCycle_Private(fas->next, X_c);CHKERRQ(ierr); */
    ierr = SNESSolve(fas->next, B_c, X_c);CHKERRQ(ierr);
    ierr = SNESGetConvergedReason(fas->next,&reason);CHKERRQ(ierr);
    if (reason < 0 && reason != SNES_DIVERGED_MAX_IT) {
      snes->reason = SNES_DIVERGED_INNER;
      PetscFunctionReturn(0);
    }
    /* fas->next->vec_rhs = PETSC_NULL; */

    /* correct as x <- x + I(x^c - Rx)*/
    ierr = VecAXPY(X_c, -1.0, Xo_c);CHKERRQ(ierr);
    ierr = MatInterpolateAdd(fas->interpolate, X_c, X, X_new);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FASCycle_Additive"
/*

The additive cycle looks like:

xhat = x
xhat = dS(x, b)
x = coarsecorrection(xhat, b_d)
x = x + nu*(xhat - x);
(optional) x = uS(x, b)

With the coarse RHS (defect correction) as below.

 */
PetscErrorCode FASCycle_Additive(SNES snes, Vec X) {
  Vec                 F, B, Xhat;
  Vec                 X_c, Xo_c, F_c, B_c, G, W;
  PetscErrorCode      ierr;
  SNES_FAS *          fas = (SNES_FAS *)snes->data;
  SNESConvergedReason reason;
  PetscReal           xnorm = 0., fnorm = 0., gnorm = 0., ynorm = 0.;
  PetscBool           lssucceed;
  PetscFunctionBegin;

  F = snes->vec_func;
  B = snes->vec_rhs;
  Xhat = snes->work[1];
  G    = snes->work[2];
  W    = snes->work[3];
  ierr = VecCopy(X, Xhat);CHKERRQ(ierr);
  /* recurse first */
  if (fas->next) {
    ierr = SNESComputeFunction(snes, Xhat, F);CHKERRQ(ierr);

    X_c  = fas->next->vec_sol;
    Xo_c = fas->next->work[0];
    F_c  = fas->next->vec_func;
    B_c  = fas->next->vec_rhs;

    /* inject the solution */
    if (fas->inject) {
      ierr = MatRestrict(fas->inject, Xhat, Xo_c);CHKERRQ(ierr);
    } else {
      ierr = MatRestrict(fas->restrct, Xhat, Xo_c);CHKERRQ(ierr);
      ierr = VecPointwiseMult(Xo_c, fas->rscale, Xo_c);CHKERRQ(ierr);
    }
    ierr = VecScale(F, -1.0);CHKERRQ(ierr);

    /* restrict the defect */
    ierr = MatRestrict(fas->restrct, F, B_c);CHKERRQ(ierr);

    /* solve the coarse problem corresponding to F^c(x^c) = b^c = Rb + F^c(Rx) - RF(x) */
    fas->next->vec_rhs = PETSC_NULL;                                           /*unset the RHS to evaluate function instead of residual*/
    ierr = SNESComputeFunction(fas->next, Xo_c, F_c);CHKERRQ(ierr);

    ierr = VecAXPY(B_c, 1.0, F_c);CHKERRQ(ierr);                               /* add F_c(X) to the RHS */

    /* set initial guess of the coarse problem to the projected fine solution */
    ierr = VecCopy(Xo_c, X_c);CHKERRQ(ierr);

    /* recurse */
    fas->next->vec_rhs = B_c;
    ierr = SNESSolve(fas->next, B_c, X_c);CHKERRQ(ierr);

    /* smooth on this level */
    ierr = FASDownSmooth(snes, B, X, F);CHKERRQ(ierr);

   ierr = SNESGetConvergedReason(fas->next,&reason);CHKERRQ(ierr);
    if (reason < 0 && reason != SNES_DIVERGED_MAX_IT) {
      snes->reason = SNES_DIVERGED_INNER;
      PetscFunctionReturn(0);
    }

    /* correct as x <- x + I(x^c - Rx)*/
    ierr = VecAXPY(X_c, -1.0, Xo_c);CHKERRQ(ierr);
    ierr = MatInterpolate(fas->interpolate, X_c, Xhat);CHKERRQ(ierr);

    /* additive correction of the coarse direction*/
    ierr = SNESComputeFunction(snes, X, F);CHKERRQ(ierr);
    ierr = VecNorm(F, NORM_2, &fnorm);CHKERRQ(ierr);
    ierr = (*snes->ops->linesearch)(snes,snes->lsP,X,F,Xhat,fnorm,xnorm,G,W,&ynorm,&gnorm,&lssucceed);CHKERRQ(ierr);
    ierr = VecCopy(W, X);CHKERRQ(ierr);
    ierr = VecCopy(G, F);CHKERRQ(ierr);
    fnorm = gnorm;
  } else {
    ierr = FASDownSmooth(snes, B, X, F);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FASCycle_Multiplicative"
/*

Defines the FAS cycle as:

fine problem: F(x) = 0
coarse problem: F^c(x) = b^c

b^c = F^c(I^c_fx^f - I^c_fF(x))

correction:

x = x + I(x^c - Rx)

 */
PetscErrorCode FASCycle_Multiplicative(SNES snes, Vec X) {

  PetscErrorCode      ierr;
  Vec                 F,B;
  SNES_FAS            *fas = (SNES_FAS *)snes->data;

  PetscFunctionBegin;
  F = snes->vec_func;
  B = snes->vec_rhs;
  /* pre-smooth -- just update using the pre-smoother */
  ierr = FASDownSmooth(snes, B, X, F);CHKERRQ(ierr);

  ierr = FASCoarseCorrection(snes, X, F, X);CHKERRQ(ierr);

  if (fas->level != 0) {
    ierr = FASUpSmooth(snes, B, X, F);CHKERRQ(ierr);
  }
  ierr = SNESComputeFunction(snes, X, F);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSolve_FAS"

PetscErrorCode SNESSolve_FAS(SNES snes)
{
  PetscErrorCode ierr;
  PetscInt       i, maxits;
  Vec            X, B, F;
  PetscReal      fnorm;
  SNES_FAS       *fas = (SNES_FAS *)snes->data;
  PetscFunctionBegin;
  maxits = snes->max_its;            /* maximum number of iterations */
  snes->reason = SNES_CONVERGED_ITERATING;
  X = snes->vec_sol;
  B = snes->vec_rhs;
  F = snes->vec_func;

  /*norm setup */
  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  snes->iter = 0;
  snes->norm = 0.;
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);
  if (snes->domainerror) {
    snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN;
    PetscFunctionReturn(0);
  }
  ierr = VecNorm(F, NORM_2, &fnorm);CHKERRQ(ierr); /* fnorm <- ||F||  */
  if (PetscIsInfOrNanReal(fnorm)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"Infinite or not-a-number generated in norm");
  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  snes->norm = fnorm;
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
  SNESLogConvHistory(snes,fnorm,0);
  ierr = SNESMonitor(snes,0,fnorm);CHKERRQ(ierr);

  /* set parameter for default relative tolerance convergence test */
  snes->ttol = fnorm*snes->rtol;
  /* test convergence */
  ierr = (*snes->ops->converged)(snes,0,0.0,0.0,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
  if (snes->reason) PetscFunctionReturn(0);
  for (i = 0; i < maxits; i++) {
    /* Call general purpose update function */

    if (snes->ops->update) {
      ierr = (*snes->ops->update)(snes, snes->iter);CHKERRQ(ierr);
    }
    if (fas->fastype == SNES_FAS_MULTIPLICATIVE) {
      ierr = FASCycle_Multiplicative(snes, X);CHKERRQ(ierr);
    } else {
      ierr = FASCycle_Additive(snes, X);CHKERRQ(ierr);
    }

    /* check for FAS cycle divergence */
    if (snes->reason != SNES_CONVERGED_ITERATING) {
      PetscFunctionReturn(0);
    }
    ierr = VecNorm(F, NORM_2, &fnorm);CHKERRQ(ierr); /* fnorm <- ||F||  */
    /* Monitor convergence */
    ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
    snes->iter = i+1;
    snes->norm = fnorm;
    ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
    SNESLogConvHistory(snes,snes->norm,0);
    ierr = SNESMonitor(snes,snes->iter,snes->norm);CHKERRQ(ierr);
    /* Test for convergence */
    ierr = (*snes->ops->converged)(snes,snes->iter,0.0,0.0,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
    if (snes->reason) break;
  }
  if (i == maxits) {
    ierr = PetscInfo1(snes, "Maximum number of iterations has been reached: %D\n", maxits);CHKERRQ(ierr);
    if (!snes->reason) snes->reason = SNES_DIVERGED_MAX_IT;
  }
  PetscFunctionReturn(0);
}
