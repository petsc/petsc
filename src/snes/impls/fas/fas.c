/* Defines the basic SNES object */
#include <../src/snes/impls/fas/fasimpls.h>    /*I  "petscsnesfas.h"  I*/

const char *SNESFASTypes[] = {"MULTIPLICATIVE","ADDITIVE","SNESFASType","SNES_FAS",0};

extern PetscErrorCode SNESDestroy_FAS(SNES snes);
extern PetscErrorCode SNESSetUp_FAS(SNES snes);
extern PetscErrorCode SNESSetFromOptions_FAS(SNES snes);
extern PetscErrorCode SNESView_FAS(SNES snes, PetscViewer viewer);
extern PetscErrorCode SNESSolve_FAS(SNES snes);
extern PetscErrorCode SNESReset_FAS(SNES snes);
extern PetscErrorCode SNESFASGalerkinDefaultFunction(SNES, Vec, Vec, void *);

/*MC

SNESFAS - Full Approximation Scheme nonlinear multigrid solver.

The nonlinear problem is solved via the repeated application of nonlinear preconditioners and coarse-grid corrections

Level: advanced

.seealso: SNESCreate(), SNES, SNESSetType(), SNESType (for list of available types)
M*/

#undef __FUNCT__
#define __FUNCT__ "SNESCreate_FAS"
PETSC_EXTERN_C PetscErrorCode SNESCreate_FAS(SNES snes)
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

  snes->max_funcs = 30000;
  snes->max_its   = 10000;

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
  fas->linesearch_smooth      = PETSC_NULL;

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESFASGetLevels"
/*@
   SNESFASGetLevels - Gets the number of levels in a FAS, including fine and coarse grids

   Input Parameter:
.  snes - the nonlinear solver context

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
   SNESFASSetCycles - Sets the number of FAS multigrid cycles to use each time a grid is visited.  Use SNESFASSetCyclesOnLevel() for more
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
  if (level > top_level) SETERRQ3(((PetscObject)snes)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Cannot set level %D cycle count from level %D of SNESFAS with %D levels total",level,top_level,fas->levels);
  /* get to the correct level */
  for (i = fas->level; i > level; i--) {
    fas = (SNES_FAS *)fas->next->data;
  }
  if (fas->level != level) SETERRQ(((PetscObject)snes)->comm,PETSC_ERR_PLIB,"SNESFAS level hierarchy corrupt");
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

  if (gsfunc) {
    ierr = SNESSetGS(snes, gsfunc, ctx);CHKERRQ(ierr);
    /* push the provided GS up the tree */
    if (fas->next) ierr = SNESFASSetGS(fas->next, gsfunc, ctx, use_gs);CHKERRQ(ierr);
  } else {
    ierr = SNESGetGS(snes,&gsfunc,&ctx);CHKERRQ(ierr);
    if (gsfunc) {
      /* assume that the user has set the GS solver at this level */
      if (fas->next) ierr = SNESFASSetGS(fas->next, PETSC_NULL, PETSC_NULL, use_gs);CHKERRQ(ierr);
    } else if (use_gs) {
      SETERRQ1(((PetscObject)snes)->comm, PETSC_ERR_ARG_WRONG, "No user Gauss-Seidel function provided in SNESFASSetGS on level %D", fas->level);
    }
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
  if (level > top_level) SETERRQ3(((PetscObject)snes)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Cannot set level %D GS from level %D of SNESFAS with %D levels total",level,top_level,fas->levels);
  /* get to the correct level */
  for (i = fas->level; i > level; i--) {
    fas = (SNES_FAS *)fas->next->data;
    cur_snes = fas->next;
  }
  if (fas->level != level) SETERRQ(((PetscObject)snes)->comm,PETSC_ERR_PLIB,"SNESFAS level hierarchy corrupt");
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
PetscErrorCode SNESFASGetSNES(SNES snes,PetscInt level,SNES *lsnes) {
  SNES_FAS *fas = (SNES_FAS*)snes->data;
  PetscInt i;

  PetscFunctionBegin;
  if (level > fas->levels-1) SETERRQ2(((PetscObject)snes)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Requested level %D from SNESFAS containing %D levels",level,fas->levels);
  if (fas->level < level) SETERRQ2(((PetscObject)snes)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Requested level %D from SNESFAS on level %D, must call from SNES at least as fine as level",level,fas->level);
  *lsnes = snes;
  for (i = fas->level; i > level; i--) {
    *lsnes = fas->next;
    fas = (SNES_FAS *)(*lsnes)->data;
  }
  if (fas->level != level) SETERRQ(((PetscObject)snes)->comm,PETSC_ERR_PLIB,"SNESFAS level hierarchy corrupt");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESFASSetType"
/*@
SNESFASSetType - Sets the update and correction type used for FAS.

Logically Collective

Input Arguments:
snes - nonlinear solver
fastype - SNES_FAS_ADDITIVE or SNES_FAS_MULTIPLICATIVE

.seealso: PCMGSetType()
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

   Logically Collective on SNES

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

   Logically Collective on SNES

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

.seealso: SNESFASSetInjection(), SNESFASSetRestriction(), SNESFASSetRScale()
@*/
PetscErrorCode SNESFASSetInterpolation(SNES snes, PetscInt level, Mat mat) {
  SNES_FAS * fas =  (SNES_FAS *)snes->data;
  PetscInt top_level = fas->level,i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (level > top_level) SETERRQ3(((PetscObject)snes)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Cannot set level %D interpolation from level %D of SNESFAS with %D levels total",level,top_level,fas->levels);
  /* get to the correct level */
  for (i = fas->level; i > level; i--) {
    fas = (SNES_FAS *)fas->next->data;
  }
  if (fas->level != level) SETERRQ(((PetscObject)snes)->comm,PETSC_ERR_PLIB,"SNESFAS level hierarchy corrupt");
  ierr = PetscObjectReference((PetscObject)mat);CHKERRQ(ierr);
  ierr = MatDestroy(&fas->interpolate);CHKERRQ(ierr);
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

         If you do not set this, the transpose of the Mat set with SNESFASSetInterpolation()
    is used.

.keywords: FAS, MG, set, multigrid, restriction, level

.seealso: SNESFASSetInterpolation(), SNESFASSetInjection()
@*/
PetscErrorCode SNESFASSetRestriction(SNES snes, PetscInt level, Mat mat) {
  SNES_FAS * fas =  (SNES_FAS *)snes->data;
  PetscInt top_level = fas->level,i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (level > top_level) SETERRQ3(((PetscObject)snes)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Cannot set level %D restriction from level %D of SNESFAS with %D levels total",level,top_level,fas->levels);
  /* get to the correct level */
  for (i = fas->level; i > level; i--) {
    fas = (SNES_FAS *)fas->next->data;
  }
  if (fas->level != level) SETERRQ(((PetscObject)snes)->comm,PETSC_ERR_PLIB,"SNESFAS level hierarchy corrupt");
  ierr = PetscObjectReference((PetscObject)mat);CHKERRQ(ierr);
  ierr = MatDestroy(&fas->restrct);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (level > top_level) SETERRQ3(((PetscObject)snes)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Cannot set level %D rscale from level %D of SNESFAS with %D levels total",level,top_level,fas->levels);
  /* get to the correct level */
  for (i = fas->level; i > level; i--) {
    fas = (SNES_FAS *)fas->next->data;
  }
  if (fas->level != level) SETERRQ(((PetscObject)snes)->comm,PETSC_ERR_PLIB,"SNESFAS level hierarchy corrupt");
  ierr = PetscObjectReference((PetscObject)rscale);CHKERRQ(ierr);
  ierr = VecDestroy(&fas->rscale);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (level > top_level) SETERRQ3(((PetscObject)snes)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Cannot set level %D injection from level %D of SNESFAS with %D levels total",level,top_level,fas->levels);
  /* get to the correct level */
  for (i = fas->level; i > level; i--) {
    fas = (SNES_FAS *)fas->next->data;
  }
  if (fas->level != level) SETERRQ(((PetscObject)snes)->comm,PETSC_ERR_PLIB,"SNESFAS level hierarchy corrupt");
  ierr = PetscObjectReference((PetscObject)mat);CHKERRQ(ierr);
  ierr = MatDestroy(&fas->inject);CHKERRQ(ierr);
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
  ierr = SNESDestroy(&fas->upsmooth);CHKERRQ(ierr);
  ierr = SNESDestroy(&fas->downsmooth);CHKERRQ(ierr);
  ierr = MatDestroy(&fas->inject);CHKERRQ(ierr);
  ierr = MatDestroy(&fas->interpolate);CHKERRQ(ierr);
  ierr = MatDestroy(&fas->restrct);CHKERRQ(ierr);
  ierr = VecDestroy(&fas->rscale);CHKERRQ(ierr);

  if (fas->upsmooth)   ierr = SNESReset(fas->upsmooth);CHKERRQ(ierr);
  if (fas->downsmooth) ierr = SNESReset(fas->downsmooth);CHKERRQ(ierr);
  if (fas->next)       ierr = SNESReset(fas->next);CHKERRQ(ierr);

  ierr = PetscLineSearchDestroy(&fas->linesearch_smooth);CHKERRQ(ierr);

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
  ierr = SNESReset(snes);CHKERRQ(ierr);
  if (fas->next)         ierr = SNESDestroy(&fas->next);CHKERRQ(ierr);
  ierr = PetscFree(fas);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSetUp_FAS"
PetscErrorCode SNESSetUp_FAS(SNES snes)
{
  SNES_FAS                *fas = (SNES_FAS *) snes->data;
  PetscErrorCode          ierr;
  const char              *optionsprefix;
  VecScatter              injscatter;
  PetscInt                dm_levels;
  Vec                     vec_sol, vec_func, vec_sol_update, vec_rhs; /* preserve these if they're set through the reset */

  PetscFunctionBegin;

  if (fas->usedmfornumberoflevels && (fas->level == fas->levels - 1)) {
    ierr = DMGetRefineLevel(snes->dm,&dm_levels);CHKERRQ(ierr);
    dm_levels++;
    if (dm_levels > fas->levels) {

      /* we don't want the solution and func vectors to be destroyed in the SNESReset when it's called in SNESFASSetLevels_FAS*/
      vec_sol = snes->vec_sol;
      vec_func = snes->vec_func;
      vec_sol_update = snes->vec_sol_update;
      vec_rhs = snes->vec_rhs;
      snes->vec_sol = PETSC_NULL;
      snes->vec_func = PETSC_NULL;
      snes->vec_sol_update = PETSC_NULL;
      snes->vec_rhs = PETSC_NULL;

      /* reset the number of levels */
      ierr = SNESFASSetLevels(snes,dm_levels,PETSC_NULL);CHKERRQ(ierr);
      ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

      snes->vec_sol = vec_sol;
      snes->vec_func = vec_func;
      snes->vec_rhs = vec_rhs;
      snes->vec_sol_update = vec_sol_update;
    }
  }

  if (fas->level != fas->levels - 1) snes->gridsequence = 0; /* no grid sequencing inside the multigrid hierarchy! */

  if (fas->fastype == SNES_FAS_MULTIPLICATIVE) {
    ierr = SNESDefaultGetWork(snes, 4);CHKERRQ(ierr); /* work vectors used for intergrid transfers */
  } else {
    ierr = SNESDefaultGetWork(snes, 4);CHKERRQ(ierr); /* work vectors used for intergrid transfers */
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
        if (!fas->restrct) {
          ierr = PetscObjectReference((PetscObject)fas->interpolate);CHKERRQ(ierr);
          fas->restrct = fas->interpolate;
        }
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
  /*pass the smoother, function, and jacobian up to the next level if it's not user set already */

 if (fas->next) {
    if (fas->galerkin) {
      ierr = SNESSetFunction(fas->next, PETSC_NULL, SNESFASGalerkinDefaultFunction, fas->next);CHKERRQ(ierr);
    }
  }

  if (fas->next) {
    /* gotta set up the solution vector for this to work */
    if (!fas->next->vec_sol) {ierr = SNESFASCreateCoarseVec(snes,&fas->next->vec_sol);CHKERRQ(ierr);}
    if (!fas->next->vec_rhs) {ierr = SNESFASCreateCoarseVec(snes,&fas->next->vec_rhs);CHKERRQ(ierr);}
    ierr = SNESSetUp(fas->next);CHKERRQ(ierr);
  }

  /* setup the pre and post smoothers */
  if (fas->upsmooth) {ierr = SNESSetFromOptions(fas->upsmooth);CHKERRQ(ierr);}
  if (fas->downsmooth) {ierr = SNESSetFromOptions(fas->downsmooth);CHKERRQ(ierr);}

  /* if the pre and post smoothers don't exist, set up line searches in their place */
  ierr = SNESGetOptionsPrefix(snes, &optionsprefix);CHKERRQ(ierr);
  if (!fas->upsmooth || !fas->downsmooth) {
    ierr = PetscLineSearchCreate(((PetscObject)snes)->comm, &fas->linesearch_smooth);CHKERRQ(ierr);
    ierr = PetscLineSearchSetSNES(fas->linesearch_smooth, snes);CHKERRQ(ierr);
    ierr = PetscLineSearchSetType(fas->linesearch_smooth, PETSCLINESEARCHL2);CHKERRQ(ierr);
    ierr = PetscLineSearchAppendOptionsPrefix(fas->linesearch_smooth, "fas_");CHKERRQ(ierr);
    ierr = PetscLineSearchAppendOptionsPrefix(fas->linesearch_smooth, optionsprefix);CHKERRQ(ierr);
    ierr = PetscLineSearchSetFromOptions(fas->linesearch_smooth);CHKERRQ(ierr);
  }

  /* setup FAS work vectors */
  if (fas->galerkin) {
    ierr = VecDuplicate(snes->vec_sol, &fas->Xg);CHKERRQ(ierr);
    ierr = VecDuplicate(snes->vec_sol, &fas->Fg);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSetFromOptions_FAS"
PetscErrorCode SNESSetFromOptions_FAS(SNES snes)
{
  SNES_FAS       *fas = (SNES_FAS *) snes->data;
  PetscInt       levels = 1;
  PetscBool      flg, smoothflg, smoothupflg, smoothdownflg, smoothcoarseflg = PETSC_FALSE, monflg;
  PetscErrorCode ierr;
  char           monfilename[PETSC_MAX_PATH_LEN];
  SNESFASType    fastype;
  const char     *optionsprefix;
  const char     *prefix;
  PetscLineSearch linesearch;

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
  fastype = fas->fastype;
  ierr = PetscOptionsEnum("-snes_fas_type","FAS correction type","SNESFASSetType",SNESFASTypes,(PetscEnum)fastype,(PetscEnum*)&fastype,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = SNESFASSetType(snes, fastype);CHKERRQ(ierr);
  }

  ierr = SNESGetOptionsPrefix(snes, &optionsprefix);CHKERRQ(ierr);

  /* smoother setup options */
  ierr = PetscOptionsHasName(optionsprefix, "-fas_snes_type", &smoothflg);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(optionsprefix, "-fas_up_snes_type", &smoothupflg);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(optionsprefix, "-fas_down_snes_type", &smoothdownflg);CHKERRQ(ierr);
  if (fas->level == 0) {
    ierr = PetscOptionsHasName(optionsprefix, "-fas_coarse_snes_type", &smoothcoarseflg);CHKERRQ(ierr);
  }
  /* options for the number of preconditioning cycles and cycle type */

  ierr = PetscOptionsInt("-snes_fas_cycles","Number of cycles","SNESFASSetCycles",fas->n_cycles,&fas->n_cycles,&flg);CHKERRQ(ierr);

  ierr = PetscOptionsBool("-snes_fas_galerkin", "Form coarse problems with Galerkin","SNESFAS",fas->galerkin,&fas->galerkin,&flg);CHKERRQ(ierr);

  ierr = PetscOptionsString("-snes_fas_monitor","Monitor FAS progress","SNESMonitorSet","stdout",monfilename,PETSC_MAX_PATH_LEN,&monflg);CHKERRQ(ierr);


  ierr = PetscOptionsTail();CHKERRQ(ierr);
  /* setup from the determined types if there is no pointwise procedure or smoother defined */

  if ((!fas->downsmooth) && smoothcoarseflg) {
    ierr = SNESGetOptionsPrefix(snes,&prefix);CHKERRQ(ierr);
    ierr = SNESCreate(((PetscObject)snes)->comm, &fas->downsmooth);CHKERRQ(ierr);
    ierr = SNESSetOptionsPrefix(fas->downsmooth,prefix);CHKERRQ(ierr);
    ierr = SNESAppendOptionsPrefix(fas->downsmooth,"fas_coarse_");CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)fas->downsmooth, (PetscObject)snes, 1);CHKERRQ(ierr);
  }

  if ((!fas->downsmooth) && smoothdownflg) {
    ierr = SNESGetOptionsPrefix(snes,&prefix);CHKERRQ(ierr);
    ierr = SNESCreate(((PetscObject)snes)->comm, &fas->downsmooth);CHKERRQ(ierr);
    ierr = SNESSetOptionsPrefix(fas->downsmooth,prefix);CHKERRQ(ierr);
    ierr = SNESAppendOptionsPrefix(fas->downsmooth,"fas_down_");CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)fas->downsmooth, (PetscObject)snes, 1);CHKERRQ(ierr);
  }

  if ((!fas->upsmooth) && (fas->level != 0) && smoothupflg) {
    ierr = SNESGetOptionsPrefix(snes,&prefix);CHKERRQ(ierr);
    ierr = SNESCreate(((PetscObject)snes)->comm, &fas->upsmooth);CHKERRQ(ierr);
    ierr = SNESSetOptionsPrefix(fas->upsmooth,prefix);CHKERRQ(ierr);
    ierr = SNESAppendOptionsPrefix(fas->upsmooth,"fas_up_");CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)fas->upsmooth, (PetscObject)snes, 1);CHKERRQ(ierr);
  }

  if ((!fas->downsmooth) && smoothflg) {
    ierr = SNESGetOptionsPrefix(snes,&prefix);CHKERRQ(ierr);
    ierr = SNESCreate(((PetscObject)snes)->comm, &fas->downsmooth);CHKERRQ(ierr);
    ierr = SNESSetOptionsPrefix(fas->downsmooth,prefix);CHKERRQ(ierr);
    ierr = SNESAppendOptionsPrefix(fas->downsmooth,"fas_");CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)fas->downsmooth, (PetscObject)snes, 1);CHKERRQ(ierr);
  }

  if ((!fas->upsmooth) && (fas->level != 0) && smoothflg) {
    ierr = SNESGetOptionsPrefix(snes,&prefix);CHKERRQ(ierr);
    ierr = SNESCreate(((PetscObject)snes)->comm, &fas->upsmooth);CHKERRQ(ierr);
    ierr = SNESSetOptionsPrefix(fas->upsmooth,prefix);CHKERRQ(ierr);
    ierr = SNESAppendOptionsPrefix(fas->upsmooth,"fas_");CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)fas->upsmooth, (PetscObject)snes, 1);CHKERRQ(ierr);
  }

  if (fas->upsmooth) {
    ierr = SNESSetTolerances(fas->upsmooth, fas->upsmooth->abstol, fas->upsmooth->rtol, fas->upsmooth->xtol, 1, 1000);CHKERRQ(ierr);
  }

  if (fas->downsmooth) {
    ierr = SNESSetTolerances(fas->downsmooth, fas->downsmooth->abstol, fas->downsmooth->rtol, fas->downsmooth->xtol, 1, 1000);CHKERRQ(ierr);
  }

  if (fas->level != fas->levels - 1) {
    ierr = SNESSetTolerances(snes, snes->abstol, snes->rtol, snes->xtol, fas->n_cycles, 1000);CHKERRQ(ierr);
  }

  /* control the simple Richardson smoother that is default if there's no upsmooth or downsmooth */
  if (!fas->downsmooth) {
    ierr = PetscOptionsInt("-fas_snes_max_it","Down smooth iterations","SNESFASSetCycles",fas->max_down_it,&fas->max_down_it,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-fas_down_snes_max_it","Down smooth iterations","SNESFASSetCycles",fas->max_down_it,&fas->max_down_it,&flg);CHKERRQ(ierr);
    if (fas->level == 0) {
      ierr = PetscOptionsInt("-fas_coarse_snes_max_it","Coarse smooth iterations","SNESFASSetCycles",fas->max_down_it,&fas->max_down_it,&flg);CHKERRQ(ierr);
    }
  }

  if (!fas->upsmooth) {
    ierr = PetscOptionsInt("-fas_snes_max_it","Upsmooth iterations","SNESFASSetCycles",fas->max_up_it,&fas->max_up_it,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-fas_up_snes_max_it","Upsmooth iterations","SNESFASSetCycles",fas->max_up_it,&fas->max_up_it,&flg);CHKERRQ(ierr);
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


  /* set up the default line search for coarse grid corrections */
  if (fas->fastype == SNES_FAS_ADDITIVE) {
    if (!snes->linesearch) {
      ierr = SNESGetPetscLineSearch(snes, &linesearch);CHKERRQ(ierr);
      ierr = PetscLineSearchSetType(linesearch, PETSCLINESEARCHL2);CHKERRQ(ierr);
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
  ierr = PetscTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer, "FAS, levels = %D\n",  fas->levels);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);
    ierr = PetscViewerASCIIPrintf(viewer, "level: %D\n",  fas->level);CHKERRQ(ierr);
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
  Vec                 Y, FPC;
  PetscBool           lssuccess;
  PetscInt            k;
  PetscFunctionBegin;
  Y = snes->work[3];
  if (fas->downsmooth) {
    ierr = SNESSolve(fas->downsmooth, B, X);CHKERRQ(ierr);
    /* check convergence reason for the smoother */
    ierr = SNESGetConvergedReason(fas->downsmooth,&reason);CHKERRQ(ierr);
    if (reason < 0 && reason != SNES_DIVERGED_MAX_IT) {
      snes->reason = SNES_DIVERGED_INNER;
      PetscFunctionReturn(0);
    }
    ierr = SNESGetFunction(fas->downsmooth, &FPC, PETSC_NULL, PETSC_NULL);CHKERRQ(ierr);
    ierr = VecCopy(FPC, F);CHKERRQ(ierr);
  } else {
    /* basic richardson smoother */
    for (k = 0; k < fas->max_up_it; k++) {
      ierr = SNESComputeFunction(snes, X, F);CHKERRQ(ierr);
      ierr = VecNorm(F, NORM_2, &fnorm);CHKERRQ(ierr);
      ierr = VecCopy(F, Y);CHKERRQ(ierr);
      ierr = PetscLineSearchApply(fas->linesearch_smooth,X,F,&fnorm,Y);CHKERRQ(ierr);
      ierr = PetscLineSearchGetSuccess(fas->linesearch_smooth, &lssuccess);CHKERRQ(ierr);
      if (!lssuccess) {
        if (++snes->numFailures >= snes->maxFailures) {
          snes->reason = SNES_DIVERGED_LINE_SEARCH;
          PetscFunctionReturn(0);
        }
      }
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
  Vec                 Y, FPC;
  PetscBool           lssuccess;
  PetscInt            k;
  PetscFunctionBegin;
  Y = snes->work[3];
  if (fas->upsmooth) {
    ierr = SNESSolve(fas->upsmooth, B, X);CHKERRQ(ierr);
    /* check convergence reason for the smoother */
    ierr = SNESGetConvergedReason(fas->upsmooth,&reason);CHKERRQ(ierr);
    if (reason < 0 && reason != SNES_DIVERGED_MAX_IT) {
      snes->reason = SNES_DIVERGED_INNER;
      PetscFunctionReturn(0);
    }
    ierr = SNESGetFunction(fas->upsmooth, &FPC, PETSC_NULL, PETSC_NULL);CHKERRQ(ierr);
    ierr = VecCopy(FPC, F);CHKERRQ(ierr);
  } else {
    /* basic richardson smoother */
    for (k = 0; k < fas->max_up_it; k++) {
      ierr = SNESComputeFunction(snes, X, F);CHKERRQ(ierr);
      ierr = VecNorm(F, NORM_2, &fnorm);CHKERRQ(ierr);
      ierr = VecCopy(F, Y);CHKERRQ(ierr);
      ierr = PetscLineSearchApply(fas->linesearch_smooth,X,F,&fnorm,Y);CHKERRQ(ierr);
      ierr = PetscLineSearchGetSuccess(fas->linesearch_smooth, &lssuccess);CHKERRQ(ierr);
      if (!lssuccess) {
        if (++snes->numFailures >= snes->maxFailures) {
          snes->reason = SNES_DIVERGED_LINE_SEARCH;
          PetscFunctionReturn(0);
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESFASCreateCoarseVec"
/*@
   SNESFASCreateCoarseVec - create Vec corresponding to a state vector on one level coarser than current level

   Collective

   Input Arguments:
.  snes - SNESFAS

   Output Arguments:
.  Xcoarse - vector on level one coarser than snes

   Level: developer

.seealso: SNESFASSetRestriction(), SNESFASRestrict()
@*/
PetscErrorCode SNESFASCreateCoarseVec(SNES snes,Vec *Xcoarse)
{
  PetscErrorCode ierr;
  SNES_FAS       *fas = (SNES_FAS*)snes->data;

  PetscFunctionBegin;
  if (fas->rscale) {ierr = VecDuplicate(fas->rscale,Xcoarse);CHKERRQ(ierr);}
  else if (fas->inject) {ierr = MatGetVecs(fas->inject,Xcoarse,PETSC_NULL);CHKERRQ(ierr);}
  else SETERRQ(((PetscObject)snes)->comm,PETSC_ERR_ARG_WRONGSTATE,"Must set restriction or injection");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESFASRestrict"
/*@
   SNESFASRestrict - restrict a Vec to the next coarser level

   Collective

   Input Arguments:
+  fine - SNES from which to restrict
-  Xfine - vector to restrict

   Output Arguments:
.  Xcoarse - result of restriction

   Level: developer

.seealso: SNESFASSetRestriction(), SNESFASSetInjection()
@*/
PetscErrorCode SNESFASRestrict(SNES fine,Vec Xfine,Vec Xcoarse)
{
  PetscErrorCode ierr;
  SNES_FAS       *fas = (SNES_FAS*)fine->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fine,SNES_CLASSID,1);
  PetscValidHeaderSpecific(Xfine,VEC_CLASSID,2);
  PetscValidHeaderSpecific(Xcoarse,VEC_CLASSID,3);
  if (fas->inject) {
    ierr = MatRestrict(fas->inject,Xfine,Xcoarse);CHKERRQ(ierr);
  } else {
    ierr = MatRestrict(fas->restrct,Xfine,Xcoarse);CHKERRQ(ierr);
    ierr = VecPointwiseMult(Xcoarse,fas->rscale,Xcoarse);CHKERRQ(ierr);
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
    X_c  = fas->next->vec_sol;
    Xo_c = fas->next->work[0];
    F_c  = fas->next->vec_func;
    B_c  = fas->next->vec_rhs;

    ierr = SNESFASRestrict(snes,X,Xo_c);CHKERRQ(ierr);
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
  Vec                 X_c, Xo_c, F_c, B_c;
  PetscErrorCode      ierr;
  SNES_FAS *          fas = (SNES_FAS *)snes->data;
  SNESConvergedReason reason;
  PetscReal           xnorm, fnorm, ynorm;
  PetscBool           lssuccess;
  PetscFunctionBegin;

  F = snes->vec_func;
  B = snes->vec_rhs;
  Xhat = snes->work[3];
  ierr = VecCopy(X, Xhat);CHKERRQ(ierr);
  /* recurse first */
  if (fas->next) {
    ierr = SNESComputeFunction(snes, Xhat, F);CHKERRQ(ierr);

    X_c  = fas->next->vec_sol;
    Xo_c = fas->next->work[0];
    F_c  = fas->next->vec_func;
    B_c  = fas->next->vec_rhs;

    ierr = SNESFASRestrict(snes,Xhat,Xo_c);CHKERRQ(ierr);
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
    ierr = VecAYPX(X_c, -1.0, Xo_c);CHKERRQ(ierr);
    ierr = MatInterpolate(fas->interpolate, X_c, Xhat);CHKERRQ(ierr);

    /* additive correction of the coarse direction*/
    ierr = SNESComputeFunction(snes, X, F);CHKERRQ(ierr);
    ierr = VecNorm(F, NORM_2, &fnorm);CHKERRQ(ierr);
    ierr = PetscLineSearchApply(snes->linesearch, X, F, &fnorm, Xhat);CHKERRQ(ierr);
    ierr = PetscLineSearchGetSuccess(snes->linesearch, &lssuccess);CHKERRQ(ierr);
    if (!lssuccess) {
      if (++snes->numFailures >= snes->maxFailures) {
        snes->reason = SNES_DIVERGED_LINE_SEARCH;
        PetscFunctionReturn(0);
      }
    }
    ierr = PetscLineSearchGetNorms(snes->linesearch, &xnorm, &fnorm, &ynorm);CHKERRQ(ierr);
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
  Vec            X, F;
  PetscReal      fnorm;
  SNES_FAS       *fas = (SNES_FAS *)snes->data,*ffas;
  DM             dm;

  PetscFunctionBegin;
  maxits = snes->max_its;            /* maximum number of iterations */
  snes->reason = SNES_CONVERGED_ITERATING;
  X = snes->vec_sol;
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

  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  for (ffas=fas; ffas->next; ffas=(SNES_FAS*)ffas->next->data) {
    DM dmcoarse;
    ierr = SNESGetDM(ffas->next,&dmcoarse);CHKERRQ(ierr);
    ierr = DMRestrict(dm,ffas->restrct,ffas->rscale,ffas->inject,dmcoarse);CHKERRQ(ierr);
    dm = dmcoarse;
  }

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
