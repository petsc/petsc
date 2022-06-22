#include <../src/tao/constrained/impls/almm/almm.h> /*I "petsctao.h" I*/ /*I "petscvec.h" I*/
#include <petsctao.h>
#include <petsc/private/petscimpl.h>
#include <petsc/private/vecimpl.h>

/*@
   TaoALMMGetType - Retrieve the augmented Lagrangian formulation type for the subproblem.

   Input Parameters:
.  tao - the Tao context for the TAOALMM solver

   Output Parameters:
.  type - augmented Lagragrangian type

   Level: advanced

.seealso: `TAOALMM`, `TaoALMMSetType()`, `TaoALMMType`
@*/
PetscErrorCode TaoALMMGetType(Tao tao, TaoALMMType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidPointer(type, 2);
  PetscUseMethod(tao,"TaoALMMGetType_C",(Tao,TaoALMMType *),(tao,type));
  PetscFunctionReturn(0);
}

PetscErrorCode TaoALMMGetType_Private(Tao tao, TaoALMMType *type)
{
  TAO_ALMM *auglag = (TAO_ALMM*)tao->data;

  PetscFunctionBegin;
  *type = auglag->type;
  PetscFunctionReturn(0);
}

/*@
   TaoALMMSetType - Determine the augmented Lagrangian formulation type for the subproblem.

   Input Parameters:
+  tao - the Tao context for the TAOALMM solver
-  type - augmented Lagragrangian type

   Level: advanced

.seealso: `TAOALMM`, `TaoALMMGetType()`, `TaoALMMType`
@*/
PetscErrorCode TaoALMMSetType(Tao tao, TaoALMMType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscTryMethod(tao,"TaoALMMSetType_C",(Tao,TaoALMMType),(tao,type));
  PetscFunctionReturn(0);
}

PetscErrorCode TaoALMMSetType_Private(Tao tao, TaoALMMType type)
{
  TAO_ALMM *auglag = (TAO_ALMM*)tao->data;

  PetscFunctionBegin;
  PetscCheck(!tao->setupcalled,PetscObjectComm((PetscObject)tao), PETSC_ERR_ORDER, "TaoALMMSetType() must be called before TaoSetUp()");
  auglag->type = type;
  PetscFunctionReturn(0);
}

/*@
   TaoALMMGetSubsolver - Retrieve a pointer to the TAOALMM.

   Input Parameters:
.  tao - the Tao context for the TAOALMM solver

   Output Parameter:
.  subsolver - the Tao context for the subsolver

   Level: advanced

.seealso: `TAOALMM`, `TaoALMMSetSubsolver()`
@*/
PetscErrorCode TaoALMMGetSubsolver(Tao tao, Tao *subsolver)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidPointer(subsolver, 2);
  PetscUseMethod(tao,"TaoALMMGetSubsolver_C",(Tao,Tao *),(tao,subsolver));
  PetscFunctionReturn(0);
}

PetscErrorCode TaoALMMGetSubsolver_Private(Tao tao, Tao *subsolver)
{
  TAO_ALMM *auglag = (TAO_ALMM*)tao->data;

  PetscFunctionBegin;
  *subsolver = auglag->subsolver;
  PetscFunctionReturn(0);
}

/*@
   TaoALMMSetSubsolver - Changes the subsolver inside TAOALMM with the user provided one.

   Input Parameters:
+  tao - the Tao context for the TAOALMM solver
-  subsolver - the Tao context for the subsolver

   Level: advanced

.seealso: `TAOALMM`, `TaoALMMGetSubsolver()`
@*/
PetscErrorCode TaoALMMSetSubsolver(Tao tao, Tao subsolver)
{
   PetscFunctionBegin;
   PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
   PetscValidHeaderSpecific(subsolver, TAO_CLASSID, 2);
   PetscTryMethod(tao,"TaoALMMSetSubsolver_C",(Tao,Tao),(tao,subsolver));
   PetscFunctionReturn(0);
}

PetscErrorCode TaoALMMSetSubsolver_Private(Tao tao, Tao subsolver)
{
  TAO_ALMM       *auglag = (TAO_ALMM*)tao->data;
  PetscBool      compatible;

  PetscFunctionBegin;
  if (subsolver == auglag->subsolver) PetscFunctionReturn(0);
  if (tao->bounded) {
    PetscCall(PetscObjectTypeCompareAny((PetscObject)subsolver, &compatible, TAOSHELL, TAOBNCG, TAOBQNLS, TAOBQNKLS, TAOBQNKTR, TAOBQNKTL, ""));
    PetscCheck(compatible,PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_INCOMP, "Subsolver must be a bound-constrained first-order method");
  } else {
    PetscCall(PetscObjectTypeCompareAny((PetscObject)subsolver, &compatible, TAOSHELL, TAOCG, TAOLMVM, TAOBNCG, TAOBQNLS, TAOBQNKLS, TAOBQNKTR, TAOBQNKTL, ""));
    PetscCheck(compatible,PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_INCOMP, "Subsolver must be a first-order method");
  }
  PetscCheck(compatible,PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_INCOMP, "Subsolver must be a first-order method");
  PetscCall(PetscObjectReference((PetscObject)subsolver));
  PetscCall(TaoDestroy(&auglag->subsolver));
  auglag->subsolver = subsolver;
  if (tao->setupcalled) {
    PetscCall(TaoSetSolution(auglag->subsolver, auglag->P));
    PetscCall(TaoSetObjective(auglag->subsolver, TaoALMMSubsolverObjective_Private, (void*)auglag));
    PetscCall(TaoSetObjectiveAndGradient(auglag->subsolver, NULL, TaoALMMSubsolverObjectiveAndGradient_Private, (void*)auglag));
    PetscCall(TaoSetVariableBounds(auglag->subsolver, auglag->PL, auglag->PU));
  }
  PetscFunctionReturn(0);
}

/*@
   TaoALMMGetMultipliers - Retrieve a pointer to the Lagrange multipliers.

   Input Parameters:
.  tao - the Tao context for the TAOALMM solver

   Output Parameters:
.  Y - vector of Lagrange multipliers

   Level: advanced

   Notes:
   For problems with both equality and inequality constraints,
   the multipliers are combined together as Y = (Ye, Yi). Users
   can recover copies of the subcomponents using index sets
   provided by TaoALMMGetDualIS() and use VecGetSubVector().

.seealso: `TAOALMM`, `TaoALMMSetMultipliers()`, `TaoALMMGetDualIS()`
@*/
PetscErrorCode TaoALMMGetMultipliers(Tao tao, Vec *Y)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidPointer(Y, 2);
  PetscUseMethod(tao,"TaoALMMGetMultipliers_C",(Tao,Vec *),(tao,Y));
  PetscFunctionReturn(0);
}

PetscErrorCode TaoALMMGetMultipliers_Private(Tao tao, Vec *Y)
{
  TAO_ALMM *auglag = (TAO_ALMM*)tao->data;

  PetscFunctionBegin;
  PetscCheck(tao->setupcalled,PetscObjectComm((PetscObject)tao), PETSC_ERR_ORDER, "TaoSetUp() must be called first for scatters to be constructed");
  *Y = auglag->Y;
  PetscFunctionReturn(0);
}

/*@
   TaoALMMSetMultipliers - Set user-defined Lagrange multipliers. The vector type and
                             parallel structure of the given vectormust match equality and
                             inequality constraints. The vector must have a local size equal
                             to the sum of the local sizes for the constraint vectors, and a
                             global size equal to the sum of the global sizes of the constraint
                             vectors.

   Input Parameters:
+  tao - the Tao context for the TAOALMM solver
-  Y - vector of Lagrange multipliers

   Level: advanced

   Notes:
   This routine is only useful if the user wants to change the
   parallel distribution of the combined dual vector in problems that
   feature both equality and inequality constraints. For other tasks,
   it is strongly recommended that the user retreive the dual vector
   created by the solver using TaoALMMGetMultipliers().

.seealso: `TAOALMM`, `TaoALMMGetMultipliers()`
@*/
PetscErrorCode TaoALMMSetMultipliers(Tao tao, Vec Y)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidHeaderSpecific(Y, VEC_CLASSID, 2);
  PetscTryMethod(tao,"TaoALMMSetMultipliers_C",(Tao,Vec),(tao,Y));
  PetscFunctionReturn(0);
}

PetscErrorCode TaoALMMSetMultipliers_Private(Tao tao, Vec Y)
{
  TAO_ALMM       *auglag = (TAO_ALMM*)tao->data;
  VecType        Ytype;
  PetscInt       Nuser, Neq, Nineq, N;
  PetscBool      same = PETSC_FALSE;

  PetscFunctionBegin;
  /* no-op if user provides vector from TaoALMMGetMultipliers() */
  if (Y == auglag->Y) PetscFunctionReturn(0);
  /* make sure vector type is same as equality and inequality constraints */
  if (tao->eq_constrained) {
    PetscCall(VecGetType(tao->constraints_equality, &Ytype));
  } else {
    PetscCall(VecGetType(tao->constraints_inequality, &Ytype));
  }
  PetscCall(PetscObjectTypeCompare((PetscObject)Y, Ytype, &same));
  PetscCheck(same,PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_INCOMP, "Given vector for multipliers is not the same type as constraint vectors");
  /* make sure global size matches sum of equality and inequality */
  if (tao->eq_constrained) {
    PetscCall(VecGetSize(tao->constraints_equality, &Neq));
  } else {
    Neq = 0;
  }
  if (tao->ineq_constrained) {
    PetscCall(VecGetSize(tao->constraints_inequality, &Nineq));
  } else {
    Nineq = 0;
  }
  N = Neq + Nineq;
  PetscCall(VecGetSize(Y, &Nuser));
  PetscCheck(Nuser == N,PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_INCOMP, "Given vector has wrong global size");
  /* if there is only one type of constraint, then we need the local size to match too */
  if (Neq == 0) {
    PetscCall(VecGetLocalSize(tao->constraints_inequality, &Nineq));
    PetscCall(VecGetLocalSize(Y, &Nuser));
    PetscCheck(Nuser == Nineq,PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_INCOMP, "Given vector has wrong local size");
  }
  if (Nineq == 0) {
    PetscCall(VecGetLocalSize(tao->constraints_equality, &Neq));
    PetscCall(VecGetLocalSize(Y, &Nuser));
    PetscCheck(Nuser == Neq,PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_INCOMP, "Given vector has wrong local size");
  }
  /* if we got here, the given vector is compatible so we can replace the current one */
  PetscCall(PetscObjectReference((PetscObject)Y));
  PetscCall(VecDestroy(&auglag->Y));
  auglag->Y = Y;
  /* if there are both types of constraints and the solver has already been set up,
     then we need to regenerate VecScatter objects for the new combined dual vector */
  if (tao->setupcalled && tao->eq_constrained && tao->ineq_constrained) {
    PetscCall(VecDestroy(&auglag->C));
    PetscCall(VecDuplicate(auglag->Y, &auglag->C));
    PetscCall(VecScatterDestroy(&auglag->Yscatter[0]));
    PetscCall(VecScatterCreate(auglag->Y, auglag->Yis[0], auglag->Ye, NULL, &auglag->Yscatter[0]));
    PetscCall(VecScatterDestroy(&auglag->Yscatter[1]));
    PetscCall(VecScatterCreate(auglag->Y, auglag->Yis[1], auglag->Yi, NULL, &auglag->Yscatter[1]));
  }
  PetscFunctionReturn(0);
}

/*@
   TaoALMMGetPrimalIS - Retrieve a pointer to the index set that identifies optimization
                        and slack variable components of the subsolver's solution vector.
                        Not valid for problems with only equality constraints.

   Input Parameter:
.  tao - the Tao context for the TAOALMM solver

   Output Parameters:
+  opt_is - index set associated with the optimization variables (NULL if not needed)
-  slack_is - index set associated with the slack variables (NULL if not needed)

   Level: advanced

.seealso: `TAOALMM`, `TaoALMMGetPrimalVector()`
@*/
PetscErrorCode TaoALMMGetPrimalIS(Tao tao, IS *opt_is, IS *slack_is)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscUseMethod(tao,"TaoALMMGetPrimalIS_C",(Tao,IS *,IS *),(tao,opt_is,slack_is));
  PetscFunctionReturn(0);
}

PetscErrorCode TaoALMMGetPrimalIS_Private(Tao tao, IS *opt_is, IS *slack_is)
{
  TAO_ALMM *auglag = (TAO_ALMM*)tao->data;

  PetscFunctionBegin;
  PetscCheck(tao->ineq_constrained,PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_WRONGSTATE, "Primal space has index sets only for inequality constrained problems");
  PetscCheck(tao->setupcalled,PetscObjectComm((PetscObject)tao), PETSC_ERR_ORDER, "TaoSetUp() must be called first for index sets to be constructed");
  if (opt_is) *opt_is = auglag->Pis[0];
  if (slack_is) *slack_is = auglag->Pis[1];
  PetscFunctionReturn(0);
}

/*@
   TaoALMMGetDualIS - Retrieve a pointer to the index set that identifies equality
                      and inequality constraint components of the dual vector returned
                      by TaoALMMGetMultipliers(). Not valid for problems with only one
                      type of constraint.

   Input Parameter:
.  tao - the Tao context for the TAOALMM solver

   Output Parameters:
+  eq_is - index set associated with the equality constraints (NULL if not needed)
-  ineq_is - index set associated with the inequality constraints (NULL if not needed)

   Level: advanced

.seealso: `TAOALMM`, `TaoALMMGetMultipliers()`
@*/
PetscErrorCode TaoALMMGetDualIS(Tao tao, IS *eq_is, IS *ineq_is)
{
   PetscFunctionBegin;
   PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
   PetscUseMethod(tao,"TaoALMMGetDualIS_C",(Tao,IS *,IS *),(tao,eq_is,ineq_is));
   PetscFunctionReturn(0);
}

PetscErrorCode TaoALMMGetDualIS_Private(Tao tao, IS *eq_is, IS *ineq_is)
{
  TAO_ALMM *auglag = (TAO_ALMM*)tao->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscCheck(tao->ineq_constrained && tao->ineq_constrained,PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_WRONGSTATE, "Dual space has index sets only when problem has both equality and inequality constraints");
  PetscCheck(tao->setupcalled,PetscObjectComm((PetscObject)tao), PETSC_ERR_ORDER, "TaoSetUp() must be called first for index sets to be constructed");
  if (eq_is) *eq_is = auglag->Yis[0];
  if (ineq_is) *ineq_is = auglag->Yis[1];
  PetscFunctionReturn(0);
}
