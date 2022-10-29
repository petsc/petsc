#include <petsc/private/tsimpl.h> /*I "petscts.h"  I*/
#include <petscdraw.h>

PetscLogEvent TS_AdjointStep, TS_ForwardStep, TS_JacobianPEval;

/* #define TSADJOINT_STAGE */

/* ------------------------ Sensitivity Context ---------------------------*/

/*@C
  TSSetRHSJacobianP - Sets the function that computes the Jacobian of G w.r.t. the parameters P where U_t = G(U,P,t), as well as the location to store the matrix.

  Logically Collective on ts

  Input Parameters:
+ ts - `TS` context obtained from `TSCreate()`
. Amat - JacobianP matrix
. func - function
- ctx - [optional] user-defined function context

  Calling sequence of func:
$ func (TS ts,PetscReal t,Vec y,Mat A,void *ctx);
+   t - current timestep
.   U - input vector (current ODE solution)
.   A - output matrix
-   ctx - [optional] user-defined function context

  Level: intermediate

  Note:
    Amat has the same number of rows and the same row parallel layout as u, Amat has the same number of columns and parallel layout as p

.seealso: [](chapter_ts), `TS`, `TSGetRHSJacobianP()`
@*/
PetscErrorCode TSSetRHSJacobianP(TS ts, Mat Amat, PetscErrorCode (*func)(TS, PetscReal, Vec, Mat, void *), void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidHeaderSpecific(Amat, MAT_CLASSID, 2);

  ts->rhsjacobianp    = func;
  ts->rhsjacobianpctx = ctx;
  if (Amat) {
    PetscCall(PetscObjectReference((PetscObject)Amat));
    PetscCall(MatDestroy(&ts->Jacprhs));
    ts->Jacprhs = Amat;
  }
  PetscFunctionReturn(0);
}

/*@C
  TSGetRHSJacobianP - Gets the function that computes the Jacobian of G w.r.t. the parameters P where U_t = G(U,P,t), as well as the location to store the matrix.

  Logically Collective on ts

  Input Parameter:
. ts - `TS` context obtained from `TSCreate()`

  Output Parameters:
+ Amat - JacobianP matrix
. func - function
- ctx - [optional] user-defined function context

  Calling sequence of func:
$ func (TS ts,PetscReal t,Vec y,Mat A,void *ctx);
+   t - current timestep
.   U - input vector (current ODE solution)
.   A - output matrix
-   ctx - [optional] user-defined function context

  Level: intermediate

  Note:
    Amat has the same number of rows and the same row parallel layout as u, Amat has the same number of columns and parallel layout as p

.seealso: [](chapter_ts), `TSSetRHSJacobianP()`, `TS`, `TSGetRHSJacobianP()`
@*/
PetscErrorCode TSGetRHSJacobianP(TS ts, Mat *Amat, PetscErrorCode (**func)(TS, PetscReal, Vec, Mat, void *), void **ctx)
{
  PetscFunctionBegin;
  if (func) *func = ts->rhsjacobianp;
  if (ctx) *ctx = ts->rhsjacobianpctx;
  if (Amat) *Amat = ts->Jacprhs;
  PetscFunctionReturn(0);
}

/*@C
  TSComputeRHSJacobianP - Runs the user-defined JacobianP function.

  Collective on ts

  Input Parameters:
. ts   - The `TS` context obtained from `TSCreate()`

  Level: developer

.seealso: [](chapter_ts), `TSSetRHSJacobianP()`, `TS`
@*/
PetscErrorCode TSComputeRHSJacobianP(TS ts, PetscReal t, Vec U, Mat Amat)
{
  PetscFunctionBegin;
  if (!Amat) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidHeaderSpecific(U, VEC_CLASSID, 3);

  PetscCallBack("TS callback JacobianP for sensitivity analysis", (*ts->rhsjacobianp)(ts, t, U, Amat, ts->rhsjacobianpctx));
  PetscFunctionReturn(0);
}

/*@C
  TSSetIJacobianP - Sets the function that computes the Jacobian of F w.r.t. the parameters P where F(Udot,U,t) = G(U,P,t), as well as the location to store the matrix.

  Logically Collective on ts

  Input Parameters:
+ ts - `TS` context obtained from `TSCreate()`
. Amat - JacobianP matrix
. func - function
- ctx - [optional] user-defined function context

  Calling sequence of func:
$ func (TS ts,PetscReal t,Vec y,Mat A,void *ctx);
+   t - current timestep
.   U - input vector (current ODE solution)
.   Udot - time derivative of state vector
.   shift - shift to apply, see note below
.   A - output matrix
-   ctx - [optional] user-defined function context

  Level: intermediate

  Note:
    Amat has the same number of rows and the same row parallel layout as u, Amat has the same number of columns and parallel layout as p

.seealso: [](chapter_ts), `TSSetRHSJacobianP()`, `TS`
@*/
PetscErrorCode TSSetIJacobianP(TS ts, Mat Amat, PetscErrorCode (*func)(TS, PetscReal, Vec, Vec, PetscReal, Mat, void *), void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidHeaderSpecific(Amat, MAT_CLASSID, 2);

  ts->ijacobianp    = func;
  ts->ijacobianpctx = ctx;
  if (Amat) {
    PetscCall(PetscObjectReference((PetscObject)Amat));
    PetscCall(MatDestroy(&ts->Jacp));
    ts->Jacp = Amat;
  }
  PetscFunctionReturn(0);
}

/*@C
  TSComputeIJacobianP - Runs the user-defined IJacobianP function.

  Collective on ts

  Input Parameters:
+ ts - the `TS` context
. t - current timestep
. U - state vector
. Udot - time derivative of state vector
. shift - shift to apply, see note below
- imex - flag indicates if the method is IMEX so that the RHSJacobian should be kept separate

  Output Parameters:
. A - Jacobian matrix

  Level: developer

.seealso: [](chapter_ts), `TS`, `TSSetIJacobianP()`
@*/
PetscErrorCode TSComputeIJacobianP(TS ts, PetscReal t, Vec U, Vec Udot, PetscReal shift, Mat Amat, PetscBool imex)
{
  PetscFunctionBegin;
  if (!Amat) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidHeaderSpecific(U, VEC_CLASSID, 3);
  PetscValidHeaderSpecific(Udot, VEC_CLASSID, 4);

  PetscCall(PetscLogEventBegin(TS_JacobianPEval, ts, U, Amat, 0));
  if (ts->ijacobianp) PetscCallBack("TS callback JacobianP for sensitivity analysis", (*ts->ijacobianp)(ts, t, U, Udot, shift, Amat, ts->ijacobianpctx));
  if (imex) {
    if (!ts->ijacobianp) { /* system was written as Udot = G(t,U) */
      PetscBool assembled;
      PetscCall(MatZeroEntries(Amat));
      PetscCall(MatAssembled(Amat, &assembled));
      if (!assembled) {
        PetscCall(MatAssemblyBegin(Amat, MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyEnd(Amat, MAT_FINAL_ASSEMBLY));
      }
    }
  } else {
    if (ts->rhsjacobianp) PetscCall(TSComputeRHSJacobianP(ts, t, U, ts->Jacprhs));
    if (ts->Jacprhs == Amat) { /* No IJacobian, so we only have the RHS matrix */
      PetscCall(MatScale(Amat, -1));
    } else if (ts->Jacprhs) { /* Both IJacobian and RHSJacobian */
      MatStructure axpy = DIFFERENT_NONZERO_PATTERN;
      if (!ts->ijacobianp) { /* No IJacobianp provided, but we have a separate RHS matrix */
        PetscCall(MatZeroEntries(Amat));
      }
      PetscCall(MatAXPY(Amat, -1, ts->Jacprhs, axpy));
    }
  }
  PetscCall(PetscLogEventEnd(TS_JacobianPEval, ts, U, Amat, 0));
  PetscFunctionReturn(0);
}

/*@C
    TSSetCostIntegrand - Sets the routine for evaluating the integral term in one or more cost functions

    Logically Collective on ts

    Input Parameters:
+   ts - the `TS` context obtained from `TSCreate()`
.   numcost - number of gradients to be computed, this is the number of cost functions
.   costintegral - vector that stores the integral values
.   rf - routine for evaluating the integrand function
.   drduf - function that computes the gradients of the r's with respect to u
.   drdpf - function that computes the gradients of the r's with respect to p, can be NULL if parametric sensitivity is not desired (mu=NULL)
.   fwd - flag indicating whether to evaluate cost integral in the forward run or the adjoint run
-   ctx - [optional] user-defined context for private data for the function evaluation routine (may be NULL)

    Calling sequence of rf:
$   PetscErrorCode rf(TS ts,PetscReal t,Vec U,Vec F,void *ctx);

    Calling sequence of drduf:
$   PetscErroCode drduf(TS ts,PetscReal t,Vec U,Vec *dRdU,void *ctx);

    Calling sequence of drdpf:
$   PetscErroCode drdpf(TS ts,PetscReal t,Vec U,Vec *dRdP,void *ctx);

    Level: deprecated

    Note:
    For optimization there is usually a single cost function (numcost = 1). For sensitivities there may be multiple cost functions

.seealso: [](chapter_ts), `TS`, `TSSetRHSJacobianP()`, `TSGetCostGradients()`, `TSSetCostGradients()`
@*/
PetscErrorCode TSSetCostIntegrand(TS ts, PetscInt numcost, Vec costintegral, PetscErrorCode (*rf)(TS, PetscReal, Vec, Vec, void *), PetscErrorCode (*drduf)(TS, PetscReal, Vec, Vec *, void *), PetscErrorCode (*drdpf)(TS, PetscReal, Vec, Vec *, void *), PetscBool fwd, void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  if (costintegral) PetscValidHeaderSpecific(costintegral, VEC_CLASSID, 3);
  PetscCheck(!ts->numcost || ts->numcost == numcost, PetscObjectComm((PetscObject)ts), PETSC_ERR_USER, "The number of cost functions (2nd parameter of TSSetCostIntegrand()) is inconsistent with the one set by TSSetCostGradients() or TSForwardSetIntegralGradients()");
  if (!ts->numcost) ts->numcost = numcost;

  if (costintegral) {
    PetscCall(PetscObjectReference((PetscObject)costintegral));
    PetscCall(VecDestroy(&ts->vec_costintegral));
    ts->vec_costintegral = costintegral;
  } else {
    if (!ts->vec_costintegral) { /* Create a seq vec if user does not provide one */
      PetscCall(VecCreateSeq(PETSC_COMM_SELF, numcost, &ts->vec_costintegral));
    } else {
      PetscCall(VecSet(ts->vec_costintegral, 0.0));
    }
  }
  if (!ts->vec_costintegrand) {
    PetscCall(VecDuplicate(ts->vec_costintegral, &ts->vec_costintegrand));
  } else {
    PetscCall(VecSet(ts->vec_costintegrand, 0.0));
  }
  ts->costintegralfwd  = fwd; /* Evaluate the cost integral in forward run if fwd is true */
  ts->costintegrand    = rf;
  ts->costintegrandctx = ctx;
  ts->drdufunction     = drduf;
  ts->drdpfunction     = drdpf;
  PetscFunctionReturn(0);
}

/*@C
   TSGetCostIntegral - Returns the values of the integral term in the cost functions.
   It is valid to call the routine after a backward run.

   Not Collective

   Input Parameter:
.  ts - the `TS` context obtained from `TSCreate()`

   Output Parameter:
.  v - the vector containing the integrals for each cost function

   Level: intermediate

.seealso: [](chapter_ts), `TS`, `TSAdjointSolve()`, ``TSSetCostIntegrand()`
@*/
PetscErrorCode TSGetCostIntegral(TS ts, Vec *v)
{
  TS quadts;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidPointer(v, 2);
  PetscCall(TSGetQuadratureTS(ts, NULL, &quadts));
  *v = quadts->vec_sol;
  PetscFunctionReturn(0);
}

/*@C
   TSComputeCostIntegrand - Evaluates the integral function in the cost functions.

   Input Parameters:
+  ts - the `TS` context
.  t - current time
-  U - state vector, i.e. current solution

   Output Parameter:
.  Q - vector of size numcost to hold the outputs

   Level: deprecated

   Note:
   Most users should not need to explicitly call this routine, as it
   is used internally within the sensitivity analysis context.

.seealso: [](chapter_ts), `TS`, `TSAdjointSolve()`, `TSSetCostIntegrand()`
@*/
PetscErrorCode TSComputeCostIntegrand(TS ts, PetscReal t, Vec U, Vec Q)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidHeaderSpecific(U, VEC_CLASSID, 3);
  PetscValidHeaderSpecific(Q, VEC_CLASSID, 4);

  PetscCall(PetscLogEventBegin(TS_FunctionEval, ts, U, Q, 0));
  if (ts->costintegrand) PetscCallBack("TS callback integrand in the cost function", (*ts->costintegrand)(ts, t, U, Q, ts->costintegrandctx));
  else PetscCall(VecZeroEntries(Q));
  PetscCall(PetscLogEventEnd(TS_FunctionEval, ts, U, Q, 0));
  PetscFunctionReturn(0);
}

/*@C
  TSComputeDRDUFunction - Deprecated, use `TSGetQuadratureTS()` then `TSComputeRHSJacobian()`

  Level: deprecated

@*/
PetscErrorCode TSComputeDRDUFunction(TS ts, PetscReal t, Vec U, Vec *DRDU)
{
  PetscFunctionBegin;
  if (!DRDU) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidHeaderSpecific(U, VEC_CLASSID, 3);

  PetscCallBack("TS callback DRDU for sensitivity analysis", (*ts->drdufunction)(ts, t, U, DRDU, ts->costintegrandctx));
  PetscFunctionReturn(0);
}

/*@C
  TSComputeDRDPFunction - Deprecated, use `TSGetQuadratureTS()` then `TSComputeRHSJacobianP()`

  Level: deprecated

@*/
PetscErrorCode TSComputeDRDPFunction(TS ts, PetscReal t, Vec U, Vec *DRDP)
{
  PetscFunctionBegin;
  if (!DRDP) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidHeaderSpecific(U, VEC_CLASSID, 3);

  PetscCallBack("TS callback DRDP for sensitivity analysis", (*ts->drdpfunction)(ts, t, U, DRDP, ts->costintegrandctx));
  PetscFunctionReturn(0);
}

/*@C
  TSSetIHessianProduct - Sets the function that computes the vector-Hessian-vector product. The Hessian is the second-order derivative of F (IFunction) w.r.t. the state variable.

  Logically Collective on ts

  Input Parameters:
+ ts - `TS` context obtained from `TSCreate()`
. ihp1 - an array of vectors storing the result of vector-Hessian-vector product for F_UU
. hessianproductfunc1 - vector-Hessian-vector product function for F_UU
. ihp2 - an array of vectors storing the result of vector-Hessian-vector product for F_UP
. hessianproductfunc2 - vector-Hessian-vector product function for F_UP
. ihp3 - an array of vectors storing the result of vector-Hessian-vector product for F_PU
. hessianproductfunc3 - vector-Hessian-vector product function for F_PU
. ihp4 - an array of vectors storing the result of vector-Hessian-vector product for F_PP
- hessianproductfunc4 - vector-Hessian-vector product function for F_PP

  Calling sequence of ihessianproductfunc:
$ ihessianproductfunc (TS ts,PetscReal t,Vec U,Vec *Vl,Vec Vr,Vec *VHV,void *ctx);
+   t - current timestep
.   U - input vector (current ODE solution)
.   Vl - an array of input vectors to be left-multiplied with the Hessian
.   Vr - input vector to be right-multiplied with the Hessian
.   VHV - an array of output vectors for vector-Hessian-vector product
-   ctx - [optional] user-defined function context

  Level: intermediate

  Notes:
  The first Hessian function and the working array are required.
  As an example to implement the callback functions, the second callback function calculates the vector-Hessian-vector product
  $ Vl_n^T*F_UP*Vr
  where the vector Vl_n (n-th element in the array Vl) and Vr are of size N and M respectively, and the Hessian F_UP is of size N x N x M.
  Each entry of F_UP corresponds to the derivative
  $ F_UP[i][j][k] = \frac{\partial^2 F[i]}{\partial U[j] \partial P[k]}.
  The result of the vector-Hessian-vector product for Vl_n needs to be stored in vector VHV_n with the j-th entry being
  $ VHV_n[j] = \sum_i \sum_k {Vl_n[i] * F_UP[i][j][k] * Vr[k]}
  If the cost function is a scalar, there will be only one vector in Vl and VHV.

.seealso: [](chapter_ts), `TS`
@*/
PetscErrorCode TSSetIHessianProduct(TS ts, Vec *ihp1, PetscErrorCode (*ihessianproductfunc1)(TS, PetscReal, Vec, Vec *, Vec, Vec *, void *), Vec *ihp2, PetscErrorCode (*ihessianproductfunc2)(TS, PetscReal, Vec, Vec *, Vec, Vec *, void *), Vec *ihp3, PetscErrorCode (*ihessianproductfunc3)(TS, PetscReal, Vec, Vec *, Vec, Vec *, void *), Vec *ihp4, PetscErrorCode (*ihessianproductfunc4)(TS, PetscReal, Vec, Vec *, Vec, Vec *, void *), void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidPointer(ihp1, 2);

  ts->ihessianproductctx = ctx;
  if (ihp1) ts->vecs_fuu = ihp1;
  if (ihp2) ts->vecs_fup = ihp2;
  if (ihp3) ts->vecs_fpu = ihp3;
  if (ihp4) ts->vecs_fpp = ihp4;
  ts->ihessianproduct_fuu = ihessianproductfunc1;
  ts->ihessianproduct_fup = ihessianproductfunc2;
  ts->ihessianproduct_fpu = ihessianproductfunc3;
  ts->ihessianproduct_fpp = ihessianproductfunc4;
  PetscFunctionReturn(0);
}

/*@C
  TSComputeIHessianProductFunctionUU - Runs the user-defined vector-Hessian-vector product function for Fuu.

  Collective on ts

  Input Parameters:
. ts   - The `TS` context obtained from `TSCreate()`

  Level: developer

  Note:
  `TSComputeIHessianProductFunctionUU()` is typically used for sensitivity implementation,
  so most users would not generally call this routine themselves.

.seealso: [](chapter_ts), `TSSetIHessianProduct()`
@*/
PetscErrorCode TSComputeIHessianProductFunctionUU(TS ts, PetscReal t, Vec U, Vec *Vl, Vec Vr, Vec *VHV)
{
  PetscFunctionBegin;
  if (!VHV) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidHeaderSpecific(U, VEC_CLASSID, 3);

  if (ts->ihessianproduct_fuu) PetscCallBack("TS callback IHessianProduct 1 for sensitivity analysis", (*ts->ihessianproduct_fuu)(ts, t, U, Vl, Vr, VHV, ts->ihessianproductctx));

  /* does not consider IMEX for now, so either IHessian or RHSHessian will be calculated, using the same output VHV */
  if (ts->rhshessianproduct_guu) {
    PetscInt nadj;
    PetscCall(TSComputeRHSHessianProductFunctionUU(ts, t, U, Vl, Vr, VHV));
    for (nadj = 0; nadj < ts->numcost; nadj++) PetscCall(VecScale(VHV[nadj], -1));
  }
  PetscFunctionReturn(0);
}

/*@C
  TSComputeIHessianProductFunctionUP - Runs the user-defined vector-Hessian-vector product function for Fup.

  Collective on ts

  Input Parameters:
. ts   - The `TS` context obtained from `TSCreate()`

  Level: developer

  Note:
  `TSComputeIHessianProductFunctionUP()` is typically used for sensitivity implementation,
  so most users would not generally call this routine themselves.

.seealso: [](chapter_ts), `TSSetIHessianProduct()`
@*/
PetscErrorCode TSComputeIHessianProductFunctionUP(TS ts, PetscReal t, Vec U, Vec *Vl, Vec Vr, Vec *VHV)
{
  PetscFunctionBegin;
  if (!VHV) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidHeaderSpecific(U, VEC_CLASSID, 3);

  if (ts->ihessianproduct_fup) PetscCallBack("TS callback IHessianProduct 2 for sensitivity analysis", (*ts->ihessianproduct_fup)(ts, t, U, Vl, Vr, VHV, ts->ihessianproductctx));

  /* does not consider IMEX for now, so either IHessian or RHSHessian will be calculated, using the same output VHV */
  if (ts->rhshessianproduct_gup) {
    PetscInt nadj;
    PetscCall(TSComputeRHSHessianProductFunctionUP(ts, t, U, Vl, Vr, VHV));
    for (nadj = 0; nadj < ts->numcost; nadj++) PetscCall(VecScale(VHV[nadj], -1));
  }
  PetscFunctionReturn(0);
}

/*@C
  TSComputeIHessianProductFunctionPU - Runs the user-defined vector-Hessian-vector product function for Fpu.

  Collective on ts

  Input Parameters:
. ts   - The `TS` context obtained from `TSCreate()`

  Level: developer

  Note:
  `TSComputeIHessianProductFunctionPU()` is typically used for sensitivity implementation,
  so most users would not generally call this routine themselves.

.seealso: [](chapter_ts), `TSSetIHessianProduct()`
@*/
PetscErrorCode TSComputeIHessianProductFunctionPU(TS ts, PetscReal t, Vec U, Vec *Vl, Vec Vr, Vec *VHV)
{
  PetscFunctionBegin;
  if (!VHV) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidHeaderSpecific(U, VEC_CLASSID, 3);

  if (ts->ihessianproduct_fpu) PetscCallBack("TS callback IHessianProduct 3 for sensitivity analysis", (*ts->ihessianproduct_fpu)(ts, t, U, Vl, Vr, VHV, ts->ihessianproductctx));

  /* does not consider IMEX for now, so either IHessian or RHSHessian will be calculated, using the same output VHV */
  if (ts->rhshessianproduct_gpu) {
    PetscInt nadj;
    PetscCall(TSComputeRHSHessianProductFunctionPU(ts, t, U, Vl, Vr, VHV));
    for (nadj = 0; nadj < ts->numcost; nadj++) PetscCall(VecScale(VHV[nadj], -1));
  }
  PetscFunctionReturn(0);
}

/*@C
  TSComputeIHessianProductFunctionPP - Runs the user-defined vector-Hessian-vector product function for Fpp.

  Collective on ts

  Input Parameters:
. ts   - The `TS` context obtained from `TSCreate()`

  Level: developer

  Note:
  `TSComputeIHessianProductFunctionPP()` is typically used for sensitivity implementation,
  so most users would not generally call this routine themselves.

.seealso: [](chapter_ts), `TSSetIHessianProduct()`
@*/
PetscErrorCode TSComputeIHessianProductFunctionPP(TS ts, PetscReal t, Vec U, Vec *Vl, Vec Vr, Vec *VHV)
{
  PetscFunctionBegin;
  if (!VHV) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidHeaderSpecific(U, VEC_CLASSID, 3);

  if (ts->ihessianproduct_fpp) PetscCallBack("TS callback IHessianProduct 3 for sensitivity analysis", (*ts->ihessianproduct_fpp)(ts, t, U, Vl, Vr, VHV, ts->ihessianproductctx));

  /* does not consider IMEX for now, so either IHessian or RHSHessian will be calculated, using the same output VHV */
  if (ts->rhshessianproduct_gpp) {
    PetscInt nadj;
    PetscCall(TSComputeRHSHessianProductFunctionPP(ts, t, U, Vl, Vr, VHV));
    for (nadj = 0; nadj < ts->numcost; nadj++) PetscCall(VecScale(VHV[nadj], -1));
  }
  PetscFunctionReturn(0);
}

/*@C
  TSSetRHSHessianProduct - Sets the function that computes the vector-Hessian-vector product. The Hessian is the second-order derivative of G (RHSFunction) w.r.t. the state variable.

  Logically Collective on ts

  Input Parameters:
+ ts - `TS` context obtained from `TSCreate()`
. rhshp1 - an array of vectors storing the result of vector-Hessian-vector product for G_UU
. hessianproductfunc1 - vector-Hessian-vector product function for G_UU
. rhshp2 - an array of vectors storing the result of vector-Hessian-vector product for G_UP
. hessianproductfunc2 - vector-Hessian-vector product function for G_UP
. rhshp3 - an array of vectors storing the result of vector-Hessian-vector product for G_PU
. hessianproductfunc3 - vector-Hessian-vector product function for G_PU
. rhshp4 - an array of vectors storing the result of vector-Hessian-vector product for G_PP
- hessianproductfunc4 - vector-Hessian-vector product function for G_PP

  Calling sequence of ihessianproductfunc:
$ rhshessianproductfunc (TS ts,PetscReal t,Vec U,Vec *Vl,Vec Vr,Vec *VHV,void *ctx);
+   t - current timestep
.   U - input vector (current ODE solution)
.   Vl - an array of input vectors to be left-multiplied with the Hessian
.   Vr - input vector to be right-multiplied with the Hessian
.   VHV - an array of output vectors for vector-Hessian-vector product
-   ctx - [optional] user-defined function context

  Level: intermediate

  Notes:
  The first Hessian function and the working array are required.
  As an example to implement the callback functions, the second callback function calculates the vector-Hessian-vector product
  $ Vl_n^T*G_UP*Vr
  where the vector Vl_n (n-th element in the array Vl) and Vr are of size N and M respectively, and the Hessian G_UP is of size N x N x M.
  Each entry of G_UP corresponds to the derivative
  $ G_UP[i][j][k] = \frac{\partial^2 G[i]}{\partial U[j] \partial P[k]}.
  The result of the vector-Hessian-vector product for Vl_n needs to be stored in vector VHV_n with j-th entry being
  $ VHV_n[j] = \sum_i \sum_k {Vl_n[i] * G_UP[i][j][k] * Vr[k]}
  If the cost function is a scalar, there will be only one vector in Vl and VHV.

.seealso: `TS`
@*/
PetscErrorCode TSSetRHSHessianProduct(TS ts, Vec *rhshp1, PetscErrorCode (*rhshessianproductfunc1)(TS, PetscReal, Vec, Vec *, Vec, Vec *, void *), Vec *rhshp2, PetscErrorCode (*rhshessianproductfunc2)(TS, PetscReal, Vec, Vec *, Vec, Vec *, void *), Vec *rhshp3, PetscErrorCode (*rhshessianproductfunc3)(TS, PetscReal, Vec, Vec *, Vec, Vec *, void *), Vec *rhshp4, PetscErrorCode (*rhshessianproductfunc4)(TS, PetscReal, Vec, Vec *, Vec, Vec *, void *), void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidPointer(rhshp1, 2);

  ts->rhshessianproductctx = ctx;
  if (rhshp1) ts->vecs_guu = rhshp1;
  if (rhshp2) ts->vecs_gup = rhshp2;
  if (rhshp3) ts->vecs_gpu = rhshp3;
  if (rhshp4) ts->vecs_gpp = rhshp4;
  ts->rhshessianproduct_guu = rhshessianproductfunc1;
  ts->rhshessianproduct_gup = rhshessianproductfunc2;
  ts->rhshessianproduct_gpu = rhshessianproductfunc3;
  ts->rhshessianproduct_gpp = rhshessianproductfunc4;
  PetscFunctionReturn(0);
}

/*@C
  TSComputeRHSHessianProductFunctionUU - Runs the user-defined vector-Hessian-vector product function for Guu.

  Collective on ts

  Input Parameters:
. ts   - The `TS` context obtained from `TSCreate()`

  Level: developer

  Note:
  `TSComputeRHSHessianProductFunctionUU()` is typically used for sensitivity implementation,
  so most users would not generally call this routine themselves.

.seealso: [](chapter_ts), `TS`, `TSSetRHSHessianProduct()`
@*/
PetscErrorCode TSComputeRHSHessianProductFunctionUU(TS ts, PetscReal t, Vec U, Vec *Vl, Vec Vr, Vec *VHV)
{
  PetscFunctionBegin;
  if (!VHV) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidHeaderSpecific(U, VEC_CLASSID, 3);

  PetscCallBack("TS callback RHSHessianProduct 1 for sensitivity analysis", (*ts->rhshessianproduct_guu)(ts, t, U, Vl, Vr, VHV, ts->rhshessianproductctx));
  PetscFunctionReturn(0);
}

/*@C
  TSComputeRHSHessianProductFunctionUP - Runs the user-defined vector-Hessian-vector product function for Gup.

  Collective on ts

  Input Parameters:
. ts   - The `TS` context obtained from `TSCreate()`

  Level: developer

  Note:
  `TSComputeRHSHessianProductFunctionUP()` is typically used for sensitivity implementation,
  so most users would not generally call this routine themselves.

.seealso: [](chapter_ts), `TS`, `TSSetRHSHessianProduct()`
@*/
PetscErrorCode TSComputeRHSHessianProductFunctionUP(TS ts, PetscReal t, Vec U, Vec *Vl, Vec Vr, Vec *VHV)
{
  PetscFunctionBegin;
  if (!VHV) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidHeaderSpecific(U, VEC_CLASSID, 3);

  PetscCallBack("TS callback RHSHessianProduct 2 for sensitivity analysis", (*ts->rhshessianproduct_gup)(ts, t, U, Vl, Vr, VHV, ts->rhshessianproductctx));
  PetscFunctionReturn(0);
}

/*@C
  TSComputeRHSHessianProductFunctionPU - Runs the user-defined vector-Hessian-vector product function for Gpu.

  Collective on ts

  Input Parameters:
. ts   - The `TS` context obtained from `TSCreate()`

  Level: developer

  Note:
  `TSComputeRHSHessianProductFunctionPU()` is typically used for sensitivity implementation,
  so most users would not generally call this routine themselves.

.seealso: [](chapter_ts), `TSSetRHSHessianProduct()`
@*/
PetscErrorCode TSComputeRHSHessianProductFunctionPU(TS ts, PetscReal t, Vec U, Vec *Vl, Vec Vr, Vec *VHV)
{
  PetscFunctionBegin;
  if (!VHV) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidHeaderSpecific(U, VEC_CLASSID, 3);

  PetscCallBack("TS callback RHSHessianProduct 3 for sensitivity analysis", (*ts->rhshessianproduct_gpu)(ts, t, U, Vl, Vr, VHV, ts->rhshessianproductctx));
  PetscFunctionReturn(0);
}

/*@C
  TSComputeRHSHessianProductFunctionPP - Runs the user-defined vector-Hessian-vector product function for Gpp.

  Collective on ts

  Input Parameters:
. ts   - The `TS` context obtained from `TSCreate()`

  Level: developer

  Note:
  `TSComputeRHSHessianProductFunctionPP()` is typically used for sensitivity implementation,
  so most users would not generally call this routine themselves.

.seealso: [](chapter_ts), `TSSetRHSHessianProduct()`
@*/
PetscErrorCode TSComputeRHSHessianProductFunctionPP(TS ts, PetscReal t, Vec U, Vec *Vl, Vec Vr, Vec *VHV)
{
  PetscFunctionBegin;
  if (!VHV) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidHeaderSpecific(U, VEC_CLASSID, 3);

  PetscCallBack("TS callback RHSHessianProduct 3 for sensitivity analysis", (*ts->rhshessianproduct_gpp)(ts, t, U, Vl, Vr, VHV, ts->rhshessianproductctx));
  PetscFunctionReturn(0);
}

/* --------------------------- Adjoint sensitivity ---------------------------*/

/*@
   TSSetCostGradients - Sets the initial value of the gradients of the cost function w.r.t. initial values and w.r.t. the problem parameters
      for use by the `TS` adjoint routines.

   Logically Collective on ts

   Input Parameters:
+  ts - the `TS` context obtained from `TSCreate()`
.  numcost - number of gradients to be computed, this is the number of cost functions
.  lambda - gradients with respect to the initial condition variables, the dimension and parallel layout of these vectors is the same as the ODE solution vector
-  mu - gradients with respect to the parameters, the number of entries in these vectors is the same as the number of parameters

   Level: beginner

   Notes:
    the entries in these vectors must be correctly initialized with the values lamda_i = df/dy|finaltime  mu_i = df/dp|finaltime

   After `TSAdjointSolve()` is called the lamba and the mu contain the computed sensitivities

.seealso: `TS`, `TSAdjointSolve()`, `TSGetCostGradients()`
@*/
PetscErrorCode TSSetCostGradients(TS ts, PetscInt numcost, Vec *lambda, Vec *mu)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidPointer(lambda, 3);
  ts->vecs_sensi  = lambda;
  ts->vecs_sensip = mu;
  PetscCheck(!ts->numcost || ts->numcost == numcost, PetscObjectComm((PetscObject)ts), PETSC_ERR_USER, "The number of cost functions (2nd parameter of TSSetCostIntegrand()) is inconsistent with the one set by TSSetCostIntegrand");
  ts->numcost = numcost;
  PetscFunctionReturn(0);
}

/*@
   TSGetCostGradients - Returns the gradients from the `TSAdjointSolve()`

   Not Collective, but the vectors returned are parallel if `TS` is parallel

   Input Parameter:
.  ts - the `TS` context obtained from `TSCreate()`

   Output Parameters:
+  numcost - size of returned arrays
.  lambda - vectors containing the gradients of the cost functions with respect to the ODE/DAE solution variables
-  mu - vectors containing the gradients of the cost functions with respect to the problem parameters

   Level: intermediate

.seealso: [](chapter_ts), `TS`, `TSAdjointSolve()`, `TSSetCostGradients()`
@*/
PetscErrorCode TSGetCostGradients(TS ts, PetscInt *numcost, Vec **lambda, Vec **mu)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  if (numcost) *numcost = ts->numcost;
  if (lambda) *lambda = ts->vecs_sensi;
  if (mu) *mu = ts->vecs_sensip;
  PetscFunctionReturn(0);
}

/*@
   TSSetCostHessianProducts - Sets the initial value of the Hessian-vector products of the cost function w.r.t. initial values and w.r.t. the problem parameters
   for use by the `TS` adjoint routines.

   Logically Collective on ts

   Input Parameters:
+  ts - the `TS` context obtained from `TSCreate()`
.  numcost - number of cost functions
.  lambda2 - Hessian-vector product with respect to the initial condition variables, the dimension and parallel layout of these vectors is the same as the ODE solution vector
.  mu2 - Hessian-vector product with respect to the parameters, the number of entries in these vectors is the same as the number of parameters
-  dir - the direction vector that are multiplied with the Hessian of the cost functions

   Level: beginner

   Notes:
   Hessian of the cost function is completely different from Hessian of the ODE/DAE system

   For second-order adjoint, one needs to call this function and then `TSAdjointSetForward()` before `TSSolve()`.

   After `TSAdjointSolve()` is called, the lamba2 and the mu2 will contain the computed second-order adjoint sensitivities, and can be used to produce Hessian-vector product (not the full Hessian matrix). Users must provide a direction vector; it is usually generated by an optimization solver.

   Passing NULL for lambda2 disables the second-order calculation.

.seealso: [](chapter_ts), `TS`, `TSAdjointSolve()`, `TSAdjointSetForward()`
@*/
PetscErrorCode TSSetCostHessianProducts(TS ts, PetscInt numcost, Vec *lambda2, Vec *mu2, Vec dir)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscCheck(!ts->numcost || ts->numcost == numcost, PetscObjectComm((PetscObject)ts), PETSC_ERR_USER, "The number of cost functions (2nd parameter of TSSetCostIntegrand()) is inconsistent with the one set by TSSetCostIntegrand");
  ts->numcost      = numcost;
  ts->vecs_sensi2  = lambda2;
  ts->vecs_sensi2p = mu2;
  ts->vec_dir      = dir;
  PetscFunctionReturn(0);
}

/*@
   TSGetCostHessianProducts - Returns the gradients from the `TSAdjointSolve()`

   Not Collective, but vectors returned are parallel if `TS` is parallel

   Input Parameter:
.  ts - the `TS` context obtained from `TSCreate()`

   Output Parameters:
+  numcost - number of cost functions
.  lambda2 - Hessian-vector product with respect to the initial condition variables, the dimension and parallel layout of these vectors is the same as the ODE solution vector
.  mu2 - Hessian-vector product with respect to the parameters, the number of entries in these vectors is the same as the number of parameters
-  dir - the direction vector that are multiplied with the Hessian of the cost functions

   Level: intermediate

.seealso: [](chapter_ts), `TSAdjointSolve()`, `TSSetCostHessianProducts()`
@*/
PetscErrorCode TSGetCostHessianProducts(TS ts, PetscInt *numcost, Vec **lambda2, Vec **mu2, Vec *dir)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  if (numcost) *numcost = ts->numcost;
  if (lambda2) *lambda2 = ts->vecs_sensi2;
  if (mu2) *mu2 = ts->vecs_sensi2p;
  if (dir) *dir = ts->vec_dir;
  PetscFunctionReturn(0);
}

/*@
  TSAdjointSetForward - Trigger the tangent linear solver and initialize the forward sensitivities

  Logically Collective on ts

  Input Parameters:
+  ts - the `TS` context obtained from `TSCreate()`
-  didp - the derivative of initial values w.r.t. parameters

  Level: intermediate

  Notes:
  When computing sensitivies w.r.t. initial condition, set didp to NULL so that the solver will take it as an identity matrix mathematically.
  `TS` adjoint does not reset the tangent linear solver automatically, `TSAdjointResetForward()` should be called to reset the tangent linear solver.

.seealso: [](chapter_ts), `TSAdjointSolve()`, `TSSetCostHessianProducts()`, `TSAdjointResetForward()`
@*/
PetscErrorCode TSAdjointSetForward(TS ts, Mat didp)
{
  Mat          A;
  Vec          sp;
  PetscScalar *xarr;
  PetscInt     lsize;

  PetscFunctionBegin;
  ts->forward_solve = PETSC_TRUE; /* turn on tangent linear mode */
  PetscCheck(ts->vecs_sensi2, PetscObjectComm((PetscObject)ts), PETSC_ERR_USER, "Must call TSSetCostHessianProducts() first");
  PetscCheck(ts->vec_dir, PetscObjectComm((PetscObject)ts), PETSC_ERR_USER, "Directional vector is missing. Call TSSetCostHessianProducts() to set it.");
  /* create a single-column dense matrix */
  PetscCall(VecGetLocalSize(ts->vec_sol, &lsize));
  PetscCall(MatCreateDense(PetscObjectComm((PetscObject)ts), lsize, PETSC_DECIDE, PETSC_DECIDE, 1, NULL, &A));

  PetscCall(VecDuplicate(ts->vec_sol, &sp));
  PetscCall(MatDenseGetColumn(A, 0, &xarr));
  PetscCall(VecPlaceArray(sp, xarr));
  if (ts->vecs_sensi2p) { /* tangent linear variable initialized as 2*dIdP*dir */
    if (didp) {
      PetscCall(MatMult(didp, ts->vec_dir, sp));
      PetscCall(VecScale(sp, 2.));
    } else {
      PetscCall(VecZeroEntries(sp));
    }
  } else { /* tangent linear variable initialized as dir */
    PetscCall(VecCopy(ts->vec_dir, sp));
  }
  PetscCall(VecResetArray(sp));
  PetscCall(MatDenseRestoreColumn(A, &xarr));
  PetscCall(VecDestroy(&sp));

  PetscCall(TSForwardSetInitialSensitivities(ts, A)); /* if didp is NULL, identity matrix is assumed */

  PetscCall(MatDestroy(&A));
  PetscFunctionReturn(0);
}

/*@
  TSAdjointResetForward - Reset the tangent linear solver and destroy the tangent linear context

  Logically Collective on ts

  Input Parameters:
.  ts - the `TS` context obtained from `TSCreate()`

  Level: intermediate

.seealso: [](chapter_ts), `TSAdjointSetForward()`
@*/
PetscErrorCode TSAdjointResetForward(TS ts)
{
  PetscFunctionBegin;
  ts->forward_solve = PETSC_FALSE; /* turn off tangent linear mode */
  PetscCall(TSForwardReset(ts));
  PetscFunctionReturn(0);
}

/*@
   TSAdjointSetUp - Sets up the internal data structures for the later use
   of an adjoint solver

   Collective on ts

   Input Parameter:
.  ts - the `TS` context obtained from `TSCreate()`

   Level: advanced

.seealso: [](chapter_ts), `TSCreate()`, `TSAdjointStep()`, `TSSetCostGradients()`
@*/
PetscErrorCode TSAdjointSetUp(TS ts)
{
  TSTrajectory tj;
  PetscBool    match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  if (ts->adjointsetupcalled) PetscFunctionReturn(0);
  PetscCheck(ts->vecs_sensi, PetscObjectComm((PetscObject)ts), PETSC_ERR_ARG_WRONGSTATE, "Must call TSSetCostGradients() first");
  PetscCheck(!ts->vecs_sensip || ts->Jacp || ts->Jacprhs, PetscObjectComm((PetscObject)ts), PETSC_ERR_ARG_WRONGSTATE, "Must call TSSetRHSJacobianP() or TSSetIJacobianP() first");
  PetscCall(TSGetTrajectory(ts, &tj));
  PetscCall(PetscObjectTypeCompare((PetscObject)tj, TSTRAJECTORYBASIC, &match));
  if (match) {
    PetscBool solution_only;
    PetscCall(TSTrajectoryGetSolutionOnly(tj, &solution_only));
    PetscCheck(!solution_only, PetscObjectComm((PetscObject)ts), PETSC_ERR_USER, "TSAdjoint cannot use the solution-only mode when choosing the Basic TSTrajectory type. Turn it off with -ts_trajectory_solution_only 0");
  }
  PetscCall(TSTrajectorySetUseHistory(tj, PETSC_FALSE)); /* not use TSHistory */

  if (ts->quadraturets) { /* if there is integral in the cost function */
    PetscCall(VecDuplicate(ts->vecs_sensi[0], &ts->vec_drdu_col));
    if (ts->vecs_sensip) PetscCall(VecDuplicate(ts->vecs_sensip[0], &ts->vec_drdp_col));
  }

  PetscTryTypeMethod(ts, adjointsetup);
  ts->adjointsetupcalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
  TSAdjointReset - Resets a `TS` adjoint context and removes any allocated `Vec`s and `Mat`s.

   Collective on ts

   Input Parameter:
.  ts - the `TS` context obtained from `TSCreate()`

   Level: beginner

.seealso: [](chapter_ts), `TSCreate()`, `TSAdjointSetUp()`, `TSADestroy()`
@*/
PetscErrorCode TSAdjointReset(TS ts)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscTryTypeMethod(ts, adjointreset);
  if (ts->quadraturets) { /* if there is integral in the cost function */
    PetscCall(VecDestroy(&ts->vec_drdu_col));
    if (ts->vecs_sensip) PetscCall(VecDestroy(&ts->vec_drdp_col));
  }
  ts->vecs_sensi         = NULL;
  ts->vecs_sensip        = NULL;
  ts->vecs_sensi2        = NULL;
  ts->vecs_sensi2p       = NULL;
  ts->vec_dir            = NULL;
  ts->adjointsetupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@
   TSAdjointSetSteps - Sets the number of steps the adjoint solver should take backward in time

   Logically Collective on ts

   Input Parameters:
+  ts - the `TS` context obtained from `TSCreate()`
-  steps - number of steps to use

   Level: intermediate

   Notes:
    Normally one does not call this and `TSAdjointSolve()` integrates back to the original timestep. One can call this
          so as to integrate back to less than the original timestep

.seealso: [](chapter_ts), `TSAdjointSolve()`, `TS`, `TSSetExactFinalTime()`
@*/
PetscErrorCode TSAdjointSetSteps(TS ts, PetscInt steps)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidLogicalCollectiveInt(ts, steps, 2);
  PetscCheck(steps >= 0, PetscObjectComm((PetscObject)ts), PETSC_ERR_ARG_OUTOFRANGE, "Cannot step back a negative number of steps");
  PetscCheck(steps <= ts->steps, PetscObjectComm((PetscObject)ts), PETSC_ERR_ARG_OUTOFRANGE, "Cannot step back more than the total number of forward steps");
  ts->adjoint_max_steps = steps;
  PetscFunctionReturn(0);
}

/*@C
  TSAdjointSetRHSJacobian - Deprecated, use `TSSetRHSJacobianP()`

  Level: deprecated

@*/
PetscErrorCode TSAdjointSetRHSJacobian(TS ts, Mat Amat, PetscErrorCode (*func)(TS, PetscReal, Vec, Mat, void *), void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidHeaderSpecific(Amat, MAT_CLASSID, 2);

  ts->rhsjacobianp    = func;
  ts->rhsjacobianpctx = ctx;
  if (Amat) {
    PetscCall(PetscObjectReference((PetscObject)Amat));
    PetscCall(MatDestroy(&ts->Jacp));
    ts->Jacp = Amat;
  }
  PetscFunctionReturn(0);
}

/*@C
  TSAdjointComputeRHSJacobian - Deprecated, use `TSComputeRHSJacobianP()`

  Level: deprecated

@*/
PetscErrorCode TSAdjointComputeRHSJacobian(TS ts, PetscReal t, Vec U, Mat Amat)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidHeaderSpecific(U, VEC_CLASSID, 3);
  PetscValidPointer(Amat, 4);

  PetscCallBack("TS callback JacobianP for sensitivity analysis", (*ts->rhsjacobianp)(ts, t, U, Amat, ts->rhsjacobianpctx));
  PetscFunctionReturn(0);
}

/*@
  TSAdjointComputeDRDYFunction - Deprecated, use `TSGetQuadratureTS()` then `TSComputeRHSJacobian()`

  Level: deprecated

@*/
PetscErrorCode TSAdjointComputeDRDYFunction(TS ts, PetscReal t, Vec U, Vec *DRDU)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidHeaderSpecific(U, VEC_CLASSID, 3);

  PetscCallBack("TS callback DRDY for sensitivity analysis", (*ts->drdufunction)(ts, t, U, DRDU, ts->costintegrandctx));
  PetscFunctionReturn(0);
}

/*@
  TSAdjointComputeDRDPFunction - Deprecated, use `TSGetQuadratureTS()` then `TSComputeRHSJacobianP()`

  Level: deprecated

@*/
PetscErrorCode TSAdjointComputeDRDPFunction(TS ts, PetscReal t, Vec U, Vec *DRDP)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidHeaderSpecific(U, VEC_CLASSID, 3);

  PetscCallBack("TS callback DRDP for sensitivity analysis", (*ts->drdpfunction)(ts, t, U, DRDP, ts->costintegrandctx));
  PetscFunctionReturn(0);
}

/*@C
   TSAdjointMonitorSensi - monitors the first lambda sensitivity

   Level: intermediate

.seealso: [](chapter_ts), `TSAdjointMonitorSet()`
@*/
PetscErrorCode TSAdjointMonitorSensi(TS ts, PetscInt step, PetscReal ptime, Vec v, PetscInt numcost, Vec *lambda, Vec *mu, PetscViewerAndFormat *vf)
{
  PetscViewer viewer = vf->viewer;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 8);
  PetscCall(PetscViewerPushFormat(viewer, vf->format));
  PetscCall(VecView(lambda[0], viewer));
  PetscCall(PetscViewerPopFormat(viewer));
  PetscFunctionReturn(0);
}

/*@C
   TSAdjointMonitorSetFromOptions - Sets a monitor function and viewer appropriate for the type indicated by the user

   Collective on ts

   Input Parameters:
+  ts - `TS` object you wish to monitor
.  name - the monitor type one is seeking
.  help - message indicating what monitoring is done
.  manual - manual page for the monitor
.  monitor - the monitor function
-  monitorsetup - a function that is called once ONLY if the user selected this monitor that may set additional features of the `TS` or `PetscViewer` objects

   Level: developer

.seealso: [](chapter_ts), `PetscOptionsGetViewer()`, `PetscOptionsGetReal()`, `PetscOptionsHasName()`, `PetscOptionsGetString()`,
          `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsBool()`
          `PetscOptionsInt()`, `PetscOptionsString()`, `PetscOptionsReal()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`
@*/
PetscErrorCode TSAdjointMonitorSetFromOptions(TS ts, const char name[], const char help[], const char manual[], PetscErrorCode (*monitor)(TS, PetscInt, PetscReal, Vec, PetscInt, Vec *, Vec *, PetscViewerAndFormat *), PetscErrorCode (*monitorsetup)(TS, PetscViewerAndFormat *))
{
  PetscViewer       viewer;
  PetscViewerFormat format;
  PetscBool         flg;

  PetscFunctionBegin;
  PetscCall(PetscOptionsGetViewer(PetscObjectComm((PetscObject)ts), ((PetscObject)ts)->options, ((PetscObject)ts)->prefix, name, &viewer, &format, &flg));
  if (flg) {
    PetscViewerAndFormat *vf;
    PetscCall(PetscViewerAndFormatCreate(viewer, format, &vf));
    PetscCall(PetscObjectDereference((PetscObject)viewer));
    if (monitorsetup) PetscCall((*monitorsetup)(ts, vf));
    PetscCall(TSAdjointMonitorSet(ts, (PetscErrorCode(*)(TS, PetscInt, PetscReal, Vec, PetscInt, Vec *, Vec *, void *))monitor, vf, (PetscErrorCode(*)(void **))PetscViewerAndFormatDestroy));
  }
  PetscFunctionReturn(0);
}

/*@C
   TSAdjointMonitorSet - Sets an ADDITIONAL function that is to be used at every
   timestep to display the iteration's  progress.

   Logically Collective on ts

   Input Parameters:
+  ts - the `TS` context obtained from `TSCreate()`
.  adjointmonitor - monitoring routine
.  adjointmctx - [optional] user-defined context for private data for the
             monitor routine (use NULL if no context is desired)
-  adjointmonitordestroy - [optional] routine that frees monitor context
          (may be NULL)

   Calling sequence of monitor:
$    int adjointmonitor(TS ts,PetscInt steps,PetscReal time,Vec u,PetscInt numcost,Vec *lambda, Vec *mu,void *adjointmctx)

+    ts - the `TS` context
.    steps - iteration number (after the final time step the monitor routine is called with a step of -1, this is at the final time which may have
                               been interpolated to)
.    time - current time
.    u - current iterate
.    numcost - number of cost functionos
.    lambda - sensitivities to initial conditions
.    mu - sensitivities to parameters
-    adjointmctx - [optional] adjoint monitoring context

   Level: intermediate

   Note:
   This routine adds an additional monitor to the list of monitors that
   already has been loaded.

   Fortran Note:
   Only a single monitor function can be set for each TS object

.seealso: [](chapter_ts), `TS`, `TSAdjointSolve()`, `TSAdjointMonitorCancel()`
@*/
PetscErrorCode TSAdjointMonitorSet(TS ts, PetscErrorCode (*adjointmonitor)(TS, PetscInt, PetscReal, Vec, PetscInt, Vec *, Vec *, void *), void *adjointmctx, PetscErrorCode (*adjointmdestroy)(void **))
{
  PetscInt  i;
  PetscBool identical;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  for (i = 0; i < ts->numbermonitors; i++) {
    PetscCall(PetscMonitorCompare((PetscErrorCode(*)(void))adjointmonitor, adjointmctx, adjointmdestroy, (PetscErrorCode(*)(void))ts->adjointmonitor[i], ts->adjointmonitorcontext[i], ts->adjointmonitordestroy[i], &identical));
    if (identical) PetscFunctionReturn(0);
  }
  PetscCheck(ts->numberadjointmonitors < MAXTSMONITORS, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Too many adjoint monitors set");
  ts->adjointmonitor[ts->numberadjointmonitors]          = adjointmonitor;
  ts->adjointmonitordestroy[ts->numberadjointmonitors]   = adjointmdestroy;
  ts->adjointmonitorcontext[ts->numberadjointmonitors++] = (void *)adjointmctx;
  PetscFunctionReturn(0);
}

/*@C
   TSAdjointMonitorCancel - Clears all the adjoint monitors that have been set on a time-step object.

   Logically Collective on ts

   Input Parameters:
.  ts - the `TS` context obtained from `TSCreate()`

   Notes:
   There is no way to remove a single, specific monitor.

   Level: intermediate

.seealso: [](chapter_ts), `TS`, `TSAdjointSolve()`, `TSAdjointMonitorSet()`
@*/
PetscErrorCode TSAdjointMonitorCancel(TS ts)
{
  PetscInt i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  for (i = 0; i < ts->numberadjointmonitors; i++) {
    if (ts->adjointmonitordestroy[i]) PetscCall((*ts->adjointmonitordestroy[i])(&ts->adjointmonitorcontext[i]));
  }
  ts->numberadjointmonitors = 0;
  PetscFunctionReturn(0);
}

/*@C
   TSAdjointMonitorDefault - the default monitor of adjoint computations

   Level: intermediate

.seealso: [](chapter_ts), `TS`, `TSAdjointSolve()`, `TSAdjointMonitorSet()`
@*/
PetscErrorCode TSAdjointMonitorDefault(TS ts, PetscInt step, PetscReal ptime, Vec v, PetscInt numcost, Vec *lambda, Vec *mu, PetscViewerAndFormat *vf)
{
  PetscViewer viewer = vf->viewer;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 8);
  PetscCall(PetscViewerPushFormat(viewer, vf->format));
  PetscCall(PetscViewerASCIIAddTab(viewer, ((PetscObject)ts)->tablevel));
  PetscCall(PetscViewerASCIIPrintf(viewer, "%" PetscInt_FMT " TS dt %g time %g%s", step, (double)ts->time_step, (double)ptime, ts->steprollback ? " (r)\n" : "\n"));
  PetscCall(PetscViewerASCIISubtractTab(viewer, ((PetscObject)ts)->tablevel));
  PetscCall(PetscViewerPopFormat(viewer));
  PetscFunctionReturn(0);
}

/*@C
   TSAdjointMonitorDrawSensi - Monitors progress of the adjoint `TS` solvers by calling
   `VecView()` for the sensitivities to initial states at each timestep

   Collective on ts

   Input Parameters:
+  ts - the `TS` context
.  step - current time-step
.  ptime - current time
.  u - current state
.  numcost - number of cost functions
.  lambda - sensitivities to initial conditions
.  mu - sensitivities to parameters
-  dummy - either a viewer or NULL

   Level: intermediate

.seealso: [](chapter_ts), `TSAdjointSolve()`, `TSAdjointMonitorSet()`, `TSAdjointMonitorDefault()`, `VecView()`
@*/
PetscErrorCode TSAdjointMonitorDrawSensi(TS ts, PetscInt step, PetscReal ptime, Vec u, PetscInt numcost, Vec *lambda, Vec *mu, void *dummy)
{
  TSMonitorDrawCtx ictx = (TSMonitorDrawCtx)dummy;
  PetscDraw        draw;
  PetscReal        xl, yl, xr, yr, h;
  char             time[32];

  PetscFunctionBegin;
  if (!(((ictx->howoften > 0) && (!(step % ictx->howoften))) || ((ictx->howoften == -1) && ts->reason))) PetscFunctionReturn(0);

  PetscCall(VecView(lambda[0], ictx->viewer));
  PetscCall(PetscViewerDrawGetDraw(ictx->viewer, 0, &draw));
  PetscCall(PetscSNPrintf(time, 32, "Timestep %d Time %g", (int)step, (double)ptime));
  PetscCall(PetscDrawGetCoordinates(draw, &xl, &yl, &xr, &yr));
  h = yl + .95 * (yr - yl);
  PetscCall(PetscDrawStringCentered(draw, .5 * (xl + xr), h, PETSC_DRAW_BLACK, time));
  PetscCall(PetscDrawFlush(draw));
  PetscFunctionReturn(0);
}

/*
   TSAdjointSetFromOptions - Sets various `TS` adjoint parameters from user options.

   Collective on ts

   Input Parameter:
.  ts - the `TS` context

   Options Database Keys:
+  -ts_adjoint_solve <yes,no> After solving the ODE/DAE solve the adjoint problem (requires -ts_save_trajectory)
.  -ts_adjoint_monitor - print information at each adjoint time step
-  -ts_adjoint_monitor_draw_sensi - monitor the sensitivity of the first cost function wrt initial conditions (lambda[0]) graphically

   Level: developer

   Note:
    This is not normally called directly by users

.seealso: [](chapter_ts), `TSSetSaveTrajectory()`, `TSTrajectorySetUp()`
*/
PetscErrorCode TSAdjointSetFromOptions(TS ts, PetscOptionItems *PetscOptionsObject)
{
  PetscBool tflg, opt;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscOptionsHeadBegin(PetscOptionsObject, "TS Adjoint options");
  tflg = ts->adjoint_solve ? PETSC_TRUE : PETSC_FALSE;
  PetscCall(PetscOptionsBool("-ts_adjoint_solve", "Solve the adjoint problem immediately after solving the forward problem", "", tflg, &tflg, &opt));
  if (opt) {
    PetscCall(TSSetSaveTrajectory(ts));
    ts->adjoint_solve = tflg;
  }
  PetscCall(TSAdjointMonitorSetFromOptions(ts, "-ts_adjoint_monitor", "Monitor adjoint timestep size", "TSAdjointMonitorDefault", TSAdjointMonitorDefault, NULL));
  PetscCall(TSAdjointMonitorSetFromOptions(ts, "-ts_adjoint_monitor_sensi", "Monitor sensitivity in the adjoint computation", "TSAdjointMonitorSensi", TSAdjointMonitorSensi, NULL));
  opt = PETSC_FALSE;
  PetscCall(PetscOptionsName("-ts_adjoint_monitor_draw_sensi", "Monitor adjoint sensitivities (lambda only) graphically", "TSAdjointMonitorDrawSensi", &opt));
  if (opt) {
    TSMonitorDrawCtx ctx;
    PetscInt         howoften = 1;

    PetscCall(PetscOptionsInt("-ts_adjoint_monitor_draw_sensi", "Monitor adjoint sensitivities (lambda only) graphically", "TSAdjointMonitorDrawSensi", howoften, &howoften, NULL));
    PetscCall(TSMonitorDrawCtxCreate(PetscObjectComm((PetscObject)ts), NULL, NULL, PETSC_DECIDE, PETSC_DECIDE, 300, 300, howoften, &ctx));
    PetscCall(TSAdjointMonitorSet(ts, TSAdjointMonitorDrawSensi, ctx, (PetscErrorCode(*)(void **))TSMonitorDrawCtxDestroy));
  }
  PetscFunctionReturn(0);
}

/*@
   TSAdjointStep - Steps one time step backward in the adjoint run

   Collective on ts

   Input Parameter:
.  ts - the `TS` context obtained from `TSCreate()`

   Level: intermediate

.seealso: [](chapter_ts), `TSAdjointSetUp()`, `TSAdjointSolve()`
@*/
PetscErrorCode TSAdjointStep(TS ts)
{
  DM dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscCall(TSGetDM(ts, &dm));
  PetscCall(TSAdjointSetUp(ts));
  ts->steps--; /* must decrease the step index before the adjoint step is taken. */

  ts->reason     = TS_CONVERGED_ITERATING;
  ts->ptime_prev = ts->ptime;
  PetscCall(PetscLogEventBegin(TS_AdjointStep, ts, 0, 0, 0));
  PetscUseTypeMethod(ts, adjointstep);
  PetscCall(PetscLogEventEnd(TS_AdjointStep, ts, 0, 0, 0));
  ts->adjoint_steps++;

  if (ts->reason < 0) {
    PetscCheck(!ts->errorifstepfailed, PetscObjectComm((PetscObject)ts), PETSC_ERR_NOT_CONVERGED, "TSAdjointStep has failed due to %s", TSConvergedReasons[ts->reason]);
  } else if (!ts->reason) {
    if (ts->adjoint_steps >= ts->adjoint_max_steps) ts->reason = TS_CONVERGED_ITS;
  }
  PetscFunctionReturn(0);
}

/*@
   TSAdjointSolve - Solves the discrete ajoint problem for an ODE/DAE

   Collective on ts
`
   Input Parameter:
.  ts - the `TS` context obtained from `TSCreate()`

   Options Database Key:
. -ts_adjoint_view_solution <viewerinfo> - views the first gradient with respect to the initial values

   Level: intermediate

   Notes:
   This must be called after a call to `TSSolve()` that solves the forward problem

   By default this will integrate back to the initial time, one can use `TSAdjointSetSteps()` to step back to a later time

.seealso: [](chapter_ts), `TSAdjointSolve()`, `TSCreate()`, `TSSetCostGradients()`, `TSSetSolution()`, `TSAdjointStep()`
@*/
PetscErrorCode TSAdjointSolve(TS ts)
{
  static PetscBool cite = PETSC_FALSE;
#if defined(TSADJOINT_STAGE)
  PetscLogStage adjoint_stage;
#endif

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscCall(PetscCitationsRegister("@article{Zhang2022tsadjoint,\n"
                                   "  title         = {{PETSc TSAdjoint: A Discrete Adjoint ODE Solver for First-Order and Second-Order Sensitivity Analysis}},\n"
                                   "  author        = {Zhang, Hong and Constantinescu, Emil M.  and Smith, Barry F.},\n"
                                   "  journal       = {SIAM Journal on Scientific Computing},\n"
                                   "  volume        = {44},\n"
                                   "  number        = {1},\n"
                                   "  pages         = {C1-C24},\n"
                                   "  doi           = {10.1137/21M140078X},\n"
                                   "  year          = {2022}\n}\n",
                                   &cite));
#if defined(TSADJOINT_STAGE)
  PetscCall(PetscLogStageRegister("TSAdjoint", &adjoint_stage));
  PetscCall(PetscLogStagePush(adjoint_stage));
#endif
  PetscCall(TSAdjointSetUp(ts));

  /* reset time step and iteration counters */
  ts->adjoint_steps     = 0;
  ts->ksp_its           = 0;
  ts->snes_its          = 0;
  ts->num_snes_failures = 0;
  ts->reject            = 0;
  ts->reason            = TS_CONVERGED_ITERATING;

  if (!ts->adjoint_max_steps) ts->adjoint_max_steps = ts->steps;
  if (ts->adjoint_steps >= ts->adjoint_max_steps) ts->reason = TS_CONVERGED_ITS;

  while (!ts->reason) {
    PetscCall(TSTrajectoryGet(ts->trajectory, ts, ts->steps, &ts->ptime));
    PetscCall(TSAdjointMonitor(ts, ts->steps, ts->ptime, ts->vec_sol, ts->numcost, ts->vecs_sensi, ts->vecs_sensip));
    PetscCall(TSAdjointEventHandler(ts));
    PetscCall(TSAdjointStep(ts));
    if ((ts->vec_costintegral || ts->quadraturets) && !ts->costintegralfwd) PetscCall(TSAdjointCostIntegral(ts));
  }
  if (!ts->steps) {
    PetscCall(TSTrajectoryGet(ts->trajectory, ts, ts->steps, &ts->ptime));
    PetscCall(TSAdjointMonitor(ts, ts->steps, ts->ptime, ts->vec_sol, ts->numcost, ts->vecs_sensi, ts->vecs_sensip));
  }
  ts->solvetime = ts->ptime;
  PetscCall(TSTrajectoryViewFromOptions(ts->trajectory, NULL, "-ts_trajectory_view"));
  PetscCall(VecViewFromOptions(ts->vecs_sensi[0], (PetscObject)ts, "-ts_adjoint_view_solution"));
  ts->adjoint_max_steps = 0;
#if defined(TSADJOINT_STAGE)
  PetscCall(PetscLogStagePop());
#endif
  PetscFunctionReturn(0);
}

/*@C
   TSAdjointMonitor - Runs all user-provided adjoint monitor routines set using `TSAdjointMonitorSet()`

   Collective on ts

   Input Parameters:
+  ts - time stepping context obtained from `TSCreate()`
.  step - step number that has just completed
.  ptime - model time of the state
.  u - state at the current model time
.  numcost - number of cost functions (dimension of lambda  or mu)
.  lambda - vectors containing the gradients of the cost functions with respect to the ODE/DAE solution variables
-  mu - vectors containing the gradients of the cost functions with respect to the problem parameters

   Level: developer

   Note:
   `TSAdjointMonitor()` is typically used automatically within the time stepping implementations.
   Users would almost never call this routine directly.

.seealso: `TSAdjointMonitorSet()`, `TSAdjointSolve()`
@*/
PetscErrorCode TSAdjointMonitor(TS ts, PetscInt step, PetscReal ptime, Vec u, PetscInt numcost, Vec *lambda, Vec *mu)
{
  PetscInt i, n = ts->numberadjointmonitors;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidHeaderSpecific(u, VEC_CLASSID, 4);
  PetscCall(VecLockReadPush(u));
  for (i = 0; i < n; i++) PetscCall((*ts->adjointmonitor[i])(ts, step, ptime, u, numcost, lambda, mu, ts->adjointmonitorcontext[i]));
  PetscCall(VecLockReadPop(u));
  PetscFunctionReturn(0);
}

/*@
 TSAdjointCostIntegral - Evaluate the cost integral in the adjoint run.

 Collective on ts

 Input Parameter:
 .  ts - time stepping context

 Level: advanced

 Notes:
 This function cannot be called until `TSAdjointStep()` has been completed.

 .seealso: [](chapter_ts), `TSAdjointSolve()`, `TSAdjointStep()`
 @*/
PetscErrorCode TSAdjointCostIntegral(TS ts)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscUseTypeMethod(ts, adjointintegral);
  PetscFunctionReturn(0);
}

/* ------------------ Forward (tangent linear) sensitivity  ------------------*/

/*@
  TSForwardSetUp - Sets up the internal data structures for the later use
  of forward sensitivity analysis

  Collective on ts

  Input Parameter:
. ts - the `TS` context obtained from `TSCreate()`

  Level: advanced

.seealso: [](chapter_ts), `TS`, `TSCreate()`, `TSDestroy()`, `TSSetUp()`
@*/
PetscErrorCode TSForwardSetUp(TS ts)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  if (ts->forwardsetupcalled) PetscFunctionReturn(0);
  PetscTryTypeMethod(ts, forwardsetup);
  PetscCall(VecDuplicate(ts->vec_sol, &ts->vec_sensip_col));
  ts->forwardsetupcalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
  TSForwardReset - Reset the internal data structures used by forward sensitivity analysis

  Collective on ts

  Input Parameter:
. ts - the `TS` context obtained from `TSCreate()`

  Level: advanced

.seealso: [](chapter_ts), `TSCreate()`, `TSDestroy()`, `TSForwardSetUp()`
@*/
PetscErrorCode TSForwardReset(TS ts)
{
  TS quadts = ts->quadraturets;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscTryTypeMethod(ts, forwardreset);
  PetscCall(MatDestroy(&ts->mat_sensip));
  if (quadts) PetscCall(MatDestroy(&quadts->mat_sensip));
  PetscCall(VecDestroy(&ts->vec_sensip_col));
  ts->forward_solve      = PETSC_FALSE;
  ts->forwardsetupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@
  TSForwardSetIntegralGradients - Set the vectors holding forward sensitivities of the integral term.

  Input Parameters:
+ ts - the `TS` context obtained from `TSCreate()`
. numfwdint - number of integrals
- vp - the vectors containing the gradients for each integral w.r.t. parameters

  Level: deprecated

.seealso: [](chapter_ts), `TSForwardGetSensitivities()`, `TSForwardSetIntegralGradients()`, `TSForwardGetIntegralGradients()`, `TSForwardStep()`
@*/
PetscErrorCode TSForwardSetIntegralGradients(TS ts, PetscInt numfwdint, Vec *vp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscCheck(!ts->numcost || ts->numcost == numfwdint, PetscObjectComm((PetscObject)ts), PETSC_ERR_USER, "The number of cost functions (2nd parameter of TSSetCostIntegrand()) is inconsistent with the one set by TSSetCostIntegrand()");
  if (!ts->numcost) ts->numcost = numfwdint;

  ts->vecs_integral_sensip = vp;
  PetscFunctionReturn(0);
}

/*@
  TSForwardGetIntegralGradients - Returns the forward sensitivities ofthe integral term.

  Input Parameter:
. ts - the `TS` context obtained from `TSCreate()`

  Output Parameter:
. vp - the vectors containing the gradients for each integral w.r.t. parameters

  Level: deprecated

.seealso: [](chapter_ts), `TSForwardSetSensitivities()`, `TSForwardSetIntegralGradients()`, `TSForwardGetIntegralGradients()`, `TSForwardStep()`
@*/
PetscErrorCode TSForwardGetIntegralGradients(TS ts, PetscInt *numfwdint, Vec **vp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidPointer(vp, 3);
  if (numfwdint) *numfwdint = ts->numcost;
  if (vp) *vp = ts->vecs_integral_sensip;
  PetscFunctionReturn(0);
}

/*@
  TSForwardStep - Compute the forward sensitivity for one time step.

  Collective on ts

  Input Parameter:
. ts - time stepping context

  Level: advanced

  Notes:
  This function cannot be called until `TSStep()` has been completed.

.seealso: [](chapter_ts), `TSForwardSetSensitivities()`, `TSForwardGetSensitivities()`, `TSForwardSetIntegralGradients()`, `TSForwardGetIntegralGradients()`, `TSForwardSetUp()`
@*/
PetscErrorCode TSForwardStep(TS ts)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscCall(PetscLogEventBegin(TS_ForwardStep, ts, 0, 0, 0));
  PetscUseTypeMethod(ts, forwardstep);
  PetscCall(PetscLogEventEnd(TS_ForwardStep, ts, 0, 0, 0));
  PetscCheck(ts->reason >= 0 || !ts->errorifstepfailed, PetscObjectComm((PetscObject)ts), PETSC_ERR_NOT_CONVERGED, "TSFowardStep has failed due to %s", TSConvergedReasons[ts->reason]);
  PetscFunctionReturn(0);
}

/*@
  TSForwardSetSensitivities - Sets the initial value of the trajectory sensitivities of solution  w.r.t. the problem parameters and initial values.

  Logically Collective on ts

  Input Parameters:
+ ts - the `TS` context obtained from `TSCreate()`
. nump - number of parameters
- Smat - sensitivities with respect to the parameters, the number of entries in these vectors is the same as the number of parameters

  Level: beginner

  Notes:
  Forward sensitivity is also called 'trajectory sensitivity' in some fields such as power systems.
  This function turns on a flag to trigger `TSSolve()` to compute forward sensitivities automatically.
  You must call this function before `TSSolve()`.
  The entries in the sensitivity matrix must be correctly initialized with the values S = dy/dp|startingtime.

.seealso: [](chapter_ts), `TSForwardGetSensitivities()`, `TSForwardSetIntegralGradients()`, `TSForwardGetIntegralGradients()`, `TSForwardStep()`
@*/
PetscErrorCode TSForwardSetSensitivities(TS ts, PetscInt nump, Mat Smat)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidHeaderSpecific(Smat, MAT_CLASSID, 3);
  ts->forward_solve = PETSC_TRUE;
  if (nump == PETSC_DEFAULT) {
    PetscCall(MatGetSize(Smat, NULL, &ts->num_parameters));
  } else ts->num_parameters = nump;
  PetscCall(PetscObjectReference((PetscObject)Smat));
  PetscCall(MatDestroy(&ts->mat_sensip));
  ts->mat_sensip = Smat;
  PetscFunctionReturn(0);
}

/*@
  TSForwardGetSensitivities - Returns the trajectory sensitivities

  Not Collective, but Smat returned is parallel if ts is parallel

  Output Parameters:
+ ts - the `TS` context obtained from `TSCreate()`
. nump - number of parameters
- Smat - sensitivities with respect to the parameters, the number of entries in these vectors is the same as the number of parameters

  Level: intermediate

.seealso: [](chapter_ts), `TSForwardSetSensitivities()`, `TSForwardSetIntegralGradients()`, `TSForwardGetIntegralGradients()`, `TSForwardStep()`
@*/
PetscErrorCode TSForwardGetSensitivities(TS ts, PetscInt *nump, Mat *Smat)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  if (nump) *nump = ts->num_parameters;
  if (Smat) *Smat = ts->mat_sensip;
  PetscFunctionReturn(0);
}

/*@
   TSForwardCostIntegral - Evaluate the cost integral in the forward run.

   Collective on ts

   Input Parameter:
.  ts - time stepping context

   Level: advanced

   Note:
   This function cannot be called until `TSStep()` has been completed.

.seealso: [](chapter_ts), `TS`, `TSSolve()`, `TSAdjointCostIntegral()`
@*/
PetscErrorCode TSForwardCostIntegral(TS ts)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscUseTypeMethod(ts, forwardintegral);
  PetscFunctionReturn(0);
}

/*@
  TSForwardSetInitialSensitivities - Set initial values for tangent linear sensitivities

  Collective on ts

  Input Parameters:
+ ts - the `TS` context obtained from `TSCreate()`
- didp - parametric sensitivities of the initial condition

  Level: intermediate

  Notes:
  `TSSolve()` allows users to pass the initial solution directly to `TS`. But the tangent linear variables cannot be initialized in this way.
   This function is used to set initial values for tangent linear variables.

.seealso: [](chapter_ts), `TS`, `TSForwardSetSensitivities()`
@*/
PetscErrorCode TSForwardSetInitialSensitivities(TS ts, Mat didp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidHeaderSpecific(didp, MAT_CLASSID, 2);
  if (!ts->mat_sensip) PetscCall(TSForwardSetSensitivities(ts, PETSC_DEFAULT, didp));
  PetscFunctionReturn(0);
}

/*@
   TSForwardGetStages - Get the number of stages and the tangent linear sensitivities at the intermediate stages

   Input Parameter:
.  ts - the `TS` context obtained from `TSCreate()`

   Output Parameters:
+  ns - number of stages
-  S - tangent linear sensitivities at the intermediate stages

   Level: advanced

.seealso: `TS`
@*/
PetscErrorCode TSForwardGetStages(TS ts, PetscInt *ns, Mat **S)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);

  if (!ts->ops->getstages) *S = NULL;
  else PetscUseTypeMethod(ts, forwardgetstages, ns, S);
  PetscFunctionReturn(0);
}

/*@
   TSCreateQuadratureTS - Create a sub-`TS` that evaluates integrals over time

   Input Parameters:
+  ts - the `TS` context obtained from `TSCreate()`
-  fwd - flag indicating whether to evaluate cost integral in the forward run or the adjoint run

   Output Parameters:
.  quadts - the child `TS` context

   Level: intermediate

.seealso: [](chapter_ts), `TSGetQuadratureTS()`
@*/
PetscErrorCode TSCreateQuadratureTS(TS ts, PetscBool fwd, TS *quadts)
{
  char prefix[128];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidPointer(quadts, 3);
  PetscCall(TSDestroy(&ts->quadraturets));
  PetscCall(TSCreate(PetscObjectComm((PetscObject)ts), &ts->quadraturets));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)ts->quadraturets, (PetscObject)ts, 1));
  PetscCall(PetscSNPrintf(prefix, sizeof(prefix), "%squad_", ((PetscObject)ts)->prefix ? ((PetscObject)ts)->prefix : ""));
  PetscCall(TSSetOptionsPrefix(ts->quadraturets, prefix));
  *quadts = ts->quadraturets;

  if (ts->numcost) {
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, ts->numcost, &(*quadts)->vec_sol));
  } else {
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, 1, &(*quadts)->vec_sol));
  }
  ts->costintegralfwd = fwd;
  PetscFunctionReturn(0);
}

/*@
   TSGetQuadratureTS - Return the sub-`TS` that evaluates integrals over time

   Input Parameter:
.  ts - the `TS` context obtained from `TSCreate()`

   Output Parameters:
+  fwd - flag indicating whether to evaluate cost integral in the forward run or the adjoint run
-  quadts - the child `TS` context

   Level: intermediate

.seealso: [](chapter_ts), `TSCreateQuadratureTS()`
@*/
PetscErrorCode TSGetQuadratureTS(TS ts, PetscBool *fwd, TS *quadts)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  if (fwd) *fwd = ts->costintegralfwd;
  if (quadts) *quadts = ts->quadraturets;
  PetscFunctionReturn(0);
}

/*@
   TSComputeSNESJacobian - Compute the Jacobian needed for the `SNESSolve()` in `TS`

   Collective on ts

   Input Parameters:
+  ts - the `TS` context obtained from `TSCreate()`
-  x - state vector

   Output Parameters:
+  J - Jacobian matrix
-  Jpre - preconditioning matrix for J (may be same as J)

   Level: developer

   Note:
   Uses finite differencing when `TS` Jacobian is not available.

.seealso: `SNES`, `TS`, `SNESSetJacobian()`, TSSetRHSJacobian()`, `TSSetIJacobian()`
@*/
PetscErrorCode TSComputeSNESJacobian(TS ts, Vec x, Mat J, Mat Jpre)
{
  SNES snes                                          = ts->snes;
  PetscErrorCode (*jac)(SNES, Vec, Mat, Mat, void *) = NULL;

  PetscFunctionBegin;
  /*
    Unlike implicit methods, explicit methods do not have SNESMatFDColoring in the snes object
    because SNESSolve() has not been called yet; so querying SNESMatFDColoring does not work for
    explicit methods. Instead, we check the Jacobian compute function directly to determin if FD
    coloring is used.
  */
  PetscCall(SNESGetJacobian(snes, NULL, NULL, &jac, NULL));
  if (jac == SNESComputeJacobianDefaultColor) {
    Vec f;
    PetscCall(SNESSetSolution(snes, x));
    PetscCall(SNESGetFunction(snes, &f, NULL, NULL));
    /* Force MatFDColoringApply to evaluate the SNES residual function for the base vector */
    PetscCall(SNESComputeFunction(snes, x, f));
  }
  PetscCall(SNESComputeJacobian(snes, x, J, Jpre));
  PetscFunctionReturn(0);
}
