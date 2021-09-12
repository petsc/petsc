#include <petsc/private/tsimpl.h>        /*I "petscts.h"  I*/
#include <petscdraw.h>

PetscLogEvent TS_AdjointStep,TS_ForwardStep,TS_JacobianPEval;

/* #define TSADJOINT_STAGE */

/* ------------------------ Sensitivity Context ---------------------------*/

/*@C
  TSSetRHSJacobianP - Sets the function that computes the Jacobian of G w.r.t. the parameters P where U_t = G(U,P,t), as well as the location to store the matrix.

  Logically Collective on TS

  Input Parameters:
+ ts - TS context obtained from TSCreate()
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

  Notes:
    Amat has the same number of rows and the same row parallel layout as u, Amat has the same number of columns and parallel layout as p

.seealso: TSGetRHSJacobianP()
@*/
PetscErrorCode TSSetRHSJacobianP(TS ts,Mat Amat,PetscErrorCode (*func)(TS,PetscReal,Vec,Mat,void*),void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID,1);
  PetscValidHeaderSpecific(Amat,MAT_CLASSID,2);

  ts->rhsjacobianp    = func;
  ts->rhsjacobianpctx = ctx;
  if (Amat) {
    ierr = PetscObjectReference((PetscObject)Amat);CHKERRQ(ierr);
    ierr = MatDestroy(&ts->Jacprhs);CHKERRQ(ierr);
    ts->Jacprhs = Amat;
  }
  PetscFunctionReturn(0);
}

/*@C
  TSGetRHSJacobianP - Gets the function that computes the Jacobian of G w.r.t. the parameters P where U_t = G(U,P,t), as well as the location to store the matrix.

  Logically Collective on TS

  Input Parameter:
. ts - TS context obtained from TSCreate()

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

  Notes:
    Amat has the same number of rows and the same row parallel layout as u, Amat has the same number of columns and parallel layout as p

.seealso: TSSetRHSJacobianP()
@*/
PetscErrorCode TSGetRHSJacobianP(TS ts,Mat *Amat,PetscErrorCode (**func)(TS,PetscReal,Vec,Mat,void*),void **ctx)
{
  PetscFunctionBegin;
  if (func) *func = ts->rhsjacobianp;
  if (ctx) *ctx  = ts->rhsjacobianpctx;
  if (Amat) *Amat = ts->Jacprhs;
  PetscFunctionReturn(0);
}

/*@C
  TSComputeRHSJacobianP - Runs the user-defined JacobianP function.

  Collective on TS

  Input Parameters:
. ts   - The TS context obtained from TSCreate()

  Level: developer

.seealso: TSSetRHSJacobianP()
@*/
PetscErrorCode TSComputeRHSJacobianP(TS ts,PetscReal t,Vec U,Mat Amat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!Amat) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(U,VEC_CLASSID,3);

  PetscStackPush("TS user JacobianP function for sensitivity analysis");
  ierr = (*ts->rhsjacobianp)(ts,t,U,Amat,ts->rhsjacobianpctx);CHKERRQ(ierr);
  PetscStackPop;
  PetscFunctionReturn(0);
}

/*@C
  TSSetIJacobianP - Sets the function that computes the Jacobian of F w.r.t. the parameters P where F(Udot,U,t) = G(U,P,t), as well as the location to store the matrix.

  Logically Collective on TS

  Input Parameters:
+ ts - TS context obtained from TSCreate()
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

  Notes:
    Amat has the same number of rows and the same row parallel layout as u, Amat has the same number of columns and parallel layout as p

.seealso:
@*/
PetscErrorCode TSSetIJacobianP(TS ts,Mat Amat,PetscErrorCode (*func)(TS,PetscReal,Vec,Vec,PetscReal,Mat,void*),void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID,1);
  PetscValidHeaderSpecific(Amat,MAT_CLASSID,2);

  ts->ijacobianp    = func;
  ts->ijacobianpctx = ctx;
  if (Amat) {
    ierr = PetscObjectReference((PetscObject)Amat);CHKERRQ(ierr);
    ierr = MatDestroy(&ts->Jacp);CHKERRQ(ierr);
    ts->Jacp = Amat;
  }
  PetscFunctionReturn(0);
}

/*@C
  TSComputeIJacobianP - Runs the user-defined IJacobianP function.

  Collective on TS

  Input Parameters:
+ ts - the TS context
. t - current timestep
. U - state vector
. Udot - time derivative of state vector
. shift - shift to apply, see note below
- imex - flag indicates if the method is IMEX so that the RHSJacobian should be kept separate

  Output Parameters:
. A - Jacobian matrix

  Level: developer

.seealso: TSSetIJacobianP()
@*/
PetscErrorCode TSComputeIJacobianP(TS ts,PetscReal t,Vec U,Vec Udot,PetscReal shift,Mat Amat,PetscBool imex)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!Amat) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(U,VEC_CLASSID,3);
  PetscValidHeaderSpecific(Udot,VEC_CLASSID,4);

  ierr = PetscLogEventBegin(TS_JacobianPEval,ts,U,Amat,0);CHKERRQ(ierr);
  if (ts->ijacobianp) {
    PetscStackPush("TS user JacobianP function for sensitivity analysis");
    ierr = (*ts->ijacobianp)(ts,t,U,Udot,shift,Amat,ts->ijacobianpctx);CHKERRQ(ierr);
    PetscStackPop;
  }
  if (imex) {
    if (!ts->ijacobianp) {  /* system was written as Udot = G(t,U) */
      PetscBool assembled;
      ierr = MatZeroEntries(Amat);CHKERRQ(ierr);
      ierr = MatAssembled(Amat,&assembled);CHKERRQ(ierr);
      if (!assembled) {
        ierr = MatAssemblyBegin(Amat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(Amat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      }
    }
  } else {
    if (ts->rhsjacobianp) {
      ierr = TSComputeRHSJacobianP(ts,t,U,ts->Jacprhs);CHKERRQ(ierr);
    }
    if (ts->Jacprhs == Amat) { /* No IJacobian, so we only have the RHS matrix */
      ierr = MatScale(Amat,-1);CHKERRQ(ierr);
    } else if (ts->Jacprhs) { /* Both IJacobian and RHSJacobian */
      MatStructure axpy = DIFFERENT_NONZERO_PATTERN;
      if (!ts->ijacobianp) { /* No IJacobianp provided, but we have a separate RHS matrix */
        ierr = MatZeroEntries(Amat);CHKERRQ(ierr);
      }
      ierr = MatAXPY(Amat,-1,ts->Jacprhs,axpy);CHKERRQ(ierr);
    }
  }
  ierr = PetscLogEventEnd(TS_JacobianPEval,ts,U,Amat,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
    TSSetCostIntegrand - Sets the routine for evaluating the integral term in one or more cost functions

    Logically Collective on TS

    Input Parameters:
+   ts - the TS context obtained from TSCreate()
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

    Notes:
    For optimization there is usually a single cost function (numcost = 1). For sensitivities there may be multiple cost functions

.seealso: TSSetRHSJacobianP(), TSGetCostGradients(), TSSetCostGradients()
@*/
PetscErrorCode TSSetCostIntegrand(TS ts,PetscInt numcost,Vec costintegral,PetscErrorCode (*rf)(TS,PetscReal,Vec,Vec,void*),
                                                          PetscErrorCode (*drduf)(TS,PetscReal,Vec,Vec*,void*),
                                                          PetscErrorCode (*drdpf)(TS,PetscReal,Vec,Vec*,void*),
                                                          PetscBool fwd,void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (costintegral) PetscValidHeaderSpecific(costintegral,VEC_CLASSID,3);
  if (ts->numcost && ts->numcost!=numcost) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"The number of cost functions (2nd parameter of TSSetCostIntegrand()) is inconsistent with the one set by TSSetCostGradients() or TSForwardSetIntegralGradients()");
  if (!ts->numcost) ts->numcost=numcost;

  if (costintegral) {
    ierr = PetscObjectReference((PetscObject)costintegral);CHKERRQ(ierr);
    ierr = VecDestroy(&ts->vec_costintegral);CHKERRQ(ierr);
    ts->vec_costintegral = costintegral;
  } else {
    if (!ts->vec_costintegral) { /* Create a seq vec if user does not provide one */
      ierr = VecCreateSeq(PETSC_COMM_SELF,numcost,&ts->vec_costintegral);CHKERRQ(ierr);
    } else {
      ierr = VecSet(ts->vec_costintegral,0.0);CHKERRQ(ierr);
    }
  }
  if (!ts->vec_costintegrand) {
    ierr = VecDuplicate(ts->vec_costintegral,&ts->vec_costintegrand);CHKERRQ(ierr);
  } else {
    ierr = VecSet(ts->vec_costintegrand,0.0);CHKERRQ(ierr);
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
.  ts - the TS context obtained from TSCreate()

   Output Parameter:
.  v - the vector containing the integrals for each cost function

   Level: intermediate

.seealso: TSSetCostIntegrand()

@*/
PetscErrorCode  TSGetCostIntegral(TS ts,Vec *v)
{
  TS             quadts;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidPointer(v,2);
  ierr = TSGetQuadratureTS(ts,NULL,&quadts);CHKERRQ(ierr);
  *v = quadts->vec_sol;
  PetscFunctionReturn(0);
}

/*@C
   TSComputeCostIntegrand - Evaluates the integral function in the cost functions.

   Input Parameters:
+  ts - the TS context
.  t - current time
-  U - state vector, i.e. current solution

   Output Parameter:
.  Q - vector of size numcost to hold the outputs

   Notes:
   Most users should not need to explicitly call this routine, as it
   is used internally within the sensitivity analysis context.

   Level: deprecated

.seealso: TSSetCostIntegrand()
@*/
PetscErrorCode TSComputeCostIntegrand(TS ts,PetscReal t,Vec U,Vec Q)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(U,VEC_CLASSID,3);
  PetscValidHeaderSpecific(Q,VEC_CLASSID,4);

  ierr = PetscLogEventBegin(TS_FunctionEval,ts,U,Q,0);CHKERRQ(ierr);
  if (ts->costintegrand) {
    PetscStackPush("TS user integrand in the cost function");
    ierr = (*ts->costintegrand)(ts,t,U,Q,ts->costintegrandctx);CHKERRQ(ierr);
    PetscStackPop;
  } else {
    ierr = VecZeroEntries(Q);CHKERRQ(ierr);
  }

  ierr = PetscLogEventEnd(TS_FunctionEval,ts,U,Q,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  TSComputeDRDUFunction - Deprecated, use TSGetQuadratureTS() then TSComputeRHSJacobian()

  Level: deprecated

@*/
PetscErrorCode TSComputeDRDUFunction(TS ts,PetscReal t,Vec U,Vec *DRDU)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!DRDU) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(U,VEC_CLASSID,3);

  PetscStackPush("TS user DRDU function for sensitivity analysis");
  ierr = (*ts->drdufunction)(ts,t,U,DRDU,ts->costintegrandctx);CHKERRQ(ierr);
  PetscStackPop;
  PetscFunctionReturn(0);
}

/*@C
  TSComputeDRDPFunction - Deprecated, use TSGetQuadratureTS() then TSComputeRHSJacobianP()

  Level: deprecated

@*/
PetscErrorCode TSComputeDRDPFunction(TS ts,PetscReal t,Vec U,Vec *DRDP)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!DRDP) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(U,VEC_CLASSID,3);

  PetscStackPush("TS user DRDP function for sensitivity analysis");
  ierr = (*ts->drdpfunction)(ts,t,U,DRDP,ts->costintegrandctx);CHKERRQ(ierr);
  PetscStackPop;
  PetscFunctionReturn(0);
}

/*@C
  TSSetIHessianProduct - Sets the function that computes the vector-Hessian-vector product. The Hessian is the second-order derivative of F (IFunction) w.r.t. the state variable.

  Logically Collective on TS

  Input Parameters:
+ ts - TS context obtained from TSCreate()
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

.seealso:
@*/
PetscErrorCode TSSetIHessianProduct(TS ts,Vec *ihp1,PetscErrorCode (*ihessianproductfunc1)(TS,PetscReal,Vec,Vec*,Vec,Vec*,void*),
                                          Vec *ihp2,PetscErrorCode (*ihessianproductfunc2)(TS,PetscReal,Vec,Vec*,Vec,Vec*,void*),
                                          Vec *ihp3,PetscErrorCode (*ihessianproductfunc3)(TS,PetscReal,Vec,Vec*,Vec,Vec*,void*),
                                          Vec *ihp4,PetscErrorCode (*ihessianproductfunc4)(TS,PetscReal,Vec,Vec*,Vec,Vec*,void*),
                                    void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidPointer(ihp1,2);

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

  Collective on TS

  Input Parameters:
. ts   - The TS context obtained from TSCreate()

  Notes:
  TSComputeIHessianProductFunctionUU() is typically used for sensitivity implementation,
  so most users would not generally call this routine themselves.

  Level: developer

.seealso: TSSetIHessianProduct()
@*/
PetscErrorCode TSComputeIHessianProductFunctionUU(TS ts,PetscReal t,Vec U,Vec *Vl,Vec Vr,Vec *VHV)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!VHV) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(U,VEC_CLASSID,3);

  if (ts->ihessianproduct_fuu) {
    PetscStackPush("TS user IHessianProduct function 1 for sensitivity analysis");
    ierr = (*ts->ihessianproduct_fuu)(ts,t,U,Vl,Vr,VHV,ts->ihessianproductctx);CHKERRQ(ierr);
    PetscStackPop;
  }
  /* does not consider IMEX for now, so either IHessian or RHSHessian will be calculated, using the same output VHV */
  if (ts->rhshessianproduct_guu) {
    PetscInt nadj;
    ierr = TSComputeRHSHessianProductFunctionUU(ts,t,U,Vl,Vr,VHV);CHKERRQ(ierr);
    for (nadj=0; nadj<ts->numcost; nadj++) {
      ierr = VecScale(VHV[nadj],-1);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*@C
  TSComputeIHessianProductFunctionUP - Runs the user-defined vector-Hessian-vector product function for Fup.

  Collective on TS

  Input Parameters:
. ts   - The TS context obtained from TSCreate()

  Notes:
  TSComputeIHessianProductFunctionUP() is typically used for sensitivity implementation,
  so most users would not generally call this routine themselves.

  Level: developer

.seealso: TSSetIHessianProduct()
@*/
PetscErrorCode TSComputeIHessianProductFunctionUP(TS ts,PetscReal t,Vec U,Vec *Vl,Vec Vr,Vec *VHV)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!VHV) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(U,VEC_CLASSID,3);

  if (ts->ihessianproduct_fup) {
    PetscStackPush("TS user IHessianProduct function 2 for sensitivity analysis");
    ierr = (*ts->ihessianproduct_fup)(ts,t,U,Vl,Vr,VHV,ts->ihessianproductctx);CHKERRQ(ierr);
    PetscStackPop;
  }
  /* does not consider IMEX for now, so either IHessian or RHSHessian will be calculated, using the same output VHV */
  if (ts->rhshessianproduct_gup) {
    PetscInt nadj;
    ierr = TSComputeRHSHessianProductFunctionUP(ts,t,U,Vl,Vr,VHV);CHKERRQ(ierr);
    for (nadj=0; nadj<ts->numcost; nadj++) {
      ierr = VecScale(VHV[nadj],-1);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*@C
  TSComputeIHessianProductFunctionPU - Runs the user-defined vector-Hessian-vector product function for Fpu.

  Collective on TS

  Input Parameters:
. ts   - The TS context obtained from TSCreate()

  Notes:
  TSComputeIHessianProductFunctionPU() is typically used for sensitivity implementation,
  so most users would not generally call this routine themselves.

  Level: developer

.seealso: TSSetIHessianProduct()
@*/
PetscErrorCode TSComputeIHessianProductFunctionPU(TS ts,PetscReal t,Vec U,Vec *Vl,Vec Vr,Vec *VHV)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!VHV) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(U,VEC_CLASSID,3);

  if (ts->ihessianproduct_fpu) {
    PetscStackPush("TS user IHessianProduct function 3 for sensitivity analysis");
    ierr = (*ts->ihessianproduct_fpu)(ts,t,U,Vl,Vr,VHV,ts->ihessianproductctx);CHKERRQ(ierr);
    PetscStackPop;
  }
  /* does not consider IMEX for now, so either IHessian or RHSHessian will be calculated, using the same output VHV */
  if (ts->rhshessianproduct_gpu) {
    PetscInt nadj;
    ierr = TSComputeRHSHessianProductFunctionPU(ts,t,U,Vl,Vr,VHV);CHKERRQ(ierr);
    for (nadj=0; nadj<ts->numcost; nadj++) {
      ierr = VecScale(VHV[nadj],-1);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*@C
  TSComputeIHessianProductFunctionPP - Runs the user-defined vector-Hessian-vector product function for Fpp.

  Collective on TS

  Input Parameters:
. ts   - The TS context obtained from TSCreate()

  Notes:
  TSComputeIHessianProductFunctionPP() is typically used for sensitivity implementation,
  so most users would not generally call this routine themselves.

  Level: developer

.seealso: TSSetIHessianProduct()
@*/
PetscErrorCode TSComputeIHessianProductFunctionPP(TS ts,PetscReal t,Vec U,Vec *Vl,Vec Vr,Vec *VHV)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!VHV) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(U,VEC_CLASSID,3);

  if (ts->ihessianproduct_fpp) {
    PetscStackPush("TS user IHessianProduct function 3 for sensitivity analysis");
    ierr = (*ts->ihessianproduct_fpp)(ts,t,U,Vl,Vr,VHV,ts->ihessianproductctx);CHKERRQ(ierr);
    PetscStackPop;
  }
  /* does not consider IMEX for now, so either IHessian or RHSHessian will be calculated, using the same output VHV */
  if (ts->rhshessianproduct_gpp) {
    PetscInt nadj;
    ierr = TSComputeRHSHessianProductFunctionPP(ts,t,U,Vl,Vr,VHV);CHKERRQ(ierr);
    for (nadj=0; nadj<ts->numcost; nadj++) {
      ierr = VecScale(VHV[nadj],-1);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*@C
  TSSetRHSHessianProduct - Sets the function that computes the vector-Hessian-vector product. The Hessian is the second-order derivative of G (RHSFunction) w.r.t. the state variable.

  Logically Collective on TS

  Input Parameters:
+ ts - TS context obtained from TSCreate()
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

.seealso:
@*/
PetscErrorCode TSSetRHSHessianProduct(TS ts,Vec *rhshp1,PetscErrorCode (*rhshessianproductfunc1)(TS,PetscReal,Vec,Vec*,Vec,Vec*,void*),
                                          Vec *rhshp2,PetscErrorCode (*rhshessianproductfunc2)(TS,PetscReal,Vec,Vec*,Vec,Vec*,void*),
                                          Vec *rhshp3,PetscErrorCode (*rhshessianproductfunc3)(TS,PetscReal,Vec,Vec*,Vec,Vec*,void*),
                                          Vec *rhshp4,PetscErrorCode (*rhshessianproductfunc4)(TS,PetscReal,Vec,Vec*,Vec,Vec*,void*),
                                    void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidPointer(rhshp1,2);

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

  Collective on TS

  Input Parameters:
. ts   - The TS context obtained from TSCreate()

  Notes:
  TSComputeRHSHessianProductFunctionUU() is typically used for sensitivity implementation,
  so most users would not generally call this routine themselves.

  Level: developer

.seealso: TSSetRHSHessianProduct()
@*/
PetscErrorCode TSComputeRHSHessianProductFunctionUU(TS ts,PetscReal t,Vec U,Vec *Vl,Vec Vr,Vec *VHV)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!VHV) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(U,VEC_CLASSID,3);

  PetscStackPush("TS user RHSHessianProduct function 1 for sensitivity analysis");
  ierr = (*ts->rhshessianproduct_guu)(ts,t,U,Vl,Vr,VHV,ts->rhshessianproductctx);CHKERRQ(ierr);
  PetscStackPop;
  PetscFunctionReturn(0);
}

/*@C
  TSComputeRHSHessianProductFunctionUP - Runs the user-defined vector-Hessian-vector product function for Gup.

  Collective on TS

  Input Parameters:
. ts   - The TS context obtained from TSCreate()

  Notes:
  TSComputeRHSHessianProductFunctionUP() is typically used for sensitivity implementation,
  so most users would not generally call this routine themselves.

  Level: developer

.seealso: TSSetRHSHessianProduct()
@*/
PetscErrorCode TSComputeRHSHessianProductFunctionUP(TS ts,PetscReal t,Vec U,Vec *Vl,Vec Vr,Vec *VHV)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!VHV) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(U,VEC_CLASSID,3);

  PetscStackPush("TS user RHSHessianProduct function 2 for sensitivity analysis");
  ierr = (*ts->rhshessianproduct_gup)(ts,t,U,Vl,Vr,VHV,ts->rhshessianproductctx);CHKERRQ(ierr);
  PetscStackPop;
  PetscFunctionReturn(0);
}

/*@C
  TSComputeRHSHessianProductFunctionPU - Runs the user-defined vector-Hessian-vector product function for Gpu.

  Collective on TS

  Input Parameters:
. ts   - The TS context obtained from TSCreate()

  Notes:
  TSComputeRHSHessianProductFunctionPU() is typically used for sensitivity implementation,
  so most users would not generally call this routine themselves.

  Level: developer

.seealso: TSSetRHSHessianProduct()
@*/
PetscErrorCode TSComputeRHSHessianProductFunctionPU(TS ts,PetscReal t,Vec U,Vec *Vl,Vec Vr,Vec *VHV)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!VHV) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(U,VEC_CLASSID,3);

  PetscStackPush("TS user RHSHessianProduct function 3 for sensitivity analysis");
  ierr = (*ts->rhshessianproduct_gpu)(ts,t,U,Vl,Vr,VHV,ts->rhshessianproductctx);CHKERRQ(ierr);
  PetscStackPop;
  PetscFunctionReturn(0);
}

/*@C
  TSComputeRHSHessianProductFunctionPP - Runs the user-defined vector-Hessian-vector product function for Gpp.

  Collective on TS

  Input Parameters:
. ts   - The TS context obtained from TSCreate()

  Notes:
  TSComputeRHSHessianProductFunctionPP() is typically used for sensitivity implementation,
  so most users would not generally call this routine themselves.

  Level: developer

.seealso: TSSetRHSHessianProduct()
@*/
PetscErrorCode TSComputeRHSHessianProductFunctionPP(TS ts,PetscReal t,Vec U,Vec *Vl,Vec Vr,Vec *VHV)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!VHV) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(U,VEC_CLASSID,3);

  PetscStackPush("TS user RHSHessianProduct function 3 for sensitivity analysis");
  ierr = (*ts->rhshessianproduct_gpp)(ts,t,U,Vl,Vr,VHV,ts->rhshessianproductctx);CHKERRQ(ierr);
  PetscStackPop;
  PetscFunctionReturn(0);
}

/* --------------------------- Adjoint sensitivity ---------------------------*/

/*@
   TSSetCostGradients - Sets the initial value of the gradients of the cost function w.r.t. initial values and w.r.t. the problem parameters
      for use by the TSAdjoint routines.

   Logically Collective on TS

   Input Parameters:
+  ts - the TS context obtained from TSCreate()
.  numcost - number of gradients to be computed, this is the number of cost functions
.  lambda - gradients with respect to the initial condition variables, the dimension and parallel layout of these vectors is the same as the ODE solution vector
-  mu - gradients with respect to the parameters, the number of entries in these vectors is the same as the number of parameters

   Level: beginner

   Notes:
    the entries in these vectors must be correctly initialized with the values lamda_i = df/dy|finaltime  mu_i = df/dp|finaltime

   After TSAdjointSolve() is called the lamba and the mu contain the computed sensitivities

.seealso TSGetCostGradients()
@*/
PetscErrorCode TSSetCostGradients(TS ts,PetscInt numcost,Vec *lambda,Vec *mu)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidPointer(lambda,3);
  ts->vecs_sensi  = lambda;
  ts->vecs_sensip = mu;
  if (ts->numcost && ts->numcost!=numcost) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"The number of cost functions (2nd parameter of TSSetCostIntegrand()) is inconsistent with the one set by TSSetCostIntegrand");
  ts->numcost  = numcost;
  PetscFunctionReturn(0);
}

/*@
   TSGetCostGradients - Returns the gradients from the TSAdjointSolve()

   Not Collective, but Vec returned is parallel if TS is parallel

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

   Output Parameters:
+  lambda - vectors containing the gradients of the cost functions with respect to the ODE/DAE solution variables
-  mu - vectors containing the gradients of the cost functions with respect to the problem parameters

   Level: intermediate

.seealso: TSSetCostGradients()
@*/
PetscErrorCode TSGetCostGradients(TS ts,PetscInt *numcost,Vec **lambda,Vec **mu)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (numcost) *numcost = ts->numcost;
  if (lambda)  *lambda  = ts->vecs_sensi;
  if (mu)      *mu      = ts->vecs_sensip;
  PetscFunctionReturn(0);
}

/*@
   TSSetCostHessianProducts - Sets the initial value of the Hessian-vector products of the cost function w.r.t. initial values and w.r.t. the problem parameters
      for use by the TSAdjoint routines.

   Logically Collective on TS

   Input Parameters:
+  ts - the TS context obtained from TSCreate()
.  numcost - number of cost functions
.  lambda2 - Hessian-vector product with respect to the initial condition variables, the dimension and parallel layout of these vectors is the same as the ODE solution vector
.  mu2 - Hessian-vector product with respect to the parameters, the number of entries in these vectors is the same as the number of parameters
-  dir - the direction vector that are multiplied with the Hessian of the cost functions

   Level: beginner

   Notes: Hessian of the cost function is completely different from Hessian of the ODE/DAE system

   For second-order adjoint, one needs to call this function and then TSAdjointSetForward() before TSSolve().

   After TSAdjointSolve() is called, the lamba2 and the mu2 will contain the computed second-order adjoint sensitivities, and can be used to produce Hessian-vector product (not the full Hessian matrix). Users must provide a direction vector; it is usually generated by an optimization solver.

   Passing NULL for lambda2 disables the second-order calculation.
.seealso: TSAdjointSetForward()
@*/
PetscErrorCode TSSetCostHessianProducts(TS ts,PetscInt numcost,Vec *lambda2,Vec *mu2,Vec dir)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (ts->numcost && ts->numcost!=numcost) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"The number of cost functions (2nd parameter of TSSetCostIntegrand()) is inconsistent with the one set by TSSetCostIntegrand");
  ts->numcost       = numcost;
  ts->vecs_sensi2   = lambda2;
  ts->vecs_sensi2p  = mu2;
  ts->vec_dir       = dir;
  PetscFunctionReturn(0);
}

/*@
   TSGetCostHessianProducts - Returns the gradients from the TSAdjointSolve()

   Not Collective, but Vec returned is parallel if TS is parallel

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

   Output Parameters:
+  numcost - number of cost functions
.  lambda2 - Hessian-vector product with respect to the initial condition variables, the dimension and parallel layout of these vectors is the same as the ODE solution vector
.  mu2 - Hessian-vector product with respect to the parameters, the number of entries in these vectors is the same as the number of parameters
-  dir - the direction vector that are multiplied with the Hessian of the cost functions

   Level: intermediate

.seealso: TSSetCostHessianProducts()
@*/
PetscErrorCode TSGetCostHessianProducts(TS ts,PetscInt *numcost,Vec **lambda2,Vec **mu2, Vec *dir)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (numcost) *numcost = ts->numcost;
  if (lambda2) *lambda2 = ts->vecs_sensi2;
  if (mu2)     *mu2     = ts->vecs_sensi2p;
  if (dir)     *dir     = ts->vec_dir;
  PetscFunctionReturn(0);
}

/*@
  TSAdjointSetForward - Trigger the tangent linear solver and initialize the forward sensitivities

  Logically Collective on TS

  Input Parameters:
+  ts - the TS context obtained from TSCreate()
-  didp - the derivative of initial values w.r.t. parameters

  Level: intermediate

  Notes: When computing sensitivies w.r.t. initial condition, set didp to NULL so that the solver will take it as an identity matrix mathematically. TSAdjoint does not reset the tangent linear solver automatically, TSAdjointResetForward() should be called to reset the tangent linear solver.

.seealso: TSSetCostHessianProducts(), TSAdjointResetForward()
@*/
PetscErrorCode TSAdjointSetForward(TS ts,Mat didp)
{
  Mat            A;
  Vec            sp;
  PetscScalar    *xarr;
  PetscInt       lsize;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ts->forward_solve = PETSC_TRUE; /* turn on tangent linear mode */
  if (!ts->vecs_sensi2) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"Must call TSSetCostHessianProducts() first");
  if (!ts->vec_dir) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"Directional vector is missing. Call TSSetCostHessianProducts() to set it.");
  /* create a single-column dense matrix */
  ierr = VecGetLocalSize(ts->vec_sol,&lsize);CHKERRQ(ierr);
  ierr = MatCreateDense(PetscObjectComm((PetscObject)ts),lsize,PETSC_DECIDE,PETSC_DECIDE,1,NULL,&A);CHKERRQ(ierr);

  ierr = VecDuplicate(ts->vec_sol,&sp);CHKERRQ(ierr);
  ierr = MatDenseGetColumn(A,0,&xarr);CHKERRQ(ierr);
  ierr = VecPlaceArray(sp,xarr);CHKERRQ(ierr);
  if (ts->vecs_sensi2p) { /* tangent linear variable initialized as 2*dIdP*dir */
    if (didp) {
      ierr = MatMult(didp,ts->vec_dir,sp);CHKERRQ(ierr);
      ierr = VecScale(sp,2.);CHKERRQ(ierr);
    } else {
      ierr = VecZeroEntries(sp);CHKERRQ(ierr);
    }
  } else { /* tangent linear variable initialized as dir */
    ierr = VecCopy(ts->vec_dir,sp);CHKERRQ(ierr);
  }
  ierr = VecResetArray(sp);CHKERRQ(ierr);
  ierr = MatDenseRestoreColumn(A,&xarr);CHKERRQ(ierr);
  ierr = VecDestroy(&sp);CHKERRQ(ierr);

  ierr = TSForwardSetInitialSensitivities(ts,A);CHKERRQ(ierr); /* if didp is NULL, identity matrix is assumed */

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  TSAdjointResetForward - Reset the tangent linear solver and destroy the tangent linear context

  Logically Collective on TS

  Input Parameters:
.  ts - the TS context obtained from TSCreate()

  Level: intermediate

.seealso: TSAdjointSetForward()
@*/
PetscErrorCode TSAdjointResetForward(TS ts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ts->forward_solve = PETSC_FALSE; /* turn off tangent linear mode */
  ierr = TSForwardReset(ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   TSAdjointSetUp - Sets up the internal data structures for the later use
   of an adjoint solver

   Collective on TS

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

   Level: advanced

.seealso: TSCreate(), TSAdjointStep(), TSSetCostGradients()
@*/
PetscErrorCode TSAdjointSetUp(TS ts)
{
  TSTrajectory     tj;
  PetscBool        match;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (ts->adjointsetupcalled) PetscFunctionReturn(0);
  if (!ts->vecs_sensi) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_WRONGSTATE,"Must call TSSetCostGradients() first");
  if (ts->vecs_sensip && !ts->Jacp && !ts->Jacprhs) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_WRONGSTATE,"Must call TSSetRHSJacobianP() or TSSetIJacobianP() first");
  ierr = TSGetTrajectory(ts,&tj);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)tj,TSTRAJECTORYBASIC,&match);CHKERRQ(ierr);
  if (match) {
    PetscBool solution_only;
    ierr = TSTrajectoryGetSolutionOnly(tj,&solution_only);CHKERRQ(ierr);
    if (solution_only) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"TSAdjoint cannot use the solution-only mode when choosing the Basic TSTrajectory type. Turn it off with -ts_trajectory_solution_only 0");
  }
  ierr = TSTrajectorySetUseHistory(tj,PETSC_FALSE);CHKERRQ(ierr); /* not use TSHistory */

  if (ts->quadraturets) { /* if there is integral in the cost function */
    ierr = VecDuplicate(ts->vecs_sensi[0],&ts->vec_drdu_col);CHKERRQ(ierr);
    if (ts->vecs_sensip) {
      ierr = VecDuplicate(ts->vecs_sensip[0],&ts->vec_drdp_col);CHKERRQ(ierr);
    }
  }

  if (ts->ops->adjointsetup) {
    ierr = (*ts->ops->adjointsetup)(ts);CHKERRQ(ierr);
  }
  ts->adjointsetupcalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
   TSAdjointReset - Resets a TSAdjoint context and removes any allocated Vecs and Mats.

   Collective on TS

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

   Level: beginner

.seealso: TSCreate(), TSAdjointSetUp(), TSADestroy()
@*/
PetscErrorCode TSAdjointReset(TS ts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (ts->ops->adjointreset) {
    ierr = (*ts->ops->adjointreset)(ts);CHKERRQ(ierr);
  }
  if (ts->quadraturets) { /* if there is integral in the cost function */
    ierr = VecDestroy(&ts->vec_drdu_col);CHKERRQ(ierr);
    if (ts->vecs_sensip) {
      ierr = VecDestroy(&ts->vec_drdp_col);CHKERRQ(ierr);
    }
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

   Logically Collective on TS

   Input Parameters:
+  ts - the TS context obtained from TSCreate()
-  steps - number of steps to use

   Level: intermediate

   Notes:
    Normally one does not call this and TSAdjointSolve() integrates back to the original timestep. One can call this
          so as to integrate back to less than the original timestep

.seealso: TSSetExactFinalTime()
@*/
PetscErrorCode TSAdjointSetSteps(TS ts,PetscInt steps)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidLogicalCollectiveInt(ts,steps,2);
  if (steps < 0) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_OUTOFRANGE,"Cannot step back a negative number of steps");
  if (steps > ts->steps) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_OUTOFRANGE,"Cannot step back more than the total number of forward steps");
  ts->adjoint_max_steps = steps;
  PetscFunctionReturn(0);
}

/*@C
  TSAdjointSetRHSJacobian - Deprecated, use TSSetRHSJacobianP()

  Level: deprecated

@*/
PetscErrorCode TSAdjointSetRHSJacobian(TS ts,Mat Amat,PetscErrorCode (*func)(TS,PetscReal,Vec,Mat,void*),void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID,1);
  PetscValidHeaderSpecific(Amat,MAT_CLASSID,2);

  ts->rhsjacobianp    = func;
  ts->rhsjacobianpctx = ctx;
  if (Amat) {
    ierr = PetscObjectReference((PetscObject)Amat);CHKERRQ(ierr);
    ierr = MatDestroy(&ts->Jacp);CHKERRQ(ierr);
    ts->Jacp = Amat;
  }
  PetscFunctionReturn(0);
}

/*@C
  TSAdjointComputeRHSJacobian - Deprecated, use TSComputeRHSJacobianP()

  Level: deprecated

@*/
PetscErrorCode TSAdjointComputeRHSJacobian(TS ts,PetscReal t,Vec U,Mat Amat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(U,VEC_CLASSID,3);
  PetscValidPointer(Amat,4);

  PetscStackPush("TS user JacobianP function for sensitivity analysis");
  ierr = (*ts->rhsjacobianp)(ts,t,U,Amat,ts->rhsjacobianpctx);CHKERRQ(ierr);
  PetscStackPop;
  PetscFunctionReturn(0);
}

/*@
  TSAdjointComputeDRDYFunction - Deprecated, use TSGetQuadratureTS() then TSComputeRHSJacobian()

  Level: deprecated

@*/
PetscErrorCode TSAdjointComputeDRDYFunction(TS ts,PetscReal t,Vec U,Vec *DRDU)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(U,VEC_CLASSID,3);

  PetscStackPush("TS user DRDY function for sensitivity analysis");
  ierr = (*ts->drdufunction)(ts,t,U,DRDU,ts->costintegrandctx);CHKERRQ(ierr);
  PetscStackPop;
  PetscFunctionReturn(0);
}

/*@
  TSAdjointComputeDRDPFunction - Deprecated, use TSGetQuadratureTS() then TSComputeRHSJacobianP()

  Level: deprecated

@*/
PetscErrorCode TSAdjointComputeDRDPFunction(TS ts,PetscReal t,Vec U,Vec *DRDP)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(U,VEC_CLASSID,3);

  PetscStackPush("TS user DRDP function for sensitivity analysis");
  ierr = (*ts->drdpfunction)(ts,t,U,DRDP,ts->costintegrandctx);CHKERRQ(ierr);
  PetscStackPop;
  PetscFunctionReturn(0);
}

/*@C
   TSAdjointMonitorSensi - monitors the first lambda sensitivity

   Level: intermediate

.seealso: TSAdjointMonitorSet()
@*/
PetscErrorCode TSAdjointMonitorSensi(TS ts,PetscInt step,PetscReal ptime,Vec v,PetscInt numcost,Vec *lambda,Vec *mu,PetscViewerAndFormat *vf)
{
  PetscErrorCode ierr;
  PetscViewer    viewer = vf->viewer;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,8);
  ierr = PetscViewerPushFormat(viewer,vf->format);CHKERRQ(ierr);
  ierr = VecView(lambda[0],viewer);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TSAdjointMonitorSetFromOptions - Sets a monitor function and viewer appropriate for the type indicated by the user

   Collective on TS

   Input Parameters:
+  ts - TS object you wish to monitor
.  name - the monitor type one is seeking
.  help - message indicating what monitoring is done
.  manual - manual page for the monitor
.  monitor - the monitor function
-  monitorsetup - a function that is called once ONLY if the user selected this monitor that may set additional features of the TS or PetscViewer objects

   Level: developer

.seealso: PetscOptionsGetViewer(), PetscOptionsGetReal(), PetscOptionsHasName(), PetscOptionsGetString(),
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool()
          PetscOptionsInt(), PetscOptionsString(), PetscOptionsReal(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
@*/
PetscErrorCode TSAdjointMonitorSetFromOptions(TS ts,const char name[],const char help[], const char manual[],PetscErrorCode (*monitor)(TS,PetscInt,PetscReal,Vec,PetscInt,Vec*,Vec*,PetscViewerAndFormat*),PetscErrorCode (*monitorsetup)(TS,PetscViewerAndFormat*))
{
  PetscErrorCode    ierr;
  PetscViewer       viewer;
  PetscViewerFormat format;
  PetscBool         flg;

  PetscFunctionBegin;
  ierr = PetscOptionsGetViewer(PetscObjectComm((PetscObject)ts),((PetscObject) ts)->options,((PetscObject)ts)->prefix,name,&viewer,&format,&flg);CHKERRQ(ierr);
  if (flg) {
    PetscViewerAndFormat *vf;
    ierr = PetscViewerAndFormatCreate(viewer,format,&vf);CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject)viewer);CHKERRQ(ierr);
    if (monitorsetup) {
      ierr = (*monitorsetup)(ts,vf);CHKERRQ(ierr);
    }
    ierr = TSAdjointMonitorSet(ts,(PetscErrorCode (*)(TS,PetscInt,PetscReal,Vec,PetscInt,Vec*,Vec*,void*))monitor,vf,(PetscErrorCode (*)(void**))PetscViewerAndFormatDestroy);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   TSAdjointMonitorSet - Sets an ADDITIONAL function that is to be used at every
   timestep to display the iteration's  progress.

   Logically Collective on TS

   Input Parameters:
+  ts - the TS context obtained from TSCreate()
.  adjointmonitor - monitoring routine
.  adjointmctx - [optional] user-defined context for private data for the
             monitor routine (use NULL if no context is desired)
-  adjointmonitordestroy - [optional] routine that frees monitor context
          (may be NULL)

   Calling sequence of monitor:
$    int adjointmonitor(TS ts,PetscInt steps,PetscReal time,Vec u,PetscInt numcost,Vec *lambda, Vec *mu,void *adjointmctx)

+    ts - the TS context
.    steps - iteration number (after the final time step the monitor routine is called with a step of -1, this is at the final time which may have
                               been interpolated to)
.    time - current time
.    u - current iterate
.    numcost - number of cost functionos
.    lambda - sensitivities to initial conditions
.    mu - sensitivities to parameters
-    adjointmctx - [optional] adjoint monitoring context

   Notes:
   This routine adds an additional monitor to the list of monitors that
   already has been loaded.

   Fortran Notes:
    Only a single monitor function can be set for each TS object

   Level: intermediate

.seealso: TSAdjointMonitorCancel()
@*/
PetscErrorCode TSAdjointMonitorSet(TS ts,PetscErrorCode (*adjointmonitor)(TS,PetscInt,PetscReal,Vec,PetscInt,Vec*,Vec*,void*),void *adjointmctx,PetscErrorCode (*adjointmdestroy)(void**))
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscBool      identical;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  for (i=0; i<ts->numbermonitors;i++) {
    ierr = PetscMonitorCompare((PetscErrorCode (*)(void))adjointmonitor,adjointmctx,adjointmdestroy,(PetscErrorCode (*)(void))ts->adjointmonitor[i],ts->adjointmonitorcontext[i],ts->adjointmonitordestroy[i],&identical);CHKERRQ(ierr);
    if (identical) PetscFunctionReturn(0);
  }
  if (ts->numberadjointmonitors >= MAXTSMONITORS) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Too many adjoint monitors set");
  ts->adjointmonitor[ts->numberadjointmonitors]          = adjointmonitor;
  ts->adjointmonitordestroy[ts->numberadjointmonitors]   = adjointmdestroy;
  ts->adjointmonitorcontext[ts->numberadjointmonitors++] = (void*)adjointmctx;
  PetscFunctionReturn(0);
}

/*@C
   TSAdjointMonitorCancel - Clears all the adjoint monitors that have been set on a time-step object.

   Logically Collective on TS

   Input Parameters:
.  ts - the TS context obtained from TSCreate()

   Notes:
   There is no way to remove a single, specific monitor.

   Level: intermediate

.seealso: TSAdjointMonitorSet()
@*/
PetscErrorCode TSAdjointMonitorCancel(TS ts)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  for (i=0; i<ts->numberadjointmonitors; i++) {
    if (ts->adjointmonitordestroy[i]) {
      ierr = (*ts->adjointmonitordestroy[i])(&ts->adjointmonitorcontext[i]);CHKERRQ(ierr);
    }
  }
  ts->numberadjointmonitors = 0;
  PetscFunctionReturn(0);
}

/*@C
   TSAdjointMonitorDefault - the default monitor of adjoint computations

   Level: intermediate

.seealso: TSAdjointMonitorSet()
@*/
PetscErrorCode TSAdjointMonitorDefault(TS ts,PetscInt step,PetscReal ptime,Vec v,PetscInt numcost,Vec *lambda,Vec *mu,PetscViewerAndFormat *vf)
{
  PetscErrorCode ierr;
  PetscViewer    viewer = vf->viewer;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,8);
  ierr = PetscViewerPushFormat(viewer,vf->format);CHKERRQ(ierr);
  ierr = PetscViewerASCIIAddTab(viewer,((PetscObject)ts)->tablevel);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"%D TS dt %g time %g%s",step,(double)ts->time_step,(double)ptime,ts->steprollback ? " (r)\n" : "\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIISubtractTab(viewer,((PetscObject)ts)->tablevel);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TSAdjointMonitorDrawSensi - Monitors progress of the adjoint TS solvers by calling
   VecView() for the sensitivities to initial states at each timestep

   Collective on TS

   Input Parameters:
+  ts - the TS context
.  step - current time-step
.  ptime - current time
.  u - current state
.  numcost - number of cost functions
.  lambda - sensitivities to initial conditions
.  mu - sensitivities to parameters
-  dummy - either a viewer or NULL

   Level: intermediate

.seealso: TSAdjointMonitorSet(), TSAdjointMonitorDefault(), VecView()
@*/
PetscErrorCode TSAdjointMonitorDrawSensi(TS ts,PetscInt step,PetscReal ptime,Vec u,PetscInt numcost,Vec *lambda,Vec *mu,void *dummy)
{
  PetscErrorCode   ierr;
  TSMonitorDrawCtx ictx = (TSMonitorDrawCtx)dummy;
  PetscDraw        draw;
  PetscReal        xl,yl,xr,yr,h;
  char             time[32];

  PetscFunctionBegin;
  if (!(((ictx->howoften > 0) && (!(step % ictx->howoften))) || ((ictx->howoften == -1) && ts->reason))) PetscFunctionReturn(0);

  ierr = VecView(lambda[0],ictx->viewer);CHKERRQ(ierr);
  ierr = PetscViewerDrawGetDraw(ictx->viewer,0,&draw);CHKERRQ(ierr);
  ierr = PetscSNPrintf(time,32,"Timestep %d Time %g",(int)step,(double)ptime);CHKERRQ(ierr);
  ierr = PetscDrawGetCoordinates(draw,&xl,&yl,&xr,&yr);CHKERRQ(ierr);
  h    = yl + .95*(yr - yl);
  ierr = PetscDrawStringCentered(draw,.5*(xl+xr),h,PETSC_DRAW_BLACK,time);CHKERRQ(ierr);
  ierr = PetscDrawFlush(draw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   TSAdjointSetFromOptions - Sets various TSAdjoint parameters from user options.

   Collective on TSAdjoint

   Input Parameter:
.  ts - the TS context

   Options Database Keys:
+  -ts_adjoint_solve <yes,no> After solving the ODE/DAE solve the adjoint problem (requires -ts_save_trajectory)
.  -ts_adjoint_monitor - print information at each adjoint time step
-  -ts_adjoint_monitor_draw_sensi - monitor the sensitivity of the first cost function wrt initial conditions (lambda[0]) graphically

   Level: developer

   Notes:
    This is not normally called directly by users

.seealso: TSSetSaveTrajectory(), TSTrajectorySetUp()
*/
PetscErrorCode TSAdjointSetFromOptions(PetscOptionItems *PetscOptionsObject,TS ts)
{
  PetscBool      tflg,opt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,2);
  ierr = PetscOptionsHead(PetscOptionsObject,"TS Adjoint options");CHKERRQ(ierr);
  tflg = ts->adjoint_solve ? PETSC_TRUE : PETSC_FALSE;
  ierr = PetscOptionsBool("-ts_adjoint_solve","Solve the adjoint problem immediately after solving the forward problem","",tflg,&tflg,&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = TSSetSaveTrajectory(ts);CHKERRQ(ierr);
    ts->adjoint_solve = tflg;
  }
  ierr = TSAdjointMonitorSetFromOptions(ts,"-ts_adjoint_monitor","Monitor adjoint timestep size","TSAdjointMonitorDefault",TSAdjointMonitorDefault,NULL);CHKERRQ(ierr);
  ierr = TSAdjointMonitorSetFromOptions(ts,"-ts_adjoint_monitor_sensi","Monitor sensitivity in the adjoint computation","TSAdjointMonitorSensi",TSAdjointMonitorSensi,NULL);CHKERRQ(ierr);
  opt  = PETSC_FALSE;
  ierr = PetscOptionsName("-ts_adjoint_monitor_draw_sensi","Monitor adjoint sensitivities (lambda only) graphically","TSAdjointMonitorDrawSensi",&opt);CHKERRQ(ierr);
  if (opt) {
    TSMonitorDrawCtx ctx;
    PetscInt         howoften = 1;

    ierr = PetscOptionsInt("-ts_adjoint_monitor_draw_sensi","Monitor adjoint sensitivities (lambda only) graphically","TSAdjointMonitorDrawSensi",howoften,&howoften,NULL);CHKERRQ(ierr);
    ierr = TSMonitorDrawCtxCreate(PetscObjectComm((PetscObject)ts),NULL,NULL,PETSC_DECIDE,PETSC_DECIDE,300,300,howoften,&ctx);CHKERRQ(ierr);
    ierr = TSAdjointMonitorSet(ts,TSAdjointMonitorDrawSensi,ctx,(PetscErrorCode (*)(void**))TSMonitorDrawCtxDestroy);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
   TSAdjointStep - Steps one time step backward in the adjoint run

   Collective on TS

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

   Level: intermediate

.seealso: TSAdjointSetUp(), TSAdjointSolve()
@*/
PetscErrorCode TSAdjointStep(TS ts)
{
  DM               dm;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = TSAdjointSetUp(ts);CHKERRQ(ierr);
  ts->steps--; /* must decrease the step index before the adjoint step is taken. */

  ts->reason = TS_CONVERGED_ITERATING;
  ts->ptime_prev = ts->ptime;
  if (!ts->ops->adjointstep) SETERRQ1(PetscObjectComm((PetscObject)ts),PETSC_ERR_NOT_CONVERGED,"TSStep has failed because the adjoint of  %s has not been implemented, try other time stepping methods for adjoint sensitivity analysis",((PetscObject)ts)->type_name);
  ierr = PetscLogEventBegin(TS_AdjointStep,ts,0,0,0);CHKERRQ(ierr);
  ierr = (*ts->ops->adjointstep)(ts);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(TS_AdjointStep,ts,0,0,0);CHKERRQ(ierr);
  ts->adjoint_steps++;

  if (ts->reason < 0) {
    if (ts->errorifstepfailed) SETERRQ1(PetscObjectComm((PetscObject)ts),PETSC_ERR_NOT_CONVERGED,"TSAdjointStep has failed due to %s",TSConvergedReasons[ts->reason]);
  } else if (!ts->reason) {
    if (ts->adjoint_steps >= ts->adjoint_max_steps) ts->reason = TS_CONVERGED_ITS;
  }
  PetscFunctionReturn(0);
}

/*@
   TSAdjointSolve - Solves the discrete ajoint problem for an ODE/DAE

   Collective on TS

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

   Options Database:
. -ts_adjoint_view_solution <viewerinfo> - views the first gradient with respect to the initial values

   Level: intermediate

   Notes:
   This must be called after a call to TSSolve() that solves the forward problem

   By default this will integrate back to the initial time, one can use TSAdjointSetSteps() to step back to a later time

.seealso: TSCreate(), TSSetCostGradients(), TSSetSolution(), TSAdjointStep()
@*/
PetscErrorCode TSAdjointSolve(TS ts)
{
  static PetscBool cite = PETSC_FALSE;
#if defined(TSADJOINT_STAGE)
  PetscLogStage  adjoint_stage;
#endif
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = PetscCitationsRegister("@article{tsadjointpaper,\n"
                                "  title         = {{PETSc TSAdjoint: a discrete adjoint ODE solver for first-order and second-order sensitivity analysis}},\n"
                                "  author        = {Zhang, Hong and Constantinescu, Emil M.  and Smith, Barry F.},\n"
                                "  journal       = {arXiv e-preprints},\n"
                                "  eprint        = {1912.07696},\n"
                                "  archivePrefix = {arXiv},\n"
                                "  year          = {2019}\n}\n",&cite);CHKERRQ(ierr);
#if defined(TSADJOINT_STAGE)
  ierr = PetscLogStageRegister("TSAdjoint",&adjoint_stage);CHKERRQ(ierr);
  ierr = PetscLogStagePush(adjoint_stage);CHKERRQ(ierr);
#endif
  ierr = TSAdjointSetUp(ts);CHKERRQ(ierr);

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
    ierr = TSTrajectoryGet(ts->trajectory,ts,ts->steps,&ts->ptime);CHKERRQ(ierr);
    ierr = TSAdjointMonitor(ts,ts->steps,ts->ptime,ts->vec_sol,ts->numcost,ts->vecs_sensi,ts->vecs_sensip);CHKERRQ(ierr);
    ierr = TSAdjointEventHandler(ts);CHKERRQ(ierr);
    ierr = TSAdjointStep(ts);CHKERRQ(ierr);
    if ((ts->vec_costintegral || ts->quadraturets) && !ts->costintegralfwd) {
      ierr = TSAdjointCostIntegral(ts);CHKERRQ(ierr);
    }
  }
  ierr = TSTrajectoryGet(ts->trajectory,ts,ts->steps,&ts->ptime);CHKERRQ(ierr);
  ierr = TSAdjointMonitor(ts,ts->steps,ts->ptime,ts->vec_sol,ts->numcost,ts->vecs_sensi,ts->vecs_sensip);CHKERRQ(ierr);
  ts->solvetime = ts->ptime;
  ierr = TSTrajectoryViewFromOptions(ts->trajectory,NULL,"-ts_trajectory_view");CHKERRQ(ierr);
  ierr = VecViewFromOptions(ts->vecs_sensi[0],(PetscObject) ts, "-ts_adjoint_view_solution");CHKERRQ(ierr);
  ts->adjoint_max_steps = 0;
#if defined(TSADJOINT_STAGE)
  ierr = PetscLogStagePop();CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

/*@C
   TSAdjointMonitor - Runs all user-provided adjoint monitor routines set using TSAdjointMonitorSet()

   Collective on TS

   Input Parameters:
+  ts - time stepping context obtained from TSCreate()
.  step - step number that has just completed
.  ptime - model time of the state
.  u - state at the current model time
.  numcost - number of cost functions (dimension of lambda  or mu)
.  lambda - vectors containing the gradients of the cost functions with respect to the ODE/DAE solution variables
-  mu - vectors containing the gradients of the cost functions with respect to the problem parameters

   Notes:
   TSAdjointMonitor() is typically used automatically within the time stepping implementations.
   Users would almost never call this routine directly.

   Level: developer

@*/
PetscErrorCode TSAdjointMonitor(TS ts,PetscInt step,PetscReal ptime,Vec u,PetscInt numcost,Vec *lambda, Vec *mu)
{
  PetscErrorCode ierr;
  PetscInt       i,n = ts->numberadjointmonitors;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(u,VEC_CLASSID,4);
  ierr = VecLockReadPush(u);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    ierr = (*ts->adjointmonitor[i])(ts,step,ptime,u,numcost,lambda,mu,ts->adjointmonitorcontext[i]);CHKERRQ(ierr);
  }
  ierr = VecLockReadPop(u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
 TSAdjointCostIntegral - Evaluate the cost integral in the adjoint run.

 Collective on TS

 Input Arguments:
 .  ts - time stepping context

 Level: advanced

 Notes:
 This function cannot be called until TSAdjointStep() has been completed.

 .seealso: TSAdjointSolve(), TSAdjointStep
 @*/
PetscErrorCode TSAdjointCostIntegral(TS ts)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (!ts->ops->adjointintegral) SETERRQ1(PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"%s does not provide integral evaluation in the adjoint run",((PetscObject)ts)->type_name);
  ierr = (*ts->ops->adjointintegral)(ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------ Forward (tangent linear) sensitivity  ------------------*/

/*@
  TSForwardSetUp - Sets up the internal data structures for the later use
  of forward sensitivity analysis

  Collective on TS

  Input Parameter:
. ts - the TS context obtained from TSCreate()

  Level: advanced

.seealso: TSCreate(), TSDestroy(), TSSetUp()
@*/
PetscErrorCode TSForwardSetUp(TS ts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (ts->forwardsetupcalled) PetscFunctionReturn(0);
  if (ts->ops->forwardsetup) {
    ierr = (*ts->ops->forwardsetup)(ts);CHKERRQ(ierr);
  }
  ierr = VecDuplicate(ts->vec_sol,&ts->vec_sensip_col);CHKERRQ(ierr);
  ts->forwardsetupcalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
  TSForwardReset - Reset the internal data structures used by forward sensitivity analysis

  Collective on TS

  Input Parameter:
. ts - the TS context obtained from TSCreate()

  Level: advanced

.seealso: TSCreate(), TSDestroy(), TSForwardSetUp()
@*/
PetscErrorCode TSForwardReset(TS ts)
{
  TS             quadts = ts->quadraturets;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (ts->ops->forwardreset) {
    ierr = (*ts->ops->forwardreset)(ts);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&ts->mat_sensip);CHKERRQ(ierr);
  if (quadts) {
    ierr = MatDestroy(&quadts->mat_sensip);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&ts->vec_sensip_col);CHKERRQ(ierr);
  ts->forward_solve      = PETSC_FALSE;
  ts->forwardsetupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@
  TSForwardSetIntegralGradients - Set the vectors holding forward sensitivities of the integral term.

  Input Parameters:
+ ts- the TS context obtained from TSCreate()
. numfwdint- number of integrals
- vp = the vectors containing the gradients for each integral w.r.t. parameters

  Level: deprecated

.seealso: TSForwardGetSensitivities(), TSForwardSetIntegralGradients(), TSForwardGetIntegralGradients(), TSForwardStep()
@*/
PetscErrorCode TSForwardSetIntegralGradients(TS ts,PetscInt numfwdint,Vec *vp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (ts->numcost && ts->numcost!=numfwdint) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"The number of cost functions (2nd parameter of TSSetCostIntegrand()) is inconsistent with the one set by TSSetCostIntegrand()");
  if (!ts->numcost) ts->numcost = numfwdint;

  ts->vecs_integral_sensip = vp;
  PetscFunctionReturn(0);
}

/*@
  TSForwardGetIntegralGradients - Returns the forward sensitivities ofthe integral term.

  Input Parameter:
. ts- the TS context obtained from TSCreate()

  Output Parameter:
. vp = the vectors containing the gradients for each integral w.r.t. parameters

  Level: deprecated

.seealso: TSForwardSetSensitivities(), TSForwardSetIntegralGradients(), TSForwardGetIntegralGradients(), TSForwardStep()
@*/
PetscErrorCode TSForwardGetIntegralGradients(TS ts,PetscInt *numfwdint,Vec **vp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidPointer(vp,3);
  if (numfwdint) *numfwdint = ts->numcost;
  if (vp) *vp = ts->vecs_integral_sensip;
  PetscFunctionReturn(0);
}

/*@
  TSForwardStep - Compute the forward sensitivity for one time step.

  Collective on TS

  Input Arguments:
. ts - time stepping context

  Level: advanced

  Notes:
  This function cannot be called until TSStep() has been completed.

.seealso: TSForwardSetSensitivities(), TSForwardGetSensitivities(), TSForwardSetIntegralGradients(), TSForwardGetIntegralGradients(), TSForwardSetUp()
@*/
PetscErrorCode TSForwardStep(TS ts)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (!ts->ops->forwardstep) SETERRQ1(PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"%s does not provide forward sensitivity analysis",((PetscObject)ts)->type_name);
  ierr = PetscLogEventBegin(TS_ForwardStep,ts,0,0,0);CHKERRQ(ierr);
  ierr = (*ts->ops->forwardstep)(ts);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(TS_ForwardStep,ts,0,0,0);CHKERRQ(ierr);
  if (ts->reason < 0 && ts->errorifstepfailed) SETERRQ1(PetscObjectComm((PetscObject)ts),PETSC_ERR_NOT_CONVERGED,"TSFowardStep has failed due to %s",TSConvergedReasons[ts->reason]);
  PetscFunctionReturn(0);
}

/*@
  TSForwardSetSensitivities - Sets the initial value of the trajectory sensitivities of solution  w.r.t. the problem parameters and initial values.

  Logically Collective on TS

  Input Parameters:
+ ts - the TS context obtained from TSCreate()
. nump - number of parameters
- Smat - sensitivities with respect to the parameters, the number of entries in these vectors is the same as the number of parameters

  Level: beginner

  Notes:
  Forward sensitivity is also called 'trajectory sensitivity' in some fields such as power systems.
  This function turns on a flag to trigger TSSolve() to compute forward sensitivities automatically.
  You must call this function before TSSolve().
  The entries in the sensitivity matrix must be correctly initialized with the values S = dy/dp|startingtime.

.seealso: TSForwardGetSensitivities(), TSForwardSetIntegralGradients(), TSForwardGetIntegralGradients(), TSForwardStep()
@*/
PetscErrorCode TSForwardSetSensitivities(TS ts,PetscInt nump,Mat Smat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(Smat,MAT_CLASSID,3);
  ts->forward_solve  = PETSC_TRUE;
  if (nump == PETSC_DEFAULT) {
    ierr = MatGetSize(Smat,NULL,&ts->num_parameters);CHKERRQ(ierr);
  } else ts->num_parameters = nump;
  ierr = PetscObjectReference((PetscObject)Smat);CHKERRQ(ierr);
  ierr = MatDestroy(&ts->mat_sensip);CHKERRQ(ierr);
  ts->mat_sensip = Smat;
  PetscFunctionReturn(0);
}

/*@
  TSForwardGetSensitivities - Returns the trajectory sensitivities

  Not Collective, but Vec returned is parallel if TS is parallel

  Output Parameters:
+ ts - the TS context obtained from TSCreate()
. nump - number of parameters
- Smat - sensitivities with respect to the parameters, the number of entries in these vectors is the same as the number of parameters

  Level: intermediate

.seealso: TSForwardSetSensitivities(), TSForwardSetIntegralGradients(), TSForwardGetIntegralGradients(), TSForwardStep()
@*/
PetscErrorCode TSForwardGetSensitivities(TS ts,PetscInt *nump,Mat *Smat)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (nump) *nump = ts->num_parameters;
  if (Smat) *Smat = ts->mat_sensip;
  PetscFunctionReturn(0);
}

/*@
   TSForwardCostIntegral - Evaluate the cost integral in the forward run.

   Collective on TS

   Input Arguments:
.  ts - time stepping context

   Level: advanced

   Notes:
   This function cannot be called until TSStep() has been completed.

.seealso: TSSolve(), TSAdjointCostIntegral()
@*/
PetscErrorCode TSForwardCostIntegral(TS ts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (!ts->ops->forwardintegral) SETERRQ1(PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"%s does not provide integral evaluation in the forward run",((PetscObject)ts)->type_name);
  ierr = (*ts->ops->forwardintegral)(ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  TSForwardSetInitialSensitivities - Set initial values for tangent linear sensitivities

  Collective on TS

  Input Parameters:
+ ts - the TS context obtained from TSCreate()
- didp - parametric sensitivities of the initial condition

  Level: intermediate

  Notes: TSSolve() allows users to pass the initial solution directly to TS. But the tangent linear variables cannot be initialized in this way. This function is used to set initial values for tangent linear variables.

.seealso: TSForwardSetSensitivities()
@*/
PetscErrorCode TSForwardSetInitialSensitivities(TS ts,Mat didp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(didp,MAT_CLASSID,2);
  if (!ts->mat_sensip) {
    ierr = TSForwardSetSensitivities(ts,PETSC_DEFAULT,didp);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
   TSForwardGetStages - Get the number of stages and the tangent linear sensitivities at the intermediate stages

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

   Output Parameters:
+  ns - number of stages
-  S - tangent linear sensitivities at the intermediate stages

   Level: advanced

@*/
PetscErrorCode TSForwardGetStages(TS ts,PetscInt *ns,Mat **S)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID,1);

  if (!ts->ops->getstages) *S=NULL;
  else {
    ierr = (*ts->ops->forwardgetstages)(ts,ns,S);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
   TSCreateQuadratureTS - Create a sub-TS that evaluates integrals over time

   Input Parameters:
+  ts - the TS context obtained from TSCreate()
-  fwd - flag indicating whether to evaluate cost integral in the forward run or the adjoint run

   Output Parameters:
.  quadts - the child TS context

   Level: intermediate

.seealso: TSGetQuadratureTS()
@*/
PetscErrorCode TSCreateQuadratureTS(TS ts,PetscBool fwd,TS *quadts)
{
  char prefix[128];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidPointer(quadts,3);
  ierr = TSDestroy(&ts->quadraturets);CHKERRQ(ierr);
  ierr = TSCreate(PetscObjectComm((PetscObject)ts),&ts->quadraturets);CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject)ts->quadraturets,(PetscObject)ts,1);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)ts,(PetscObject)ts->quadraturets);CHKERRQ(ierr);
  ierr = PetscSNPrintf(prefix,sizeof(prefix),"%squad_",((PetscObject)ts)->prefix ? ((PetscObject)ts)->prefix : "");CHKERRQ(ierr);
  ierr = TSSetOptionsPrefix(ts->quadraturets,prefix);CHKERRQ(ierr);
  *quadts = ts->quadraturets;

  if (ts->numcost) {
    ierr = VecCreateSeq(PETSC_COMM_SELF,ts->numcost,&(*quadts)->vec_sol);CHKERRQ(ierr);
  } else {
    ierr = VecCreateSeq(PETSC_COMM_SELF,1,&(*quadts)->vec_sol);CHKERRQ(ierr);
  }
  ts->costintegralfwd = fwd;
  PetscFunctionReturn(0);
}

/*@
   TSGetQuadratureTS - Return the sub-TS that evaluates integrals over time

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

   Output Parameters:
+  fwd - flag indicating whether to evaluate cost integral in the forward run or the adjoint run
-  quadts - the child TS context

   Level: intermediate

.seealso: TSCreateQuadratureTS()
@*/
PetscErrorCode TSGetQuadratureTS(TS ts,PetscBool *fwd,TS *quadts)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (fwd) *fwd = ts->costintegralfwd;
  if (quadts) *quadts = ts->quadraturets;
  PetscFunctionReturn(0);
}

/*@
   TSComputeSNESJacobian - Compute the SNESJacobian

   Input Parameters:
+  ts - the TS context obtained from TSCreate()
-  x - state vector

   Output Parameters:
+  J - Jacobian matrix
-  Jpre - preconditioning matrix for J (may be same as J)

   Level: developer

   Notes:
   Using SNES to compute the Jacobian enables finite differencing when TS Jacobian is not available.
@*/
PetscErrorCode TSComputeSNESJacobian(TS ts,Vec x,Mat J,Mat Jpre)
{
  SNES           snes = ts->snes;
  PetscErrorCode (*jac)(SNES,Vec,Mat,Mat,void*) = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*
    Unlike implicit methods, explicit methods do not have SNESMatFDColoring in the snes object
    because SNESSolve() has not been called yet; so querying SNESMatFDColoring does not work for
    explicit methods. Instead, we check the Jacobian compute function directly to determin if FD
    coloring is used.
  */
  ierr = SNESGetJacobian(snes,NULL,NULL,&jac,NULL);CHKERRQ(ierr);
  if (jac == SNESComputeJacobianDefaultColor) {
    Vec f;
    ierr = SNESSetSolution(snes,x);CHKERRQ(ierr);
    ierr = SNESGetFunction(snes,&f,NULL,NULL);CHKERRQ(ierr);
    /* Force MatFDColoringApply to evaluate the SNES residual function for the base vector */
    ierr = SNESComputeFunction(snes,x,f);CHKERRQ(ierr);
  }
  ierr = SNESComputeJacobian(snes,x,J,Jpre);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
