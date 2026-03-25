#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/

typedef struct _n_TaoTerm_Callbacks TaoTerm_Callbacks;

struct _n_TaoTerm_Callbacks {
  Tao tao;
  PetscErrorCode (*objective)(Tao, Vec, PetscReal *, PetscCtx);
  PetscErrorCode (*gradient)(Tao, Vec, Vec, PetscCtx);
  PetscErrorCode (*objectiveandgradient)(Tao, Vec, PetscReal *, Vec, PetscCtx);
  PetscErrorCode (*hessian)(Tao, Vec, Mat, Mat, PetscCtx);
  PetscCtx obj_ctx;
  PetscCtx grad_ctx;
  PetscCtx objgrad_ctx;
  PetscCtx hess_ctx;
  char    *obj_name;
  char    *grad_name;
  char    *objgrad_name;
  char    *hess_name;
  char    *set_obj_name;
  char    *set_grad_name;
  char    *set_objgrad_name;
  char    *set_hess_name;
};

#define PetscCheckTaoTermCallbacksValid(term, tt, params) \
  do { \
    PetscCheck((params) == NULL, PetscObjectComm((PetscObject)(term)), PETSC_ERR_ARG_INCOMP, "TAOTERMCALLBACKS does not accept a vector of parameters"); \
    PetscCheck((tt)->tao != NULL, PetscObjectComm((PetscObject)(term)), PETSC_ERR_ARG_WRONGSTATE, "TAOTERMCALLBACKS does not have an outer Tao"); \
  } while (0)

static PetscErrorCode TaoTermDestroy_Callbacks(TaoTerm term)
{
  TaoTerm_Callbacks *tt = (TaoTerm_Callbacks *)term->data;

  PetscFunctionBegin;
  // tt->tao is a weak reference, we do not destroy it
  PetscCall(PetscFree(tt->obj_name));
  PetscCall(PetscFree(tt->grad_name));
  PetscCall(PetscFree(tt->objgrad_name));
  PetscCall(PetscFree(tt->hess_name));
  PetscCall(PetscFree(tt->set_obj_name));
  PetscCall(PetscFree(tt->set_grad_name));
  PetscCall(PetscFree(tt->set_objgrad_name));
  PetscCall(PetscFree(tt->set_hess_name));
  PetscCall(PetscFree(tt));
  term->data = NULL;
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermCallbacksSetObjective_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermCallbacksGetObjective_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermCallbacksSetGradient_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermCallbacksGetGradient_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermCallbacksSetObjectiveAndGradient_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermCallbacksGetObjectiveAndGradient_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermCallbacksSetHessian_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermCallbacksGetHessian_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermComputeObjective_Callbacks(TaoTerm term, Vec x, Vec params, PetscReal *value)
{
  TaoTerm_Callbacks *tt = (TaoTerm_Callbacks *)term->data;

  PetscFunctionBegin;
  PetscCheckTaoTermCallbacksValid(term, tt, params);
  if (tt->objective) {
    PetscCallBack(tt->obj_name, (*tt->objective)(tt->tao, x, value, tt->obj_ctx));
  } else if (tt->objectiveandgradient) {
    Vec dummy;

    PetscCall(PetscInfo(term, "%s: Duplicating variable vector in order to call func/grad routine\n", ((PetscObject)term)->prefix));
    PetscCall(VecDuplicate(x, &dummy));
    PetscCallBack(tt->objgrad_name, (*tt->objectiveandgradient)(tt->tao, x, value, dummy, tt->objgrad_ctx));
    PetscCall(VecDestroy(&dummy));
  } else SETERRQ(PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_WRONGSTATE, "Objective routine not set: call %s", tt->set_obj_name);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermComputeGradient_Callbacks(TaoTerm term, Vec x, Vec params, Vec g)
{
  TaoTerm_Callbacks *tt = (TaoTerm_Callbacks *)term->data;

  PetscFunctionBegin;
  PetscCheckTaoTermCallbacksValid(term, tt, params);
  if (tt->gradient) {
    PetscCallBack(tt->grad_name, (*tt->gradient)(tt->tao, x, g, tt->grad_ctx));
  } else if (tt->objectiveandgradient) {
    PetscReal dummy;

    PetscCallBack(tt->objgrad_name, (*tt->objectiveandgradient)(tt->tao, x, &dummy, g, tt->objgrad_ctx));
  } else SETERRQ(PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_WRONGSTATE, "Gradient routine not set: call %s", tt->set_grad_name);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermComputeObjectiveAndGradient_Callbacks(TaoTerm term, Vec x, Vec params, PetscReal *value, Vec g)
{
  TaoTerm_Callbacks *tt = (TaoTerm_Callbacks *)term->data;

  PetscFunctionBegin;
  PetscCheckTaoTermCallbacksValid(term, tt, params);
  if (tt->objectiveandgradient) {
    PetscCallBack(tt->objgrad_name, (*tt->objectiveandgradient)(tt->tao, x, value, g, tt->objgrad_ctx));
  } else if (tt->objective && tt->gradient) {
    PetscCallBack(tt->obj_name, (*tt->objective)(tt->tao, x, value, tt->obj_ctx));
    PetscCallBack(tt->grad_name, (*tt->gradient)(tt->tao, x, g, tt->grad_ctx));
  } else SETERRQ(PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_WRONGSTATE, "Objective/gradient routine not set: call %s", tt->set_objgrad_name);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermComputeHessian_Callbacks(TaoTerm term, Vec x, Vec params, Mat H, Mat Hpre)
{
  TaoTerm_Callbacks *tt = (TaoTerm_Callbacks *)term->data;

  PetscFunctionBegin;
  PetscCheckTaoTermCallbacksValid(term, tt, params);
  PetscCheck(tt->hessian, PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_WRONGSTATE, "Hessian routine not set: call %s", tt->set_hess_name);
  PetscCallBack(tt->hess_name, (*tt->hessian)(tt->tao, x, H, Hpre, tt->hess_ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermView_Callbacks(TaoTerm term, PetscViewer viewer)
{
  TaoTerm_Callbacks *tt = (TaoTerm_Callbacks *)term->data;
  PetscBool          iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    if (tt->tao == NULL) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "not attached to a Tao\n"));
    } else {
      const char *name = "[name omitted]";
      const char *prefix;

      if (!PetscCIEnabled) PetscCall(PetscObjectGetName((PetscObject)tt->tao, &name));
      else if (tt->tao->hdr.name) name = tt->tao->hdr.name;
      PetscCall(PetscObjectGetOptionsPrefix((PetscObject)tt->tao, &prefix));
      if (prefix) PetscCall(PetscViewerASCIIPrintf(viewer, "attached to Tao %s (%s)\n", name, prefix));
      else PetscCall(PetscViewerASCIIPrintf(viewer, "attached to Tao %s\n", name));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTermCallbacksSetObjective(TaoTerm term, PetscErrorCode (*tao_obj)(Tao, Vec, PetscReal *, PetscCtx), PetscCtx ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscTryMethod(term, "TaoTermCallbacksSetObjective_C", (TaoTerm, PetscErrorCode (*)(Tao, Vec, PetscReal *, PetscCtx), PetscCtx), (term, tao_obj, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermCallbacksSetObjective_Callbacks(TaoTerm term, PetscErrorCode (*tao_obj)(Tao, Vec, PetscReal *, PetscCtx), PetscCtx ctx)
{
  TaoTerm_Callbacks *tt = (TaoTerm_Callbacks *)term->data;

  PetscFunctionBegin;
  tt->objective = tao_obj;
  tt->obj_ctx   = ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTermCallbacksGetObjective(TaoTerm term, PetscErrorCode (**tao_obj)(Tao, Vec, PetscReal *, PetscCtx), PetscCtxRt ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  if (tao_obj) *tao_obj = NULL;
  if (ctx) *(void **)ctx = NULL;
  PetscTryMethod(term, "TaoTermCallbacksGetObjective_C", (TaoTerm, PetscErrorCode (**)(Tao, Vec, PetscReal *, PetscCtx), PetscCtxRt), (term, tao_obj, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermCallbacksGetObjective_Callbacks(TaoTerm term, PetscErrorCode (**tao_obj)(Tao, Vec, PetscReal *, PetscCtx), PetscCtxRt ctx)
{
  TaoTerm_Callbacks *tt = (TaoTerm_Callbacks *)term->data;

  PetscFunctionBegin;
  if (tao_obj) *tao_obj = tt->objective;
  if (ctx) *(void **)ctx = tt->obj_ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTermCallbacksSetGradient(TaoTerm term, PetscErrorCode (*tao_grad)(Tao, Vec, Vec, PetscCtx), PetscCtx ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscTryMethod(term, "TaoTermCallbacksSetGradient_C", (TaoTerm, PetscErrorCode (*)(Tao, Vec, Vec, PetscCtx), PetscCtx), (term, tao_grad, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermCallbacksSetGradient_Callbacks(TaoTerm term, PetscErrorCode (*tao_grad)(Tao, Vec, Vec, PetscCtx), PetscCtx ctx)
{
  TaoTerm_Callbacks *tt = (TaoTerm_Callbacks *)term->data;

  PetscFunctionBegin;
  tt->gradient = tao_grad;
  tt->grad_ctx = ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTermCallbacksGetGradient(TaoTerm term, PetscErrorCode (**tao_grad)(Tao, Vec, Vec, PetscCtx), PetscCtxRt ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  if (tao_grad) *tao_grad = NULL;
  if (ctx) *(void **)ctx = NULL;
  PetscTryMethod(term, "TaoTermCallbacksGetGradient_C", (TaoTerm, PetscErrorCode (**)(Tao, Vec, Vec, PetscCtx), PetscCtxRt), (term, tao_grad, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermCallbacksGetGradient_Callbacks(TaoTerm term, PetscErrorCode (**tao_grad)(Tao, Vec, Vec, PetscCtx), PetscCtxRt ctx)
{
  TaoTerm_Callbacks *tt = (TaoTerm_Callbacks *)term->data;

  PetscFunctionBegin;
  if (tao_grad) *tao_grad = tt->gradient;
  if (ctx) *(void **)ctx = tt->grad_ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTermCallbacksSetObjectiveAndGradient(TaoTerm term, PetscErrorCode (*tao_objgrad)(Tao, Vec, PetscReal *, Vec, PetscCtx), PetscCtx ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscTryMethod(term, "TaoTermCallbacksSetObjectiveAndGradient_C", (TaoTerm, PetscErrorCode (*)(Tao, Vec, PetscReal *, Vec, PetscCtx), PetscCtx), (term, tao_objgrad, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermCallbacksSetObjectiveAndGradient_Callbacks(TaoTerm term, PetscErrorCode (*tao_objgrad)(Tao, Vec, PetscReal *, Vec, PetscCtx), PetscCtx ctx)
{
  TaoTerm_Callbacks *tt = (TaoTerm_Callbacks *)term->data;

  PetscFunctionBegin;
  tt->objectiveandgradient = tao_objgrad;
  tt->objgrad_ctx          = ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTermCallbacksGetObjectiveAndGradient(TaoTerm term, PetscErrorCode (**tao_objgrad)(Tao, Vec, PetscReal *, Vec, PetscCtx), PetscCtxRt ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  if (tao_objgrad) *tao_objgrad = NULL;
  if (ctx) *(void **)ctx = NULL;
  PetscTryMethod(term, "TaoTermCallbacksGetObjectiveAndGradient_C", (TaoTerm, PetscErrorCode (**)(Tao, Vec, PetscReal *, Vec, PetscCtx), PetscCtxRt), (term, tao_objgrad, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermCallbacksGetObjectiveAndGradient_Callbacks(TaoTerm term, PetscErrorCode (**tao_objgrad)(Tao, Vec, PetscReal *, Vec, PetscCtx), PetscCtxRt ctx)
{
  TaoTerm_Callbacks *tt = (TaoTerm_Callbacks *)term->data;

  PetscFunctionBegin;
  if (tao_objgrad) *tao_objgrad = tt->objectiveandgradient;
  if (ctx) *(void **)ctx = tt->objgrad_ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTermCallbacksSetHessian(TaoTerm term, PetscErrorCode (*tao_hess)(Tao, Vec, Mat, Mat, PetscCtx), PetscCtx ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscTryMethod(term, "TaoTermCallbacksSetHessian_C", (TaoTerm, PetscErrorCode (*)(Tao, Vec, Mat, Mat, PetscCtx), PetscCtx), (term, tao_hess, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermCallbacksSetHessian_Callbacks(TaoTerm term, PetscErrorCode (*tao_hess)(Tao, Vec, Mat, Mat, PetscCtx), PetscCtx ctx)
{
  TaoTerm_Callbacks *tt = (TaoTerm_Callbacks *)term->data;

  PetscFunctionBegin;
  tt->hessian  = tao_hess;
  tt->hess_ctx = ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTermCallbacksGetHessian(TaoTerm term, PetscErrorCode (**tao_hess)(Tao, Vec, Mat, Mat, PetscCtx), PetscCtxRt ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  if (tao_hess) *tao_hess = NULL;
  if (ctx) *(void **)ctx = NULL;
  PetscTryMethod(term, "TaoTermCallbacksGetHessian_C", (TaoTerm, PetscErrorCode (**)(Tao, Vec, Mat, Mat, PetscCtx), PetscCtxRt), (term, tao_hess, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermCallbacksGetHessian_Callbacks(TaoTerm term, PetscErrorCode (**tao_hess)(Tao, Vec, Mat, Mat, PetscCtx), PetscCtxRt ctx)
{
  TaoTerm_Callbacks *tt = (TaoTerm_Callbacks *)term->data;

  PetscFunctionBegin;
  if (tao_hess) *tao_hess = tt->hessian;
  if (ctx) *(void **)ctx = tt->hess_ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermIsObjectiveDefined_Callbacks(TaoTerm term, PetscBool *flg)
{
  TaoTerm_Callbacks *tt = (TaoTerm_Callbacks *)term->data;

  PetscFunctionBegin;
  *flg = (tt->objective != NULL) ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermIsGradientDefined_Callbacks(TaoTerm term, PetscBool *flg)
{
  TaoTerm_Callbacks *tt = (TaoTerm_Callbacks *)term->data;

  PetscFunctionBegin;
  *flg = (tt->gradient != NULL) ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermIsObjectiveAndGradientDefined_Callbacks(TaoTerm term, PetscBool *flg)
{
  TaoTerm_Callbacks *tt = (TaoTerm_Callbacks *)term->data;

  PetscFunctionBegin;
  *flg = (tt->objectiveandgradient != NULL) ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermIsHessianDefined_Callbacks(TaoTerm term, PetscBool *flg)
{
  TaoTerm_Callbacks *tt = (TaoTerm_Callbacks *)term->data;

  PetscFunctionBegin;
  *flg = (tt->hessian != NULL) ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermCreateHessianMatrices_Callbacks(TaoTerm term, Mat *H, Mat *Hpre)
{
  PetscFunctionBegin;
  if (H) *H = NULL;
  if (Hpre) *Hpre = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermCreate_Callbacks_Internal(TaoTerm term, const char obj[], const char set_obj[], const char grad[], const char set_grad[], const char objgrad[], const char set_objgrad[], const char hess[], const char set_hess[])
{
  TaoTerm_Callbacks *tt;
  char               buf[256];
  size_t             len = PETSC_STATIC_ARRAY_LENGTH(buf);

  PetscFunctionBegin;
  term->parameters_mode = TAOTERM_PARAMETERS_NONE;

  PetscCall(PetscNew(&tt));
  term->data = (void *)tt;

  term->ops->destroy                       = TaoTermDestroy_Callbacks;
  term->ops->objective                     = TaoTermComputeObjective_Callbacks;
  term->ops->gradient                      = TaoTermComputeGradient_Callbacks;
  term->ops->objectiveandgradient          = TaoTermComputeObjectiveAndGradient_Callbacks;
  term->ops->hessian                       = TaoTermComputeHessian_Callbacks;
  term->ops->createhessianmatrices         = TaoTermCreateHessianMatrices_Callbacks;
  term->ops->view                          = TaoTermView_Callbacks;
  term->ops->isobjectivedefined            = TaoTermIsObjectiveDefined_Callbacks;
  term->ops->isgradientdefined             = TaoTermIsGradientDefined_Callbacks;
  term->ops->isobjectiveandgradientdefined = TaoTermIsObjectiveAndGradientDefined_Callbacks;
  term->ops->ishessiandefined              = TaoTermIsHessianDefined_Callbacks;
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermCallbacksSetObjective_C", TaoTermCallbacksSetObjective_Callbacks));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermCallbacksGetObjective_C", TaoTermCallbacksGetObjective_Callbacks));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermCallbacksSetGradient_C", TaoTermCallbacksSetGradient_Callbacks));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermCallbacksGetGradient_C", TaoTermCallbacksGetGradient_Callbacks));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermCallbacksSetObjectiveAndGradient_C", TaoTermCallbacksSetObjectiveAndGradient_Callbacks));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermCallbacksGetObjectiveAndGradient_C", TaoTermCallbacksGetObjectiveAndGradient_Callbacks));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermCallbacksSetHessian_C", TaoTermCallbacksSetHessian_Callbacks));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermCallbacksGetHessian_C", TaoTermCallbacksGetHessian_Callbacks));

  PetscCall(PetscSNPrintf(buf, len, "%s callback", obj ? obj : "unknown objective"));
  PetscCall(PetscStrallocpy(buf, &tt->obj_name));
  PetscCall(PetscStrallocpy(set_obj, &tt->set_obj_name));

  PetscCall(PetscSNPrintf(buf, len, "%s callback", grad ? grad : "unknown gradient"));
  PetscCall(PetscStrallocpy(buf, &tt->grad_name));
  PetscCall(PetscStrallocpy(set_grad, &tt->set_grad_name));

  PetscCall(PetscSNPrintf(buf, len, "%s callback", objgrad ? objgrad : "unknown objective/gradient"));
  PetscCall(PetscStrallocpy(buf, &tt->objgrad_name));
  PetscCall(PetscStrallocpy(set_objgrad, &tt->set_objgrad_name));

  PetscCall(PetscSNPrintf(buf, len, "%s callback", hess ? hess : "unknown hessian"));
  PetscCall(PetscStrallocpy(buf, &tt->hess_name));
  PetscCall(PetscStrallocpy(set_hess, &tt->set_hess_name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  TAOTERMCALLBACKS - A `TaoTerm` implementation that accesses the callback functions that have
  been provided with in `TaoSetObjective()`, `TaoSetGradient()`,
  `TaoSetObjectiveAndGradient()`, and `TaoSetHessian()`.

  Level: developer

  Notes:
  If you are interested in creating your own term, you should not use this. Use
  `TAOTERMSHELL` or create your own implementation of `TaoTerm` with
  `TaoTermRegister()`.

  A `TAOTERMCALLBACKS` is always `TAOTERM_PARAMETERS_NONE`, so the `params`
  argument of `TaoTerm` evaluation routines should always be `NULL`.

  A `TAOTERMCALLBACKS` cannot create Hessian matrices; the user needs to pass the
  Hessian matrices used in algorithms in `TaoSetHessian()`.

  Developer Notes:
  Internally each `Tao` has a `TaoTerm` of type `TAOTERMCALLBACKS` that is updated
  by the `Tao` callback routines (`TaoSetObjective()`, `TaoSetGradient()`,
  `TaoSetObjectiveAndGradient()`, and `TaoSetHessian()`).

  The routines that get the user-defined `Tao` callback functions
  (`TaoGetObjective()`, `TaoGetObjectiveAndGradient()`, `TaoGetGradient()`,
  `TaoGetHessian()`) will always return those original callbacks, even if the
  objective function has been changed by `TaoAddTerm()`,
  so PETSc/TAO should not assume that those callbacks are valid in any library code.

  A `TAOTERMCALLBACKS` has a weak-reference to the `Tao` that created it,
  which may not be the `Tao` currently using it because the term could have been shared
  using `TaoGetTerm()` and `TaoAddTerm()`.

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermType`
M*/
PETSC_INTERN PetscErrorCode TaoTermCreate_Callbacks(TaoTerm term)
{
  PetscFunctionBegin;
  // clang-format off
  PetscCall(TaoTermCreate_Callbacks_Internal(term,
        "TaoComputeObjective()",            "TaoSetObjective()",
        "TaoComputeGradient()",             "TaoSetGradient()",
        "TaoComputeObjectiveAndGradient()", "TaoSetObjectiveAndGradient()",
        "TaoComputeHessian()",              "TaoSetHessian()"));
  // clang-format on
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTermCreateCallbacks(Tao tao, TaoTerm *term)
{
  PetscFunctionBegin;
  PetscCall(TaoTermCreate(PetscObjectComm((PetscObject)tao), term));
  PetscCall(TaoTermSetType(*term, TAOTERMCALLBACKS));
  {
    TaoTerm_Callbacks *tt = (TaoTerm_Callbacks *)((*term)->data);

    tt->tao = tao; // weak reference, do not increment reference count
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
