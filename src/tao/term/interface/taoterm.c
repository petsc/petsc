#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/
#include <petsc/private/vecimpl.h>

PetscClassId TAOTERM_CLASSID;

PetscLogEvent TAOTERM_ObjectiveEval;
PetscLogEvent TAOTERM_GradientEval;
PetscLogEvent TAOTERM_ObjGradEval;
PetscLogEvent TAOTERM_HessianEval;

const char *const TaoTermParametersModes[] = {"optional", "none", "required", "TaoTermParametersMode", "TAOTERM_PARAMETERS_", NULL};

/*@
  TaoTermDestroy - Destroy a `TaoTerm`.

  Collective

  Input Parameter:
. term - a `TaoTerm`

  Level: beginner

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermCreate()`,
          `TaoTermSetType()`,
          `TaoTermSetFromOptions()`,
          `TaoTermSetUp()`,
          `TaoTermView()`
@*/
PetscErrorCode TaoTermDestroy(TaoTerm *term)
{
  PetscFunctionBegin;
  if (!*term) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific(*term, TAOTERM_CLASSID, 1);
  if (--((PetscObject)*term)->refct > 0) {
    *term = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscTryTypeMethod(*term, destroy);
  PetscCall(PetscFree((*term)->H_mattype));
  PetscCall(PetscFree((*term)->Hpre_mattype));
  PetscCall(MatDestroy(&(*term)->solution_factory));
  PetscCall(MatDestroy(&(*term)->parameters_factory));
  PetscCall(MatDestroy(&(*term)->parameters_factory_orig));
  PetscCall(PetscHeaderDestroy(term));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermView - View a description of a `TaoTerm`.

  Collective

  Input Parameters:
+ term   - a `TaoTerm`
- viewer - a `PetscViewer`

  Level: beginner

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermCreate()`,
          `TaoTermSetType()`,
          `TaoTermSetFromOptions()`,
          `TaoTermSetUp()`,
          `TaoTermDestroy()`,
          `PetscViewer`
@*/
PetscErrorCode TaoTermView(TaoTerm term, PetscViewer viewer)
{
  PetscBool iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)term), &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(term, 1, viewer, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    const char       *solution_vec_type;
    PetscInt          N;
    PetscViewerFormat format;

    PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)term, viewer));
    PetscCall(PetscViewerGetFormat(viewer, &format));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(MatGetVecType(term->solution_factory, &solution_vec_type));
    PetscCall(MatGetSize(term->solution_factory, &N, NULL));
    if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      if (N < 0) PetscCall(PetscViewerASCIIPrintf(viewer, "solution vector space: not set up yet [VecType %s (tao_term_solution_vec_type)]\n", solution_vec_type));
      else PetscCall(PetscViewerASCIIPrintf(viewer, "solution vector space: N = %" PetscInt_FMT " [VecType %s (tao_term_solution_vec_type)]\n", N, solution_vec_type));
    } else {
      if (N < 0) PetscCall(PetscViewerASCIIPrintf(viewer, "solution vector space: not set up yet\n"));
      else PetscCall(PetscViewerASCIIPrintf(viewer, "solution vector space: N = %" PetscInt_FMT "\n", N));
    }
    if (format == PETSC_VIEWER_ASCII_INFO_DETAIL && term->parameters_mode == TAOTERM_PARAMETERS_NONE) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "parameter vector space: (none)\n"));
    } else if (term->parameters_mode != TAOTERM_PARAMETERS_NONE) {
      const char *parameters_vec_type;
      PetscInt    K;

      PetscCall(MatGetVecType(term->parameters_factory, &parameters_vec_type));
      PetscCall(MatGetSize(term->parameters_factory, &K, NULL));
      if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
        if (K < 0) PetscCall(PetscViewerASCIIPrintf(viewer, "parameter vector space: (%s) not set up yet [VecType %s (tao_term_parameters_vec_type)]\n", TaoTermParametersModes[term->parameters_mode], parameters_vec_type));
        else PetscCall(PetscViewerASCIIPrintf(viewer, "parameter vector space: (%s) K = %" PetscInt_FMT " [VecType %s (tao_term_parameters_vec_type)]\n", TaoTermParametersModes[term->parameters_mode], K, parameters_vec_type));
      } else {
        if (K < 0) PetscCall(PetscViewerASCIIPrintf(viewer, "parameter vector space:%s not set up yet\n", term->parameters_mode == TAOTERM_PARAMETERS_OPTIONAL ? " (optional)" : ""));
        else PetscCall(PetscViewerASCIIPrintf(viewer, "parameter vector space:%s K = %" PetscInt_FMT "\n", term->parameters_mode == TAOTERM_PARAMETERS_OPTIONAL ? " (optional)" : "", K));
      }
    }
    if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      PetscBool3 is_fdpossible;

      PetscCall(TaoTermIsComputeHessianFDPossible(term, &is_fdpossible));
      if (is_fdpossible == PETSC_BOOL3_FALSE) {
        if (term->fd_hessian) PetscCall(PetscViewerASCIIPrintf(viewer, "Finite differences for Hessian computation was requested, but ignored.\n"));
        if (term->ops->createhessianmatrices == TaoTermCreateHessianMatricesDefault) {
          PetscCall(PetscViewerASCIIPrintf(viewer, "default Hessian MatType (tao_term_hessian_mat_type): %s\n", term->H_mattype ? term->H_mattype : "(undefined)"));
          if (!term->Hpre_is_H) PetscCall(PetscViewerASCIIPrintf(viewer, "default Hessian preconditioning MatType (tao_term_hessian_pre_mat_type): %s\n", term->Hpre_mattype ? term->Hpre_mattype : "(undefined)"));
        } else {
          PetscCall(PetscViewerASCIIPrintf(viewer, "Hessian MatType (tao_term_hessian_mat_type): %s\n", term->H_mattype ? term->H_mattype : "(undefined)"));
          if (!term->Hpre_is_H) PetscCall(PetscViewerASCIIPrintf(viewer, "Hessian preconditioning MatType (tao_term_hessian_pre_mat_type): %s\n", term->Hpre_mattype ? term->Hpre_mattype : "(undefined)"));
        }
      } else {
        if (term->fd_hessian) PetscCall(PetscViewerASCIIPrintf(viewer, "Using finite differences for Hessian computation\n"));
        else if (term->ops->createhessianmatrices == TaoTermCreateHessianMatricesDefault) {
          PetscCall(PetscViewerASCIIPrintf(viewer, "default Hessian MatType (tao_term_hessian_mat_type): %s\n", term->H_mattype ? term->H_mattype : "(undefined)"));
          if (!term->Hpre_is_H) PetscCall(PetscViewerASCIIPrintf(viewer, "default Hessian preconditioning MatType (tao_term_hessian_pre_mat_type): %s\n", term->Hpre_mattype ? term->Hpre_mattype : "(undefined)"));
        } else {
          PetscBool is_h_mffd    = PETSC_FALSE;
          PetscBool is_hpre_mffd = PETSC_FALSE;

          if (term->H_mattype) PetscCall(PetscStrcmp(term->H_mattype, MATMFFD, &is_h_mffd));
          if (term->Hpre_mattype) PetscCall(PetscStrcmp(term->Hpre_mattype, MATMFFD, &is_hpre_mffd));

          PetscCall(PetscViewerASCIIPrintf(viewer, "Hessian MatType (tao_term_hessian_mat_type): %s\n", term->H_mattype ? term->H_mattype : "(undefined)"));
          if (is_h_mffd) PetscCall(PetscViewerASCIIPrintf(viewer, "total number of gradient evaluations used by MFFD=%" PetscInt_FMT "\n", term->ngrad_mffd));
          if (!term->Hpre_is_H) PetscCall(PetscViewerASCIIPrintf(viewer, "Hessian preconditioning MatType (tao_term_hessian_pre_mat_type): %s\n", term->Hpre_mattype ? term->Hpre_mattype : "(undefined)"));
        }
      }
    }
    PetscTryTypeMethod(term, view, viewer);
    if (term->nobj > 0) PetscCall(PetscViewerASCIIPrintf(viewer, "total number of function evaluations=%" PetscInt_FMT "\n", term->nobj));
    if (term->ngrad > 0) PetscCall(PetscViewerASCIIPrintf(viewer, "total number of gradient evaluations=%" PetscInt_FMT "\n", term->ngrad));
    if (term->nobjgrad > 0) PetscCall(PetscViewerASCIIPrintf(viewer, "total number of function/gradient evaluations=%" PetscInt_FMT "\n", term->nobjgrad));
    if (term->nhess > 0) PetscCall(PetscViewerASCIIPrintf(viewer, "total number of Hessian evaluations=%" PetscInt_FMT "\n", term->nhess));
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermSetUp - Set up a `TaoTerm`.

  Collective

  Input Parameter:
. term - a `TaoTerm`

  Level: intermediate

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermCreate()`,
          `TaoTermSetType()`,
          `TaoTermSetFromOptions()`,
          `TaoTermView()`,
          `TaoTermDestroy()`
@*/
PetscErrorCode TaoTermSetUp(TaoTerm term)
{
  PetscInt N;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  if (term->setup_called) PetscFunctionReturn(PETSC_SUCCESS);
  term->setup_called = PETSC_TRUE;
  PetscTryTypeMethod(term, setup);
  PetscCall(MatGetSize(term->solution_factory, &N, NULL));
  if (N < 0) {
    PetscBool is_shell;
    Vec       sol_template;

    PetscCall(PetscObjectTypeCompare((PetscObject)term, TAOTERMSHELL, &is_shell));
    if (is_shell)
      PetscCheck(term->ops->createsolutionvec, PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_WRONGSTATE, "TaoTerm solution space not known. You should have called TaoTermSetSolutionSizes(), TaoTermSetSolutionTemplate(), TaoTermSetSolutionLayout(), or TaoTermShellSetCreateSolutionVec()");
    else
      PetscCheck(term->ops->createsolutionvec, PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_WRONGSTATE, "TaoTerm solution space not known. You should have called TaoTermSetSolutionSizes(), TaoTermSetSolutionTemplate(), or TaoTermSetSolutionLayout()");

    PetscCall(TaoTermCreateSolutionVec(term, &sol_template));
    PetscCall(TaoTermSetSolutionTemplate(term, sol_template));
    PetscCall(VecDestroy(&sol_template));
  }
  PetscCall(MatSetUp(term->solution_factory));
  if (term->parameters_mode != TAOTERM_PARAMETERS_NONE) {
    PetscInt K;

    PetscCall(MatGetSize(term->parameters_factory, &K, NULL));
    if (K < 0) {
      PetscBool is_shell;
      Vec       params_template;

      PetscCall(PetscObjectTypeCompare((PetscObject)term, TAOTERMSHELL, &is_shell));
      if (is_shell)
        PetscCheck(term->ops->createparametersvec, PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_WRONGSTATE, "TaoTerm parameters space not known. You should have called TaoTermSetParametersSizes(), TaoTermSetParametersTemplate(), TaoTermSetParametersLayout(), or TaoTermShellSetCreateParametersVec()");
      else
        PetscCheck(term->ops->createparametersvec, PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_WRONGSTATE, "TaoTerm parameters space not known. You should have called TaoTermSetParametersSizes(), TaoTermSetParametersTemplate(), or TaoTermSetParametersLayout()");

      PetscCall(TaoTermCreateParametersVec(term, &params_template));
      PetscCall(TaoTermSetParametersTemplate(term, params_template));
      PetscCall(VecDestroy(&params_template));
    }
    PetscCall(MatSetUp(term->parameters_factory));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermSetFromOptions - Configure a `TaoTerm` from the PETSc options database

  Collective

  Input Parameter:
. term - a `TaoTerm`

  Options Database Keys:
+ -tao_term_type <type>                              - l1, halfl2squared; see `TaoTermType` for a complete list
. -tao_term_solution_vec_type <type>                 - the type of vector to use for the solution, see `VecType` for a complete list of vector types
. -tao_term_parameters_vec_type <type>               - the type of vector to use for the parameters, see `VecType` for a complete list of vector types
. -tao_term_parameters_mode <optional,none,required> - `TAOTERM_PARAMETERS_OPTIONAL`, `TAOTERM_PARAMETERS_NONE`, `TAOTERM_PARAMETERS_REQUIRED`
. -tao_term_hessian_pre_is_hessian <bool>            - Whether `TaoTermCreateHessianMatricesDefault()` should make a separate preconditioning matrix
. -tao_term_hessian_mat_type <type>                  - `MatType` for Hessian matrix created by `TaoTermCreateHessianMatricesDefault()`
. -tao_term_hessian_pre_mat_type <type>              - `MatType` for approximate Hessian matrix used to construct the preconditioner created by `TaoTermCreateHessianMatricesDefault()`
. -tao_term_fd_delta <real>                          - Increment for finite difference derivative approximations in `TaoTermComputeGradientFD()`
. -tao_term_gradient_use_fd <bool>                   - Use finite differences in `TaoTermComputeGradient()`, overriding other user-provided or built-in routines
- -tao_term_hessian_use_fd <bool>                    - Use finite differences in `TaoTermComputeHessian()`, overriding other user-provided or built-in routines

  Level: beginner

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermCreate()`,
          `TaoTermSetType()`,
          `TaoTermSetUp()`,
          `TaoTermView()`,
          `TaoTermDestroy()`
@*/
PetscErrorCode TaoTermSetFromOptions(TaoTerm term)
{
  const char *deft = TAOTERMSHELL;
  PetscBool   flg;
  char        typeName[256];
  VecType     sol_type, params_type;
  PetscBool   opt;
  PetscBool   grad_use_fd;
  PetscBool   hess_use_fd;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  if (((PetscObject)term)->type_name) deft = ((PetscObject)term)->type_name;
  PetscObjectOptionsBegin((PetscObject)term);
  PetscCall(PetscOptionsFList("-tao_term_type", "TaoTerm type", "TaoTermType", TaoTermList, deft, typeName, 256, &flg));
  if (flg) PetscCall(TaoTermSetType(term, typeName));
  else PetscCall(TaoTermSetType(term, deft));
  PetscCall(TaoTermGetSolutionVecType(term, &sol_type));
  PetscCall(TaoTermGetParametersVecType(term, &params_type));
  PetscCall(PetscOptionsFList("-tao_term_solution_vec_type", "Solution vector type", "TaoTermSetSolutionVecType", VecList, sol_type, typeName, 256, &opt));
  if (opt) PetscCall(TaoTermSetSolutionVecType(term, typeName));
  PetscCall(PetscOptionsFList("-tao_term_parameters_vec_type", "Parameters vector type", "TaoTermSetParametersVecType", VecList, params_type, typeName, 256, &opt));
  if (opt) PetscCall(TaoTermSetParametersVecType(term, typeName));
  PetscCall(PetscOptionsEnum("-tao_term_parameters_mode", "Parameters requirement type", "TaoTermSetParametersMode", TaoTermParametersModes, (PetscEnum)term->parameters_mode, (PetscEnum *)&term->parameters_mode, NULL));
  PetscCall(PetscOptionsBool("-tao_term_hessian_pre_is_hessian", "If the Hessian and its preconditioning matrix should be the same", "TaoTermSetCreateHessianMode", term->Hpre_is_H, &term->Hpre_is_H, NULL));

  deft = MATAIJ;
  if (term->H_mattype) deft = term->H_mattype;
  PetscCall(PetscOptionsFList("-tao_term_hessian_mat_type", "Hessian mat type", "TaoTermSetCreateHessianMode", MatList, deft, typeName, 256, &opt));
  if (opt) {
    PetscBool is_mffd, is_shell, is_callbacks;
    PetscCall(PetscStrcmp(typeName, MATMFFD, &is_mffd));
    if (is_mffd) {
      PetscCall(PetscObjectTypeCompare((PetscObject)term, TAOTERMSHELL, &is_shell));
      PetscCall(PetscObjectTypeCompare((PetscObject)term, TAOTERMCALLBACKS, &is_callbacks));
      if (is_shell || is_callbacks) {
        PetscCall(PetscFree(term->H_mattype));
        PetscCall(PetscStrallocpy(typeName, (char **)&term->H_mattype));
      } else {
        PetscCall(PetscInfo(term, "%s: TaoTerm Hessian MatType MFFD requested but TaoTerm type is neither SHELL nor CALLBACKS. Ignoring.\n", ((PetscObject)term)->prefix));
      }
    } else {
      PetscCall(PetscFree(term->H_mattype));
      PetscCall(PetscStrallocpy(typeName, (char **)&term->H_mattype));
    }
  }

  deft = MATAIJ;
  if (term->Hpre_mattype) deft = term->Hpre_mattype;
  PetscCall(PetscOptionsFList("-tao_term_hessian_pre_mat_type", "Hessian preconditioning mat type", "TaoTermSetCreateHessianMode", MatList, deft, typeName, 256, &opt));
  if (opt) {
    PetscCall(PetscFree(term->Hpre_mattype));
    PetscCall(PetscStrallocpy(typeName, (char **)&term->Hpre_mattype));
  }

  if (term->Hpre_is_H) {
    PetscBool mattypes_same;

    PetscCall(PetscStrcmp(term->H_mattype, term->Hpre_mattype, &mattypes_same));
    if (!mattypes_same) {
      PetscCall(PetscInfo(term, "%s: Hpre_is_H, but H_mattype and Hpre_mattype are different. Setting Hpre_mattype to be same as H_mattype\n", ((PetscObject)term)->prefix));
      PetscCall(PetscFree(term->Hpre_mattype));
      PetscCall(PetscStrallocpy(term->H_mattype, (char **)&term->Hpre_mattype));
    }
  }

  PetscCall(PetscOptionsBoundedReal("-tao_term_fd_delta", "Finite difference increment", "TaoTermSetFDDelta", term->fd_delta, &term->fd_delta, NULL, PETSC_SMALL));
  PetscCall(PetscInfo(term, "%s: Finite difference delta set to %g\n", ((PetscObject)term)->prefix, (double)term->fd_delta));

  grad_use_fd = term->fd_gradient;
  PetscCall(PetscOptionsBool("-tao_term_gradient_use_fd", "Use finite differences in TaoTermComputeGradient()", "TaoTermComputeGradientSetUseFD", grad_use_fd, &grad_use_fd, NULL));
  PetscCall(TaoTermComputeGradientSetUseFD(term, grad_use_fd));

  hess_use_fd = term->fd_hessian;
  PetscCall(PetscOptionsBool("-tao_term_hessian_use_fd", "Use finite differences in TaoTermComputeHessian()", "TaoTermComputeHessianSetUseFD", hess_use_fd, &hess_use_fd, NULL));
  PetscCall(TaoTermComputeHessianSetUseFD(term, hess_use_fd));

  PetscTryTypeMethod(term, setfromoptions, PetscOptionsObject);
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermSetType - Set the type of a `TaoTerm`

  Collective

  Input Parameters:
+ term - a `TaoTerm`
- type - a `TaoTermType`

  Options Database Keys:
. -tao_term_type <type> - l1, halfl2squared, `TaoTermType` for complete list

  Level: beginner

  Notes:
  Use `TaoTermCreateShell()` to define a custom term using your own function definition

  New types of `TaoTerm` can be created with `TaoTermRegister()`

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermType`,
          `TaoTermCreate()`,
          `TaoTermGetType()`,
          `TaoTermSetFromOptions()`,
          `TaoTermSetUp()`,
          `TaoTermView()`,
          `TaoTermDestroy()`
@*/
PetscErrorCode TaoTermSetType(TaoTerm term, TaoTermType type)
{
  PetscErrorCode (*create)(TaoTerm);
  PetscBool issame;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);

  PetscCall(PetscObjectTypeCompare((PetscObject)term, type, &issame));
  if (issame) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscFunctionListFind(TaoTermList, type, (void (**)(void))&create));
  PetscCheck(create, PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unable to find requested TaoTerm type %s", type);

  /* Destroy the existing term information */
  PetscTryTypeMethod(term, destroy);
  term->setup_called = PETSC_FALSE;
  PetscCall(PetscMemzero(term->ops, sizeof(struct _TaoTermOps)));

  PetscCall((*create)(term));
  PetscCall(PetscObjectChangeTypeName((PetscObject)term, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermGetType - Get the type of a `TaoTerm`

  Not collective

  Input Parameter:
. term - a `TaoTerm`

  Output Parameter:
. type - the `TaoTermType`

  Level: beginner

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermType`,
          `TaoTermCreate()`,
          `TaoTermSetType()`,
          `TaoTermSetFromOptions()`,
          `TaoTermSetUp()`,
          `TaoTermView()`,
          `TaoTermDestroy()`
@*/
PetscErrorCode TaoTermGetType(TaoTerm term, TaoTermType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscAssertPointer(type, 2);
  *type = ((PetscObject)term)->type_name;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermCreate - Create a TaoTerm to use in defining the function `Tao` is to optimize

  Collective

  Input Parameter:
. comm - communicator for MPI processes that compute the term

  Output Parameter:
. term - a new `TaoTerm`

  Level: beginner

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermSetType()`,
          `TaoAddTerm()`,
          `TaoTermSetFromOptions()`,
          `TaoTermSetUp()`,
          `TaoTermView()`,
          `TaoTermDestroy()`
@*/
PetscErrorCode TaoTermCreate(MPI_Comm comm, TaoTerm *term)
{
  TaoTerm     _term;
  PetscLayout zero_layout, rlayout, clayout;

  PetscFunctionBegin;
  PetscAssertPointer(term, 2);
  PetscCall(TaoInitializePackage());
  PetscCall(PetscHeaderCreate(_term, TAOTERM_CLASSID, "TaoTerm", "Objective function term", "Tao", comm, TaoTermDestroy, TaoTermView));
  PetscCall(MatCreate(comm, &_term->solution_factory));
  PetscCall(MatSetType(_term->solution_factory, MATDUMMY));
  PetscCall(MatCreate(comm, &_term->parameters_factory));
  PetscCall(MatSetType(_term->parameters_factory, MATDUMMY));
  PetscCall(PetscObjectReference((PetscObject)_term->parameters_factory));
  _term->parameters_factory_orig = _term->parameters_factory;
  PetscCall(PetscLayoutCreateFromSizes(comm, 0, 0, 1, &zero_layout));
  PetscCall(MatGetLayouts(_term->solution_factory, &rlayout, &clayout));
  PetscCall(MatSetLayouts(_term->solution_factory, rlayout, zero_layout));
  PetscCall(MatGetLayouts(_term->parameters_factory, &rlayout, &clayout));
  PetscCall(MatSetLayouts(_term->parameters_factory, rlayout, zero_layout));
  PetscCall(PetscLayoutDestroy(&zero_layout));
  _term->ngrad_mffd   = 0;
  _term->nobj         = 0;
  _term->ngrad        = 0;
  _term->nobjgrad     = 0;
  _term->nhess        = 0;
  _term->Hpre_is_H    = PETSC_TRUE;
  _term->fd_delta     = 0.5 * PETSC_SQRT_MACHINE_EPSILON;
  _term->H_mattype    = NULL;
  _term->Hpre_mattype = NULL;
  *term               = _term;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermComputeObjective - Evaluate a `TaoTerm` for a given solution vector and parameter vector

  Collective

  Input Parameters:
+ term   - a `TaoTerm` representing a parametric function $f(x; p)$
. x      - the solution variable $x$ in $f(x; p)$
- params - the parameters $p$ in $f(x; p)$ (may be `NULL` if the term is not parametric)

  Output Parameter:
. value - the value of $f(x; p)$

  Level: developer

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermComputeGradient()`,
          `TaoTermComputeObjectiveAndGradient()`,
          `TaoTermComputeHessian()`,
          `TaoTermShellSetObjective()`
@*/
PetscErrorCode TaoTermComputeObjective(TaoTerm term, Vec x, Vec params, PetscReal *value)
{
  PetscBool obj, objgrad;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 2);
  PetscCheckSameComm(term, 1, x, 2);
  PetscAssertPointer(value, 4);
  PetscCheck(term->parameters_mode != TAOTERM_PARAMETERS_NONE || params == NULL, PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_WRONG, "Parameters passed to a TaoTerm with TAOTERM_PARAMETERS_NONE");
  PetscCheck(term->parameters_mode != TAOTERM_PARAMETERS_REQUIRED || params, PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_WRONG, "Parameters required but not provided for a TaoTerm with TAOTERM_PARAMETERS_REQUIRED");
  if (params) {
    PetscValidHeaderSpecific(params, VEC_CLASSID, 3);
    PetscCheckSameComm(term, 1, params, 3);
    PetscCall(VecLockReadPush(params));
  }
  PetscCall(VecLockReadPush(x));
  PetscCall(TaoTermIsObjectiveDefined(term, &obj));
  PetscCall(TaoTermIsObjectiveAndGradientDefined(term, &objgrad));
  if (obj) {
    PetscCall(PetscLogEventBegin(TAOTERM_ObjectiveEval, term, NULL, NULL, NULL));
    PetscUseTypeMethod(term, objective, x, params, value);
    PetscCall(PetscLogEventEnd(TAOTERM_ObjectiveEval, term, NULL, NULL, NULL));
    term->nobj++;
  } else if (objgrad) {
    Vec temp;

    PetscCall(PetscInfo(term, "%s: Duplicating solution vector in order to call objective/gradient routine\n", ((PetscObject)term)->prefix));
    PetscCall(VecDuplicate(x, &temp));
    PetscCall(PetscLogEventBegin(TAOTERM_ObjGradEval, term, NULL, NULL, NULL));
    PetscUseTypeMethod(term, objectiveandgradient, x, params, value, temp);
    PetscCall(PetscLogEventEnd(TAOTERM_ObjGradEval, term, NULL, NULL, NULL));
    PetscCall(VecDestroy(&temp));
    term->nobjgrad++;
  } else SETERRQ(PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_WRONGSTATE, "TaoTerm does not have an objective function");
  if (params) PetscCall(VecLockReadPop(params));
  PetscCall(VecLockReadPop(x));
  PetscCall(PetscInfo(term, "%s: TaoTerm value: %20.19e\n", ((PetscObject)term)->prefix, (double)(*value)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermComputeGradient - Evaluate the gradient of a `TaoTerm` for a given solution vector and parameter vector

  Collective

  Input Parameters:
+ term   - a `TaoTerm` representing a parametric function $f(x; p)$
. x      - the solution variable $x$ in $f(x; p)$
- params - the parameters $p$ in $f(x; p)$ (may be NULL if the term is not parametric)

  Output Parameter:
. g - the value of $\nabla_x f(x; p)$

  Level: developer

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermComputeObjective()`,
          `TaoTermComputeObjectiveAndGradient()`,
          `TaoTermComputeHessian()`,
          `TaoTermShellSetGradient()`
@*/
PetscErrorCode TaoTermComputeGradient(TaoTerm term, Vec x, Vec params, Vec g)
{
  PetscBool objgrad, grad;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(g, VEC_CLASSID, 4);
  PetscCheckSameComm(term, 1, x, 2);
  PetscCheckSameComm(term, 1, g, 4);
  VecCheckSameSize(x, 2, g, 4);
  PetscCheck(term->parameters_mode != TAOTERM_PARAMETERS_NONE || params == NULL, PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_WRONG, "Parameters passed to a TaoTerm with TAOTERM_PARAMETERS_NONE");
  PetscCheck(term->parameters_mode != TAOTERM_PARAMETERS_REQUIRED || params, PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_WRONG, "Parameters required but not provided for a TaoTerm with TAOTERM_PARAMETERS_REQUIRED");
  if (params) {
    PetscValidHeaderSpecific(params, VEC_CLASSID, 3);
    PetscCheckSameComm(term, 1, params, 3);
    PetscCall(VecLockReadPush(params));
  }
  PetscCall(VecLockReadPush(x));
  PetscCall(TaoTermIsGradientDefined(term, &grad));
  PetscCall(TaoTermIsObjectiveAndGradientDefined(term, &objgrad));
  if (term->fd_gradient) {
    PetscCall(PetscLogEventBegin(TAOTERM_GradientEval, term, NULL, NULL, NULL));
    PetscCall(TaoTermComputeGradientFD(term, x, params, g));
    PetscCall(PetscLogEventEnd(TAOTERM_GradientEval, term, NULL, NULL, NULL));
    term->ngrad++;
  } else if (grad) {
    PetscCall(PetscLogEventBegin(TAOTERM_GradientEval, term, NULL, NULL, NULL));
    PetscUseTypeMethod(term, gradient, x, params, g);
    PetscCall(PetscLogEventEnd(TAOTERM_GradientEval, term, NULL, NULL, NULL));
    term->ngrad++;
  } else if (objgrad) {
    PetscReal value;

    PetscCall(PetscLogEventBegin(TAOTERM_ObjGradEval, term, NULL, NULL, NULL));
    PetscUseTypeMethod(term, objectiveandgradient, x, params, &value, g);
    PetscCall(PetscLogEventEnd(TAOTERM_ObjGradEval, term, NULL, NULL, NULL));
    term->nobjgrad++;
  } else SETERRQ(PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_WRONGSTATE, "TaoTerm does not have a gradient function");
  if (params) PetscCall(VecLockReadPop(params));
  PetscCall(VecLockReadPop(x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermComputeObjectiveAndGradient - Evaluate both the value and gradient of
  a `TaoTerm` for a given set of solution vector and parameter vector

  Collective

  Input Parameters:
+ term   - a `TaoTerm` representing a parametric function $f(x; p)$
. x      - the solution variable $x$ in $f(x; p)$
- params - the parameters $p$ in $f(x; p)$ (may be NULL if the term is not parametric)

  Output Parameters:
+ value - the value of $f(x; p)$
- g     - the value of $\nabla_x f(x; p)$

  Level: developer

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermComputeObjective()`,
          `TaoTermComputeGradient()`,
          `TaoTermComputeHessian()`,
          `TaoTermShellSetObjectiveAndGradient()`
@*/
PetscErrorCode TaoTermComputeObjectiveAndGradient(TaoTerm term, Vec x, Vec params, PetscReal *value, Vec g)
{
  PetscBool objgrad, obj, grad;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(g, VEC_CLASSID, 5);
  PetscAssertPointer(value, 4);
  PetscCheckSameComm(term, 1, x, 2);
  PetscCheckSameComm(term, 1, g, 5);
  VecCheckSameSize(x, 2, g, 5);
  PetscCheck(term->parameters_mode != TAOTERM_PARAMETERS_NONE || params == NULL, PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_WRONG, "Parameters passed to a TaoTerm with TAOTERM_PARAMETERS_NONE");
  PetscCheck(term->parameters_mode != TAOTERM_PARAMETERS_REQUIRED || params, PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_WRONG, "Parameters required but not provided for a TaoTerm with TAOTERM_PARAMETERS_REQUIRED");
  if (params) {
    PetscValidHeaderSpecific(params, VEC_CLASSID, 3);
    PetscCheckSameComm(term, 1, params, 3);
    PetscCall(VecLockReadPush(params));
  }
  PetscCall(VecLockReadPush(x));
  PetscCall(TaoTermIsObjectiveDefined(term, &obj));
  PetscCall(TaoTermIsGradientDefined(term, &grad));
  PetscCall(TaoTermIsObjectiveAndGradientDefined(term, &objgrad));
  if (term->fd_gradient) {
    PetscCall(TaoTermComputeObjective(term, x, params, value));
    PetscCall(PetscLogEventBegin(TAOTERM_GradientEval, term, NULL, NULL, NULL));
    PetscCall(TaoTermComputeGradientFD(term, x, params, g));
    PetscCall(PetscLogEventEnd(TAOTERM_GradientEval, term, NULL, NULL, NULL));
    term->ngrad++;
  } else if (objgrad) {
    PetscCall(PetscLogEventBegin(TAOTERM_ObjGradEval, term, NULL, NULL, NULL));
    PetscUseTypeMethod(term, objectiveandgradient, x, params, value, g);
    PetscCall(PetscLogEventEnd(TAOTERM_ObjGradEval, term, NULL, NULL, NULL));
    term->nobjgrad++;
  } else if (obj && grad) {
    PetscCall(PetscLogEventBegin(TAOTERM_ObjectiveEval, term, NULL, NULL, NULL));
    PetscUseTypeMethod(term, objective, x, params, value);
    PetscCall(PetscLogEventEnd(TAOTERM_ObjectiveEval, term, NULL, NULL, NULL));
    term->nobj++;
    PetscCall(PetscLogEventBegin(TAOTERM_GradientEval, term, NULL, NULL, NULL));
    PetscUseTypeMethod(term, gradient, x, params, g);
    PetscCall(PetscLogEventEnd(TAOTERM_GradientEval, term, NULL, NULL, NULL));
    term->ngrad++;
  } else SETERRQ(PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_WRONGSTATE, "TaoTerm does not have objective and gradient function");
  if (params) PetscCall(VecLockReadPop(params));
  PetscCall(VecLockReadPop(x));
  PetscCall(PetscInfo(term, "%s: TaoTerm value: %20.19e\n", ((PetscObject)term)->prefix, (double)(*value)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermComputeHessian - Evaluate the Hessian of a `TaoTerm`
  (with respect to the solution variables) for a given solution vector and parameter vector

  Collective

  Input Parameters:
+ term   - a `TaoTerm` representing a parametric function $f(x; p)$
. x      - the solution variable $x$ in $f(x; p)$
- params - the parameters $p$ in $f(x; p)$ (may be `NULL` if the term is not parametric)

  Output Parameters:
+ H    - Hessian matrix $\nabla_x^2 f(x;p)$
- Hpre - an (approximate) Hessian from which the preconditioner will be constructed, often the same as `H`

  Level: developer

  Note:
  If there is no separate matrix from which to construct the preconditioner, then `TaoTermComputeHessian(term, x, params, H, NULL)`
  and `TaoTermComputeHessian(term, x, params, H, H)` are equivalent.

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermComputeObjective()`,
          `TaoTermComputeGradient()`,
          `TaoTermComputeObjectiveAndGradient()`,
          `TaoTermShellSetHessian()`
@*/
PetscErrorCode TaoTermComputeHessian(TaoTerm term, Vec x, Vec params, Mat H, Mat Hpre)
{
  PetscBool  is_mffd       = PETSC_FALSE;
  PetscBool3 is_fdpossible = PETSC_BOOL3_UNKNOWN;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 2);
  PetscCheckSameComm(term, 1, x, 2);
  PetscCheck(term->parameters_mode != TAOTERM_PARAMETERS_NONE || params == NULL, PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_WRONG, "Parameters passed to a TaoTerm with TAOTERM_PARAMETERS_NONE");
  PetscCheck(term->parameters_mode != TAOTERM_PARAMETERS_REQUIRED || params, PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_WRONG, "Parameters required but not provided for a TaoTerm with TAOTERM_PARAMETERS_REQUIRED");
  if (params) {
    PetscValidHeaderSpecific(params, VEC_CLASSID, 3);
    PetscCheckSameComm(term, 1, params, 3);
    PetscCall(VecLockReadPush(params));
  }
  PetscCall(VecLockReadPush(x));
  if (H) {
    PetscValidHeaderSpecific(H, MAT_CLASSID, 4);
    PetscCheckSameComm(term, 1, H, 4);
  }
  if (Hpre) {
    PetscValidHeaderSpecific(Hpre, MAT_CLASSID, 5);
    PetscCheckSameComm(term, 1, Hpre, 5);
  }
  if (H) PetscCall(PetscObjectTypeCompare((PetscObject)H, MATMFFD, &is_mffd));
  PetscCall(TaoTermIsComputeHessianFDPossible(term, &is_fdpossible));
  PetscCall(PetscLogEventBegin(TAOTERM_HessianEval, term, NULL, NULL, NULL));
  if (is_fdpossible == PETSC_BOOL3_FALSE) {
    PetscUseTypeMethod(term, hessian, x, params, H, Hpre);
  } else if (term->fd_hessian) {
    if (is_fdpossible == PETSC_BOOL3_UNKNOWN) PetscCall(PetscInfo(term, "%s: Whether TaoTermComputeHessianFD is possible is unknown. Trying anyway.\n", ((PetscObject)term)->prefix));
    PetscCall(TaoTermComputeHessianFD(term, x, params, H, Hpre));
  } else if (is_mffd) {
    if (is_fdpossible == PETSC_BOOL3_UNKNOWN) PetscCall(PetscInfo(term, "%s: Whether TaoTermComputeHessianMFFD is possible is unknown. Trying anyway.\n", ((PetscObject)term)->prefix));
    PetscCall(TaoTermComputeHessianMFFD(term, x, params, H, Hpre));
  } else {
    if (term->ops->hessian) PetscUseTypeMethod(term, hessian, x, params, H, Hpre);
    else
      SETERRQ(PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_WRONGSTATE, "TaoTerm does not have TaoTermComputeHessian routine, and cannot use finite differences for Hessian computation. Either set Hessian MatType to MATMFFD, or call TaoTermComputeHessianSetUseFD()");
  }
  PetscCall(PetscLogEventEnd(TAOTERM_HessianEval, term, NULL, NULL, NULL));
  term->nhess++;
  if (params) PetscCall(VecLockReadPop(params));
  PetscCall(VecLockReadPop(x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermIsComputeHessianFDPossible - Whether this term can compute Hessian with finite differences
  with either `-tao_term_hessian_use_fd`, `TaoTermComputeHessianSetUseFD()`, or `MATMFFD`.

  Not collective

  Input Parameter:
. term - a `TaoTerm`

  Output Parameter:
. is_fdpossible - whether Hessian computation with finite differences is possible

  Level: developer

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermComputeObjective()`,
          `TaoTermShellSetObjective()`,
          `TaoTermIsGradientDefined()`,
          `TaoTermIsObjectiveAndGradientDefined()`,
          `TaoTermIsHessianDefined()`
@*/
PetscErrorCode TaoTermIsComputeHessianFDPossible(TaoTerm term, PetscBool3 *is_fdpossible)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscAssertPointer(is_fdpossible, 2);
  if (term->ops->iscomputehessianfdpossible) PetscUseTypeMethod(term, iscomputehessianfdpossible, is_fdpossible);
  else *is_fdpossible = PETSC_BOOL3_UNKNOWN;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermIsObjectiveDefined - Whether a standalone objective operation is defined for this `TaoTerm`

  Not collective

  Input Parameter:
. term - a `TaoTerm`

  Output Parameter:
. is_defined - whether the objective is defined

  Note:
  This function strictly checks whether a dedicated objective operation is defined. It does not check whether the
  objective could be computed via other operations (e.g., an objective-and-gradient callback). `TaoTermComputeObjective()`
  may still succeed even if this function returns `PETSC_FALSE`, by falling back to `TaoTermComputeObjectiveAndGradient()`.

  Level: developer

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermComputeObjective()`,
          `TaoTermShellSetObjective()`,
          `TaoTermIsGradientDefined()`,
          `TaoTermIsObjectiveAndGradientDefined()`,
          `TaoTermIsHessianDefined()`
@*/
PetscErrorCode TaoTermIsObjectiveDefined(TaoTerm term, PetscBool *is_defined)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscAssertPointer(is_defined, 2);
  if (term->ops->isobjectivedefined) PetscUseTypeMethod(term, isobjectivedefined, is_defined);
  else *is_defined = (term->ops->objective != NULL) ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermIsGradientDefined - Whether a standalone gradient operation is defined for this `TaoTerm`

  Not collective

  Input Parameter:
. term - a `TaoTerm`

  Output Parameter:
. is_defined - whether the gradient is defined

  Note:
  This function strictly checks whether a dedicated gradient operation is defined. It does not check whether the
  gradient could be computed via other operations (e.g., an objective-and-gradient callback or finite differences).
  `TaoTermComputeGradient()` may still succeed even if this function returns `PETSC_FALSE`, by falling back to
  `TaoTermComputeObjectiveAndGradient()` or finite-difference approximation.

  Level: developer

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermComputeGradient()`,
          `TaoTermShellSetGradient()`,
          `TaoTermIsObjectiveDefined()`,
          `TaoTermIsObjectiveAndGradientDefined()`,
          `TaoTermIsHessianDefined()`
@*/
PetscErrorCode TaoTermIsGradientDefined(TaoTerm term, PetscBool *is_defined)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscAssertPointer(is_defined, 2);
  if (term->ops->isgradientdefined) PetscUseTypeMethod(term, isgradientdefined, is_defined);
  else *is_defined = (term->ops->gradient != NULL) ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermIsObjectiveAndGradientDefined - Whether a combined objective-and-gradient operation is defined for this `TaoTerm`

  Not collective

  Input Parameter:
. term - a `TaoTerm`

  Output Parameter:
. is_defined - whether the objective/gradient is defined

  Note:
  This function strictly checks whether a dedicated combined objective-and-gradient operation is defined. It does not
  check whether the objective and gradient could be computed via separate objective and gradient operations.
  `TaoTermComputeObjectiveAndGradient()` may still succeed even if this function returns `PETSC_FALSE`, by falling back
  to separate `TaoTermComputeObjective()` and `TaoTermComputeGradient()` calls.

  Level: developer

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermComputeObjectiveAndGradient()`,
          `TaoTermShellSetObjectiveAndGradient()`,
          `TaoTermIsObjectiveDefined()`,
          `TaoTermIsGradientDefined()`,
          `TaoTermIsHessianDefined()`
@*/
PetscErrorCode TaoTermIsObjectiveAndGradientDefined(TaoTerm term, PetscBool *is_defined)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscAssertPointer(is_defined, 2);
  if (term->ops->isobjectiveandgradientdefined) PetscUseTypeMethod(term, isobjectiveandgradientdefined, is_defined);
  else *is_defined = (term->ops->objectiveandgradient != NULL) ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermIsHessianDefined - Whether a Hessian operation is defined for this `TaoTerm`

  Not collective

  Input Parameter:
. term - a `TaoTerm`

  Output Parameter:
. is_defined - whether the Hessian is defined

  Note:
  This function strictly checks whether a dedicated Hessian operation is defined. It does not check whether the
  Hessian could be computed via finite differences. `TaoTermComputeHessian()` may still succeed even if this function
  returns `PETSC_FALSE`, if finite-difference Hessian computation has been enabled.

  Level: developer

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermComputeHessian()`,
          `TaoTermShellSetHessian()`,
          `TaoTermIsObjectiveDefined()`,
          `TaoTermIsGradientDefined()`,
          `TaoTermIsObjectiveAndGradientDefined()`
@*/
PetscErrorCode TaoTermIsHessianDefined(TaoTerm term, PetscBool *is_defined)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscAssertPointer(is_defined, 2);
  if (term->ops->ishessiandefined) PetscUseTypeMethod(term, ishessiandefined, is_defined);
  else *is_defined = (term->ops->hessian != NULL) ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermIsCreateHessianMatricesDefined - Whether this term can call `TaoTermCreateHessianMatrices()`.

  Not collective

  Input Parameter:
. term - a `TaoTerm`

  Output Parameter:
. is_defined - whether the term can create new Hessian matrices

  Level: developer

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermCreateHessianMatrices()`,
          `TaoTermShellSetCreateHessianMatrices()`,
          `TaoTermIsObjectiveDefined()`,
          `TaoTermIsGradientDefined()`,
          `TaoTermIsObjectiveAndGradientDefined()`,
          `TaoTermIsHessianDefined()`
@*/
PetscErrorCode TaoTermIsCreateHessianMatricesDefined(TaoTerm term, PetscBool *is_defined)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscAssertPointer(is_defined, 2);
  if (term->ops->iscreatehessianmatricesdefined) PetscUseTypeMethod(term, iscreatehessianmatricesdefined, is_defined);
  else *is_defined = (term->ops->createhessianmatrices != NULL) ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermSetSolutionSizes - Set the sizes describing the layout of the solution vector space of a `TaoTerm`.

  Logically collective

  Input Parameters:
+ term - a `TaoTerm`
. n    - the size of a solution vector on the current MPI process (or `PETSC_DECIDE`)
. N    - the global size of a solution vector (or `PETSC_DECIDE`)
- bs   - the block size of a solution vector (must be >= 1)

  Level: beginner

  Notes:
  The "solution space" of a `TaoTerm` is the vector space of the optimization variable $x$ in
  $f(x; p)$. This is distinct from the "parameter space" (the space of the fixed data $p$, set
  with `TaoTermSetParametersSizes()`). Some `TaoTermType`s require the solution and parameter
  spaces to be related (e.g., have the same size); see the documentation for each type.

  When a mapping matrix $A$ is used to add a term to a `Tao` via `TaoAddTerm()`, the mapping
  transforms the `Tao` solution vector into this term's solution space.  For example, if the
  `Tao` solution vector is $x \in \mathbb{R}^n$ and the mapping matrix is $A \in \mathbb{R}^{m \times n}$,
  then the term evaluates $f(Ax; p)$ with $Ax \in \mathbb{R}^m$.  The term's solution space is
  therefore $\mathbb{R}^m$, and `TaoTermView()` will report $N = m$ for this term.

  Alternatively, one may use `TaoTermSetSolutionLayout()` or `TaoTermSetSolutionTemplate()` to define the vector sizes.

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermGetSolutionSizes()`,
          `TaoTermSetSolutionTemplate()`,
          `TaoTermGetSolutionVecType()`,
          `TaoTermSetSolutionVecType()`,
          `TaoTermGetSolutionLayout()`,
          `TaoTermSetSolutionLayout()`,
          `TaoTermCreateSolutionVec()`
@*/
PetscErrorCode TaoTermSetSolutionSizes(TaoTerm term, PetscInt n, PetscInt N, PetscInt bs)
{
  PetscLayout layout;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscCall(MatGetLayouts(term->solution_factory, &layout, NULL));
  PetscCall(PetscLayoutSetLocalSize(layout, n));
  PetscCall(PetscLayoutSetSize(layout, N));
  PetscCall(PetscLayoutSetBlockSize(layout, bs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermGetSolutionSizes - Get the sizes describing the layout of the solution vector space of a `TaoTerm`.

  Not collective

  Input Parameter:
. term - a `TaoTerm`

  Output Parameters:
+ n  - (optional) the size of a solution vector on the current MPI process
. N  - (optional) the global size of a solution vector
- bs - (optional) the block size of a solution vector

  Level: beginner

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermSetSolutionSizes()`,
          `TaoTermSetSolutionTemplate()`,
          `TaoTermGetSolutionVecType()`,
          `TaoTermSetSolutionVecType()`,
          `TaoTermGetSolutionLayout()`,
          `TaoTermSetSolutionLayout()`,
          `TaoTermCreateSolutionVec()`
@*/
PetscErrorCode TaoTermGetSolutionSizes(TaoTerm term, PetscInt *n, PetscInt *N, PetscInt *bs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  if (n) PetscCall(MatGetLocalSize(term->solution_factory, n, NULL));
  if (N) PetscCall(MatGetSize(term->solution_factory, N, NULL));
  if (bs) PetscCall(MatGetBlockSizes(term->solution_factory, bs, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermSetParametersSizes - Set the sizes describing the layout of the parameter vector space of a `TaoTerm`.

  Logically collective

  Input Parameters:
+ term - a `TaoTerm`
. k    - the size of a parameter vector on the current MPI process (or `PETSC_DECIDE`)
. K    - the global size of a parameter vector (or `PETSC_DECIDE`)
- bs   - the block size of a parameter vector (must be >= 1)

  Level: beginner

  Notes:
  The "parameter space" of a `TaoTerm` is the vector space of the fixed data $p$ in $f(x; p)$.
  Parameters are not optimized over. This is distinct from the "solution space" (set with
  `TaoTermSetSolutionSizes()`), which is the space of the optimization variable $x$.
  Some `TaoTermType`s require the solution and parameter spaces to be related (e.g., have the same size);
  see the documentation for each type.

  Alternatively, one may use `TaoTermSetParametersLayout()` or `TaoTermSetParametersTemplate()` to define the vector sizes.

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermGetParametersSizes()`,
          `TaoTermSetParametersTemplate()`,
          `TaoTermGetParametersVecType()`,
          `TaoTermSetParametersVecType()`,
          `TaoTermGetParametersLayout()`,
          `TaoTermSetParametersLayout()`,
          `TaoTermCreateParametersVec()`
@*/
PetscErrorCode TaoTermSetParametersSizes(TaoTerm term, PetscInt k, PetscInt K, PetscInt bs)
{
  PetscLayout layout;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscCall(MatGetLayouts(term->parameters_factory, &layout, NULL));
  PetscCall(PetscLayoutSetLocalSize(layout, k));
  PetscCall(PetscLayoutSetSize(layout, K));
  PetscCall(PetscLayoutSetBlockSize(layout, bs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermGetParametersSizes - Get the sizes describing the layout of the parameter vector space of a `TaoTerm`.

  Not collective

  Input Parameter:
. term - a `TaoTerm`

  Output Parameters:
+ k  - (optional) the size of a parameter vector on the current MPI process
. K  - (optional) the global size of a parameter vector
- bs - (optional) the block size of a parameter vector

  Level: beginner

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermSetParametersSizes()`,
          `TaoTermSetParametersTemplate()`,
          `TaoTermGetParametersVecType()`,
          `TaoTermSetParametersVecType()`,
          `TaoTermGetParametersLayout()`,
          `TaoTermSetParametersLayout()`,
          `TaoTermCreateParametersVec()`
@*/
PetscErrorCode TaoTermGetParametersSizes(TaoTerm term, PetscInt *k, PetscInt *K, PetscInt *bs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  if (k) PetscCall(MatGetLocalSize(term->parameters_factory, k, NULL));
  if (K) PetscCall(MatGetSize(term->parameters_factory, K, NULL));
  if (bs) PetscCall(MatGetBlockSizes(term->parameters_factory, bs, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermSetParametersLayout - Set the layout describing the parameter vector of `TaoTerm`.

  Collective

  Input Parameters:
+ term              - a `TaoTerm`
- parameters_layout - the `PetscLayout` for the parameter space

  Level: intermediate

  Notes:
  The "parameter space" of a `TaoTerm` is the vector space of the fixed data $p$ in $f(x; p)$.
  Parameters are not optimized over. This is distinct from the "solution space" (set with
  `TaoTermSetSolutionSizes()`), which is the space of the optimization variable $x$.
  Some `TaoTermType`s require the solution and parameter spaces to be related (e.g., have the same size);
  see the documentation for each type.

  Alternatively, one may use `TaoTermSetParametersSizes()` or `TaoTermSetParametersTemplate()` to define the vector sizes.

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermGetParametersVecType()`,
          `TaoTermSetParametersVecType()`,
          `TaoTermGetParametersLayout()`,
          `TaoTermSetSolutionTemplate()`,
          `TaoTermSetParametersTemplate()`,
          `TaoTermCreateParametersVec()`
@*/
PetscErrorCode TaoTermSetParametersLayout(TaoTerm term, PetscLayout parameters_layout)
{
  PetscLayout rlayout, clayout;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscCall(MatGetLayouts(term->parameters_factory, &rlayout, &clayout));
  PetscCall(MatSetLayouts(term->parameters_factory, parameters_layout, clayout));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermGetParametersLayout - Get the layouts describing the parameter vectors of a `TaoTerm`.

  Not collective

  Input Parameter:
. term - a `TaoTerm`

  Output Parameter:
. parameters_layout - the `PetscLayout` for the parameter space

  Level: intermediate

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermGetParametersVecType()`,
          `TaoTermSetParametersVecType()`,
          `TaoTermSetParametersLayout()`,
          `TaoTermSetSolutionTemplate()`,
          `TaoTermSetParametersTemplate()`,
          `TaoTermCreateParametersVec()`
@*/
PetscErrorCode TaoTermGetParametersLayout(TaoTerm term, PetscLayout *parameters_layout)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscAssertPointer(parameters_layout, 2);
  PetscCall(MatGetLayouts(term->parameters_factory, parameters_layout, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermSetSolutionLayout - Set the layout describing the solution vector of `TaoTerm`.

  Collective

  Input Parameters:
+ term            - a `TaoTerm`
- solution_layout - the `PetscLayout` for the solution space

  Level: intermediate

  Notes:
  The "solution space" of a `TaoTerm` is the vector space of the optimization variable $x$ in
  $f(x; p)$. This is distinct from the "parameter space" (the space of the fixed data $p$, set
  with `TaoTermSetParametersSizes()`). Some `TaoTermType`s require the solution and parameter
  spaces to be related (e.g., have the same size); see the documentation for each type.

  When a mapping matrix $A$ is used to add a term to a `Tao` via `TaoAddTerm()`, the mapping
  transforms the `Tao` solution vector into this term's solution space.  For example, if the
  `Tao` solution vector is $x \in \mathbb{R}^n$ and the mapping matrix is $A \in \mathbb{R}^{m \times n}$,
  then the term evaluates $f(Ax; p)$ with $Ax \in \mathbb{R}^m$.  The term's solution space is
  therefore $\mathbb{R}^m$, and `TaoTermView()` will report $N = m$ for this term.

  Alternatively, one may use `TaoTermSetSolutionSizes()` or `TaoTermSetSolutionTemplate()` to define the vector sizes.

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermGetSolutionVecType()`,
          `TaoTermSetSolutionVecType()`,
          `TaoTermGetSolutionLayout()`,
          `TaoTermSetSolutionTemplate()`,
          `TaoTermSetParametersTemplate()`,
          `TaoTermCreateSolutionVec()`
@*/
PetscErrorCode TaoTermSetSolutionLayout(TaoTerm term, PetscLayout solution_layout)
{
  PetscLayout rlayout, clayout;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscCall(MatGetLayouts(term->solution_factory, &rlayout, &clayout));
  PetscCall(MatSetLayouts(term->solution_factory, solution_layout, clayout));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermGetSolutionLayout - Get the layouts describing the solution vectors of a `TaoTerm`.

  Not collective

  Input Parameter:
. term - a `TaoTerm`

  Output Parameter:
. solution_layout - the `PetscLayout` for the solution space

  Level: intermediate

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermGetSolutionVecType()`,
          `TaoTermSetSolutionVecType()`,
          `TaoTermSetSolutionLayout()`,
          `TaoTermSetSolutionTemplate()`,
          `TaoTermSetParametersTemplate()`,
          `TaoTermCreateSolutionVec()`
@*/
PetscErrorCode TaoTermGetSolutionLayout(TaoTerm term, PetscLayout *solution_layout)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscAssertPointer(solution_layout, 2);
  PetscCall(MatGetLayouts(term->solution_factory, solution_layout, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermSetSolutionTemplate - Set the solution vector space to match a template vector

  Collective

  Input Parameters:
+ term         - a `TaoTerm`
- sol_template - a vector with the desired size, layout, and `VecType` of solution vectors for `TaoTerm`

  Level: intermediate

  Notes:
  The "solution space" of a `TaoTerm` is the vector space of the optimization variable $x$ in
  $f(x; p)$. This is distinct from the "parameter space" (the space of the fixed data $p$, set
  with `TaoTermSetParametersSizes()`). Some `TaoTermType`s require the solution and parameter
  spaces to be related (e.g., have the same size); see the documentation for each type.

  When a mapping matrix $A$ is used to add a term to a `Tao` via `TaoAddTerm()`, the mapping
  transforms the `Tao` solution vector into this term's solution space.  For example, if the
  `Tao` solution vector is $x \in \mathbb{R}^n$ and the mapping matrix is $A \in \mathbb{R}^{m \times n}$,
  then the term evaluates $f(Ax; p)$ with $Ax \in \mathbb{R}^m$.  The term's solution space is
  therefore $\mathbb{R}^m$, and `TaoTermView()` will report $N = m$ for this term.

  Alternatively, one may use `TaoTermSetSolutionSizes()` or `TaoTermSetSolutionLayout()` to define the vector sizes.

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermGetSolutionVecType()`,
          `TaoTermSetSolutionVecType()`,
          `TaoTermSetParametersTemplate()`,
          `TaoTermGetSolutionLayout()`,
          `TaoTermSetSolutionLayout()`,
          `TaoTermCreateSolutionVec()`
@*/
PetscErrorCode TaoTermSetSolutionTemplate(TaoTerm term, Vec sol_template)
{
  PetscLayout layout, clayout;
  VecType     vec_type;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscValidHeaderSpecific(sol_template, VEC_CLASSID, 2);
  PetscCheckSameComm(term, 1, sol_template, 2);
  PetscCall(VecGetType(sol_template, &vec_type));
  PetscCall(VecGetLayout(sol_template, &layout));
  PetscCall(MatGetLayouts(term->solution_factory, NULL, &clayout));
  PetscCall(MatSetLayouts(term->solution_factory, layout, clayout));
  PetscCall(MatSetVecType(term->solution_factory, vec_type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermSetParametersTemplate - Set the parameter vector space to match a template vector

  Collective

  Input Parameters:
+ term            - a `TaoTerm`
- params_template - a vector with the desired size, layout, and `VecType` of parameter vectors for `TaoTerm`

  Level: intermediate

  Notes:
  The "parameter space" of a `TaoTerm` is the vector space of the fixed data $p$ in $f(x; p)$.
  Parameters are not optimized over. This is distinct from the "solution space" (set with
  `TaoTermSetSolutionSizes()`), which is the space of the optimization variable $x$.
  Some `TaoTermType`s require the solution and parameter spaces to be related (e.g., have the same size);
  see the documentation for each type.

  Alternatively, one may use `TaoTermSetParametersSizes()` or `TaoTermSetParametersLayout()` to define the vector sizes.

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermGetParametersVecType()`,
          `TaoTermSetParametersVecType()`,
          `TaoTermSetSolutionTemplate()`,
          `TaoTermGetParametersLayout()`,
          `TaoTermSetParametersLayout()`,
          `TaoTermCreateSolutionVec()`
@*/
PetscErrorCode TaoTermSetParametersTemplate(TaoTerm term, Vec params_template)
{
  PetscLayout layout, clayout;
  VecType     vec_type;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscValidHeaderSpecific(params_template, VEC_CLASSID, 2);
  PetscCheckSameComm(term, 1, params_template, 2);
  PetscCall(VecGetType(params_template, &vec_type));
  PetscCall(VecGetLayout(params_template, &layout));
  PetscCall(MatGetLayouts(term->parameters_factory, NULL, &clayout));
  PetscCall(MatSetLayouts(term->parameters_factory, layout, clayout));
  PetscCall(MatSetVecType(term->parameters_factory, vec_type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermSetSolutionVecType - Set the vector types of the solution vector of a `TaoTerm`

  Logically collective

  Input Parameters:
+ term          - a `TaoTerm`
- solution_type - the `VecType` for the solution space

  Options Database Keys:
. -tao_term_solution_vec_type <type> - `VecType` for complete list of vector types

  Level: advanced

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermGetSolutionVecType()`,
          `TaoTermSetSolutionLayout()`,
          `TaoTermGetSolutionLayout()`,
          `TaoTermSetSolutionTemplate()`,
          `TaoTermSetParametersTemplate()`,
          `TaoTermCreateSolutionVec()`
@*/
PetscErrorCode TaoTermSetSolutionVecType(TaoTerm term, VecType solution_type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  if (solution_type) PetscCall(MatSetVecType(term->solution_factory, solution_type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermSetParametersVecType - Set the vector types of the parameters vector of a `TaoTerm`

  Logically collective

  Input Parameters:
+ term            - a `TaoTerm`
- parameters_type - the `VecType` for the parameters space

  Options Database Keys:
. -tao_term_parameters_vec_type <type> - `VecType` for complete list of vector types

  Level: advanced

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermGetParametersVecType()`,
          `TaoTermSetParametersLayout()`,
          `TaoTermGetParametersLayout()`,
          `TaoTermSetParametersTemplate()`,
          `TaoTermCreateParametersVec()`
@*/
PetscErrorCode TaoTermSetParametersVecType(TaoTerm term, VecType parameters_type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  if (parameters_type) PetscCall(MatSetVecType(term->parameters_factory, parameters_type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermGetSolutionVecType - Get the vector types of the solution vector of a `TaoTerm`

  Not collective

  Input Parameter:
. term - a `TaoTerm`

  Output Parameter:
. solution_type - the `VecType` for the solution space

  Level: advanced

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermSetSolutionVecType()`,
          `TaoTermGetSolutionLayout()`,
          `TaoTermSetSolutionLayout()`,
          `TaoTermSetSolutionTemplate()`,
          `TaoTermSetParametersTemplate()`,
          `TaoTermCreateSolutionVec()`
@*/
PetscErrorCode TaoTermGetSolutionVecType(TaoTerm term, VecType *solution_type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscAssertPointer(solution_type, 2);
  PetscCall(MatGetVecType(term->solution_factory, solution_type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermGetParametersVecType - Get the vector types of the parameter vector of a `TaoTerm`

  Not collective

  Input Parameter:
. term - a `TaoTerm`

  Output Parameter:
. parameters_type - the `VecType` for the parameter space

  Level: advanced

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermSetParametersVecType()`,
          `TaoTermGetParametersLayout()`,
          `TaoTermSetParametersLayout()`,
          `TaoTermSetSolutionTemplate()`,
          `TaoTermSetParametersTemplate()`,
          `TaoTermCreateParametersVec()`
@*/
PetscErrorCode TaoTermGetParametersVecType(TaoTerm term, VecType *parameters_type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscAssertPointer(parameters_type, 2);
  PetscCall(MatGetVecType(term->parameters_factory, parameters_type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermCreateSolutionVec - Create a solution vector for a `TaoTerm`

  Collective

  Input Parameter:
. term - a `TaoTerm`

  Output Parameter:
. solution - a compatible solution vector for `term`

  Level: advanced

  Note:
  Before a `TaoTerm` can create a solution vector, you must do one of the following\:

  * Call `TaoTermSetSolutionSizes()` to describe the size and parallel layout of a solution vector.
  * Call `TaoTermSetSolutionLayout()` to directly set `PetscLayout`s for the solution vector.
  * Call `TaoTermSetSolutionTemplate()` to set the solution vector spaces to match existing `Vec`.
  * If the `TaoTerm` is a `TAOTERMSHELL`, you can call `TaoTermShellSetCreateSolutionVec()` to use
  your own code for creating vectors.

  You can also call `TaoTermSetSolutionVecType()` to set the type of vector created (e.g. `VECCUDA`).

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermShellSetCreateSolutionVec()`,
          `TaoTermGetSolutionSizes()`,
          `TaoTermSetSolutionSizes()`,
          `TaoTermSetSolutionTemplate()`,
          `TaoTermGetSolutionVecType()`,
          `TaoTermSetSolutionVecType()`,
          `TaoTermGetSolutionLayout()`,
          `TaoTermSetSolutionLayout()`,
          `TaoTermCreateHessianMatrices()`
@*/
PetscErrorCode TaoTermCreateSolutionVec(TaoTerm term, Vec *solution)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscAssertPointer(solution, 2);
  if (term->ops->createsolutionvec) {
    PetscUseTypeMethod(term, createsolutionvec, solution);
  } else {
    PetscCall(MatCreateVecs(term->solution_factory, NULL, solution));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermCreateParametersVec - Create a parameter vector for a `TaoTerm`

  Collective

  Input Parameter:
. term - a `TaoTerm`

  Output Parameter:
. parameters - a compatible parameter vector for `term`

  Level: advanced

  Notes:
  Before a `TaoTerm` can create a parameter vector, you must do one of the following\:

  * Call `TaoTermSetParametersSizes()` to describe the size and parallel layout of a parameters vector.
  * Call `TaoTermSetParametersLayout()` to directly set `PetscLayout`s for the parameters vector.
  * Call `TaoTermSetParametersTemplate()` to set the parameters vector spaces to match existing `Vec`.
  * If the `TaoTerm` is a `TAOTERMSHELL`, you can call `TaoTermShellSetCreateParametersVec()` to use your
  own code for creating vectors.

  You can also call `TaoTermSetParametersVecType()` to set the type of vector created (e.g. `VECCUDA`).

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermShellSetCreateParametersVec()`,
          `TaoTermGetParametersSizes()`,
          `TaoTermSetParametersSizes()`,
          `TaoTermSetParametersTemplate()`,
          `TaoTermGetParametersVecType()`,
          `TaoTermSetParametersVecType()`,
          `TaoTermGetParametersLayout()`,
          `TaoTermSetParametersLayout()`,
          `TaoTermCreateHessianMatrices()`
@*/
PetscErrorCode TaoTermCreateParametersVec(TaoTerm term, Vec *parameters)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscAssertPointer(parameters, 2);
  if (term->ops->createparametersvec) {
    PetscUseTypeMethod(term, createparametersvec, parameters);
  } else {
    PetscCall(MatCreateVecs(term->parameters_factory, NULL, parameters));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermCreateHessianMatrices - Create the matrices that can be inputs to `TaoTermComputeHessian()`

  Collective

  Input Parameter:
. term - a `TaoTerm`

  Output Parameters:
+ H    - (optional) a matrix that can store the Hessian computed in `TaoTermComputeHessian()`
- Hpre - (optional) a matrix from which a preconditioner can be computed in `TaoTermComputeHessian()`

  Level: advanced

  Note:
  Before Hessian matrices can be created, the size of the solution vector space
  must be set (see the ways this can be done in `TaoTermCreateSolutionVec()`).  If the
  term is a `TAOTERMSHELL`, `TaoTermShellSetCreateHessianMatrices()` must be
  called.  Most `TaoTerm`s use `TaoTermCreateHessianMatricesDefault()` to create
  their Hessian matrices: the behavior of that function can be controlled by
  `TaoTermSetCreateHessianMode()`.

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermComputeHessian()`,
          `TaoTermShellSetCreateHessianMatrices()`,
          `TaoTermCreateSolutionVec()`,
          `TaoTermCreateHessianMatricesDefault()`,
          `TaoTermGetCreateHessianMode()`,
          `TaoTermSetCreateHessianMode()`,
          `TaoTermIsCreateHessianMatricesDefined()`
@*/
PetscErrorCode TaoTermCreateHessianMatrices(TaoTerm term, Mat *H, Mat *Hpre)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  if (H) PetscAssertPointer(H, 2);
  if (Hpre) PetscAssertPointer(Hpre, 3);
  PetscUseTypeMethod(term, createhessianmatrices, H, Hpre);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermCreateHessianMatricesDefault - Default routine for creating Hessian matrices that can be used by many `TaoTerm` implementations

  Collective

  Input Parameter:
. term - a `TaoTerm`

  Output Parameters:
+ H    - (optional) a matrix that can store the Hessian computed in `TaoTermComputeHessian()`
- Hpre - (optional) a matrix from which a preconditioner can be computed in `TaoTermComputeHessian()`

  Level: developer

  Developer Note:
  The behavior of this routine is determined by `TaoTermSetCreateHessianMode()`.
  If `Hpre_is_H`, then the same matrix will be returned for `H` and `Hpre`,
  otherwise they will be separate matrices, with the matrix types `H_mattype` and `Hpre_mattype`.
  If either type is `MATMFFD`, then it will create a shell matrix with `TaoTermCreateHessianMFFD()`.

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermComputeHessian()`,
          `TaoTermCreateHessianMatrices()`,
          `TaoTermGetCreateHessianMode()`,
          `TaoTermSetCreateHessianMode()`
@*/
PetscErrorCode TaoTermCreateHessianMatricesDefault(TaoTerm term, Mat *H, Mat *Hpre)
{
  PetscBool Hpre_is_H;
  MatType   H_mattype;
  MatType   Hpre_mattype;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscCall(TaoTermGetCreateHessianMode(term, &Hpre_is_H, &H_mattype, &Hpre_mattype));

  if (H || (Hpre && Hpre_is_H)) PetscCall(TaoTermCreateHessianMatricesDefault_H_Internal(term, H, Hpre, Hpre_is_H, H_mattype));
  if (Hpre && !Hpre_is_H) PetscCall(TaoTermCreateHessianMatricesDefault_Hpre_Internal(term, H, Hpre, Hpre_is_H, Hpre_mattype));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermSetCreateHessianMode - Determine the behavior of `TaoTermCreateHessianMatricesDefault()`.

  Logically collective

  Input Parameters:
+ term         - a `TaoTerm`
. Hpre_is_H    - should `TaoTermCreateHessianMatricesDefault()` make one matrix for `H` and `Hpre`?
. H_mattype    - the `MatType` to create for `H`
- Hpre_mattype - the `MatType` to create for `Hpre`

  Options Database Keys:
+ -tao_term_hessian_pre_is_hessian <bool> - Whether `TaoTermCreateHessianMatrices()` should make a separate matrix for constructing the preconditioner
. -tao_term_hessian_mat_type <type>       - `MatType` for Hessian matrix created by `TaoTermCreateHessianMatrices()`
- -tao_term_hessian_pre_mat_type <type>   - `MatType` for matrix from which a preconditioner can be created by `TaoTermCreateHessianMatrices()`

  Level: developer

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermComputeHessian()`,
          `TaoTermCreateHessianMatrices()`,
          `TaoTermCreateHessianMatricesDefault()`,
          `TaoTermGetCreateHessianMode()`
@*/
PetscErrorCode TaoTermSetCreateHessianMode(TaoTerm term, PetscBool Hpre_is_H, MatType H_mattype, MatType Hpre_mattype)
{
  PetscBool is_hsame, is_hpresame;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  term->Hpre_is_H = Hpre_is_H;
  PetscCall(PetscStrcmp(term->H_mattype, H_mattype, &is_hsame));
  PetscCall(PetscStrcmp(term->Hpre_mattype, Hpre_mattype, &is_hpresame));
  if (!is_hsame) {
    PetscCall(PetscFree(term->H_mattype));
    if (H_mattype) PetscCall(PetscStrallocpy(H_mattype, (char **)&term->H_mattype));
  }
  if (!is_hpresame) {
    PetscCall(PetscFree(term->Hpre_mattype));
    if (Hpre_mattype) PetscCall(PetscStrallocpy(Hpre_mattype, (char **)&term->Hpre_mattype));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermGetCreateHessianMode - Get the behavior of `TaoTermCreateHessianMatricesDefault()`.

  Not collective

  Input Parameter:
. term - a `TaoTerm`

  Output Parameters:
+ Hpre_is_H    - (optional) should `TaoTermCreateHessianMatricesDefault()` make one matrix for `H` and `Hpre`?
. H_mattype    - (optional) the `MatType` to create for `H`
- Hpre_mattype - (optional) the `MatType` to create for `Hpre`

  Level: developer

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermComputeHessian()`,
          `TaoTermCreateHessianMatrices()`,
          `TaoTermCreateHessianMatricesDefault()`,
          `TaoTermSetCreateHessianMode()`
@*/
PetscErrorCode TaoTermGetCreateHessianMode(TaoTerm term, PetscBool *Hpre_is_H, MatType *H_mattype, MatType *Hpre_mattype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  if (Hpre_is_H) *Hpre_is_H = term->Hpre_is_H;
  if (H_mattype) *H_mattype = term->fd_hessian ? MATAIJ : term->H_mattype;
  if (Hpre_mattype) *Hpre_mattype = term->fd_hessian ? MATAIJ : term->Hpre_mattype;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermDuplicate - Duplicate a `TaoTerm`

  Collective

  Input Parameters:
+ term - a `TaoTerm`
- opt  - `TAOTERM_DUPLICATE_SIZEONLY` or `TAOTERM_DUPLICATE_TYPE`

  Output Parameter:
. newterm - the duplicate `TaoTerm`

  Notes:
  This function duplicates the solution space layout and vector type, but does not duplicate
  parameters-related configuration such as the parameters layout, `TaoTermParametersMode`,
  Hessian matrix types, or finite-difference settings. These must be set separately on the
  new `TaoTerm` if needed.

  If `TAOTERM_DUPLICATE_SIZEONLY` is used, then the duplicated term must have proper `TaoTermType`
  set with `TaoTermSetType()`.

  Level: intermediate

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermDuplicateOption`
@*/
PetscErrorCode TaoTermDuplicate(TaoTerm term, TaoTermDuplicateOption opt, TaoTerm *newterm)
{
  VecType     solution_vec_type;
  PetscLayout rlayout, clayout;
  PetscBool   is_shell;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscAssertPointer(newterm, 3);
  PetscCall(TaoTermCreate(PetscObjectComm((PetscObject)term), newterm));
  PetscCall(PetscObjectTypeCompare((PetscObject)term, TAOTERMSHELL, &is_shell));
  // Check if createsolutionvec is available first (for TaoTermShell)
  if (is_shell && term->ops->createsolutionvec) {
    Vec sol_template;

    PetscCall(TaoTermCreateSolutionVec(term, &sol_template));
    PetscCall(TaoTermSetSolutionTemplate(*newterm, sol_template));
    PetscCall(VecDestroy(&sol_template));
  } else {
    PetscCall(MatGetVecType(term->solution_factory, &solution_vec_type));
    PetscCall(MatGetLayouts(term->solution_factory, &rlayout, &clayout));
    PetscCall(MatSetVecType((*newterm)->solution_factory, solution_vec_type));
    PetscCall(MatSetLayouts((*newterm)->solution_factory, rlayout, clayout));
  }
  if (opt == TAOTERM_DUPLICATE_TYPE) {
    TaoTermType type;

    PetscCall(TaoTermGetType(term, &type));
    PetscCall(TaoTermSetType(*newterm, type));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermSetParametersMode - Sets the way a `TaoTerm` can accept parameters

  Logically collective

  Input Parameters:
+ term            - a `TaoTerm`
- parameters_mode - `TAOTERM_PARAMETERS_OPTIONAL`, `TAOTERM_PARAMETERS_NONE`, `TAOTERM_PARAMETERS_REQUIRED`

  Options Database Keys:
. -tao_term_parameters_mode <optional,none,required> - `TAOTERM_PARAMETERS_OPTIONAL`, `TAOTERM_PARAMETERS_NONE`, `TAOTERM_PARAMETERS_REQUIRED`

  Level: advanced

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermParametersMode`,
          `TaoTermGetParametersMode()`
@*/
PetscErrorCode TaoTermSetParametersMode(TaoTerm term, TaoTermParametersMode parameters_mode)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscValidLogicalCollectiveEnum(term, parameters_mode, 2);
  term->parameters_mode = parameters_mode;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermGetParametersMode - Gets the way a `TaoTerm` can accept parameters

  Not collective

  Input Parameter:
. term - a `TaoTerm`

  Output Parameter:
. parameters_mode - `TAOTERM_PARAMETERS_OPTIONAL`, `TAOTERM_PARAMETERS_NONE`, `TAOTERM_PARAMETERS_REQUIRED`

  Level: intermediate

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermParametersMode`,
          `TaoTermSetParametersMode()`
@*/
PetscErrorCode TaoTermGetParametersMode(TaoTerm term, TaoTermParametersMode *parameters_mode)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscAssertPointer(parameters_mode, 2);
  *parameters_mode = term->parameters_mode;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermGetFDDelta - Get the increment used for finite difference derivative approximations in methods like `TaoTermComputeGradientFD()`

  Not collective

  Input Parameter:
. term - a `TaoTerm`

  Output Parameter:
. delta - the finite difference increment

  Options Database Key:
. -tao_term_fd_delta <delta> - the above increment

  Level: advanced

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermSetFDDelta()`,
          `TaoTermComputeGradientFD()`,
          `TaoTermComputeGradientSetUseFD()`,
          `TaoTermComputeGradientGetUseFD()`
@*/
PetscErrorCode TaoTermGetFDDelta(TaoTerm term, PetscReal *delta)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscAssertPointer(delta, 2);
  *delta = term->fd_delta;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermSetFDDelta - Set the increment used for finite difference derivative approximations in methods like `TaoTermComputeGradientFD()`

  Logically collective

  Input Parameters:
+ term  - a `TaoTerm`
- delta - the finite difference increment

  Options Database Key:
. -tao_term_fd_delta <delta> - the above increment

  Level: advanced

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermGetFDDelta()`,
          `TaoTermComputeGradientFD()`,
          `TaoTermComputeGradientSetUseFD()`,
          `TaoTermComputeGradientGetUseFD()`,
          `TaoTermComputeHessianFD()`,
          `TaoTermComputeHessianSetUseFD()`,
          `TaoTermComputeHessianGetUseFD()`
@*/
PetscErrorCode TaoTermSetFDDelta(TaoTerm term, PetscReal delta)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscValidLogicalCollectiveReal(term, delta, 2);
  PetscCheck(delta > 0.0, PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_OUTOFRANGE, "finite difference increment must be positive");
  term->fd_delta = delta;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermComputeGradientSetUseFD - Set whether to use finite differences instead of the user-provided or built-in gradient method in `TaoTermComputeGradient()`.

  Logically collective

  Input Parameters:
+ term   - a `TaoTerm`
- use_fd - `PETSC_TRUE` to use finite differences, `PETSC_FALSE` to use the user-provided or built-in gradient method

  Options Database Keys:
. -tao_term_gradient_use_fd <bool> - use finite differences for gradient computation

  Level: advanced

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermGetFDDelta()`,
          `TaoTermSetFDDelta()`,
          `TaoTermComputeGradientFD()`,
          `TaoTermComputeGradientGetUseFD()`,
          `TaoTermComputeHessianFD()`,
          `TaoTermComputeHessianSetUseFD()`,
          `TaoTermComputeHessianGetUseFD()`
@*/
PetscErrorCode TaoTermComputeGradientSetUseFD(TaoTerm term, PetscBool use_fd)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscValidLogicalCollectiveBool(term, use_fd, 2);
  term->fd_gradient = use_fd;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermComputeGradientGetUseFD - Get whether finite differences are used in `TaoTermComputeGradient()`.

  Not collective

  Input Parameter:
. term - a `TaoTerm`

  Output Parameter:
. use_fd - `PETSC_TRUE` if finite differences are used

  Level: advanced

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermGetFDDelta()`,
          `TaoTermSetFDDelta()`,
          `TaoTermComputeGradientFD()`,
          `TaoTermComputeGradientSetUseFD()`,
          `TaoTermComputeHessianFD()`,
          `TaoTermComputeHessianSetUseFD()`,
          `TaoTermComputeHessianGetUseFD()`
@*/
PetscErrorCode TaoTermComputeGradientGetUseFD(TaoTerm term, PetscBool *use_fd)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscAssertPointer(use_fd, 2);
  *use_fd = term->fd_gradient;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermComputeHessianSetUseFD - Set whether to use finite differences instead of the user-provided or built-in methods in `TaoTermComputeHessian()`.

  Logically collective

  Input Parameters:
+ term   - a `TaoTerm`
- use_fd - `PETSC_TRUE` to use finite differences, `PETSC_FALSE` to use the user-provided or built-in Hessian method

  Options Database Keys:
. -tao_term_hessian_use_fd <bool> - use finite differences for Hessian computation

  Level: advanced

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermGetFDDelta()`,
          `TaoTermSetFDDelta()`,
          `TaoTermComputeGradientFD()`,
          `TaoTermComputeGradientSetUseFD()`,
          `TaoTermComputeGradientGetUseFD()`,
          `TaoTermComputeHessianFD()`,
          `TaoTermComputeHessianGetUseFD()`
@*/
PetscErrorCode TaoTermComputeHessianSetUseFD(TaoTerm term, PetscBool use_fd)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscValidLogicalCollectiveBool(term, use_fd, 2);
  term->fd_hessian = use_fd;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermComputeHessianGetUseFD - Get whether finite differences are used in `TaoTermComputeHessian()`.

  Not collective

  Input Parameter:
. term - a `TaoTerm`

  Output Parameter:
. use_fd - `PETSC_TRUE` if finite differences are used

  Level: advanced

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermGetFDDelta()`,
          `TaoTermSetFDDelta()`,
          `TaoTermComputeGradientFD()`,
          `TaoTermComputeGradientSetUseFD()`,
          `TaoTermComputeGradientGetUseFD()`,
          `TaoTermComputeHessianFD()`,
          `TaoTermComputeHessianSetUseFD()`
@*/
PetscErrorCode TaoTermComputeHessianGetUseFD(TaoTerm term, PetscBool *use_fd)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscAssertPointer(use_fd, 2);
  *use_fd = term->fd_hessian;
  PetscFunctionReturn(PETSC_SUCCESS);
}
