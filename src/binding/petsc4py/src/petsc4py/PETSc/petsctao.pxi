cdef extern from * nogil:

    ctypedef const char* PetscTAOType "TaoType"
    PetscTAOType TAOLMVM
    PetscTAOType TAONLS
    PetscTAOType TAONTR
    PetscTAOType TAONTL
    PetscTAOType TAOCG
    PetscTAOType TAOTRON
    PetscTAOType TAOOWLQN
    PetscTAOType TAOBMRM
    PetscTAOType TAOBLMVM
    PetscTAOType TAOBQNLS
    PetscTAOType TAOBNCG
    PetscTAOType TAOBNLS
    PetscTAOType TAOBNTR
    PetscTAOType TAOBNTL
    PetscTAOType TAOBQNKLS
    PetscTAOType TAOBQNKTR
    PetscTAOType TAOBQNKTL
    PetscTAOType TAOBQPIP
    PetscTAOType TAOGPCG
    PetscTAOType TAONM
    PetscTAOType TAOPOUNDERS
    PetscTAOType TAOBRGN
    PetscTAOType TAOLCL
    PetscTAOType TAOSSILS
    PetscTAOType TAOSSFLS
    PetscTAOType TAOASILS
    PetscTAOType TAOASFLS
    PetscTAOType TAOIPM
    PetscTAOType TAOPDIPM
    PetscTAOType TAOSHELL
    PetscTAOType TAOADMM
    PetscTAOType TAOALMM
    PetscTAOType TAOPYTHON

    ctypedef enum PetscTAOConvergedReason "TaoConvergedReason":
        #iterating
        TAO_CONTINUE_ITERATING
        # converged
        TAO_CONVERGED_GATOL
        TAO_CONVERGED_GRTOL
        TAO_CONVERGED_GTTOL
        TAO_CONVERGED_STEPTOL
        TAO_CONVERGED_MINF
        TAO_CONVERGED_USER
        # diverged
        TAO_DIVERGED_MAXITS
        TAO_DIVERGED_NAN
        TAO_DIVERGED_MAXFCN
        TAO_DIVERGED_LS_FAILURE
        TAO_DIVERGED_TR_REDUCTION
        TAO_DIVERGED_USER

    ctypedef enum PetscTAOLineSearchConvergedReason "TaoLineSearchConvergedReason":
      TAOLINESEARCH_CONTINUE_ITERATING
      TAOLINESEARCH_SUCCESS

    ctypedef PetscErrorCode (*PetscTaoMonitorDestroy)(void**)
    ctypedef PetscErrorCode PetscTaoConvergenceTest(PetscTAO,void*) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode PetscTaoMonitor(PetscTAO,void*) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode PetscTaoObjective(PetscTAO,PetscVec,PetscReal*,void*) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode PetscTaoResidual(PetscTAO,PetscVec,PetscVec,void*) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode PetscTaoGradient(PetscTAO,PetscVec,PetscVec,void*) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode PetscTaoObjGrad(PetscTAO,PetscVec,PetscReal*,PetscVec,void*) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode PetscTaoRegularizerObjGrad(PetscTAO,PetscVec,PetscReal*,PetscVec,void*) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode PetscTaoVarBounds(PetscTAO,PetscVec,PetscVec,void*) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode PetscTaoConstraints(PetscTAO,PetscVec,PetscVec,void*) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode PetscTaoEqualityConstraints(PetscTAO,PetscVec,PetscVec,void*) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode PetscTaoHessian(PetscTAO,PetscVec,PetscMat,PetscMat,void*) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode PetscTaoRegularizerHessian(PetscTAO,PetscVec,PetscMat,void*) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode PetscTaoJacobian(PetscTAO,PetscVec,PetscMat,PetscMat,void*) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode PetscTaoJacobianResidual(PetscTAO,PetscVec,PetscMat,PetscMat,void*) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode PetscTaoJacobianState(PetscTAO,PetscVec,PetscMat,PetscMat,PetscMat,void*) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode PetscTaoJacobianDesign(PetscTAO,PetscVec,PetscMat,void*) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode PetscTaoJacobianEquality(PetscTAO,PetscVec,PetscMat,PetscMat,void*) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode PetscTaoUpdateFunction(PetscTAO,PetscInt,void*) except PETSC_ERR_PYTHON


    PetscErrorCode TaoMonitor(PetscTAO,PetscInt,PetscReal,PetscReal,PetscReal,PetscReal)
    PetscErrorCode TaoView(PetscTAO,PetscViewer)
    PetscErrorCode TaoDestroy(PetscTAO*)
    PetscErrorCode TaoCreate(MPI_Comm,PetscTAO*)
    PetscErrorCode TaoSetOptionsPrefix(PetscTAO,char[])
    PetscErrorCode TaoAppendOptionsPrefix(PetscTAO,char[])
    PetscErrorCode TaoGetOptionsPrefix(PetscTAO,char*[])
    PetscErrorCode TaoSetFromOptions(PetscTAO)
    PetscErrorCode TaoSetType(PetscTAO,PetscTAOType)
    PetscErrorCode TaoGetType(PetscTAO,PetscTAOType*)

    PetscErrorCode TaoSetUp(PetscTAO)
    PetscErrorCode TaoSolve(PetscTAO)

    PetscErrorCode TaoSetTolerances(PetscTAO,PetscReal,PetscReal,PetscReal)
    PetscErrorCode TaoGetTolerances(PetscTAO,PetscReal*,PetscReal*,PetscReal*)
    PetscErrorCode TaoSetConstraintTolerances(PetscTAO,PetscReal,PetscReal)
    PetscErrorCode TaoGetConstraintTolerances(PetscTAO,PetscReal*,PetscReal*)

    PetscErrorCode TaoSetFunctionLowerBound(PetscTAO,PetscReal)
    PetscErrorCode TaoSetMaximumIterations(PetscTAO,PetscInt)
    PetscErrorCode TaoGetMaximumIterations(PetscTAO,PetscInt*)
    PetscErrorCode TaoSetMaximumFunctionEvaluations(PetscTAO,PetscInt)
    PetscErrorCode TaoGetMaximumFunctionEvaluations(PetscTAO,PetscInt*)
    PetscErrorCode TaoSetIterationNumber(PetscTAO,PetscInt)
    PetscErrorCode TaoGetIterationNumber(PetscTAO,PetscInt*)

    PetscErrorCode TaoSetTrustRegionTolerance(PetscTAO,PetscReal)
    PetscErrorCode TaoGetInitialTrustRegionRadius(PetscTAO,PetscReal*)
    PetscErrorCode TaoGetTrustRegionRadius(PetscTAO,PetscReal*)
    PetscErrorCode TaoSetTrustRegionRadius(PetscTAO,PetscReal)

    PetscErrorCode TaoDefaultConvergenceTest(PetscTAO,void*) except PETSC_ERR_PYTHON
    PetscErrorCode TaoSetConvergenceTest(PetscTAO,PetscTaoConvergenceTest*, void*)
    PetscErrorCode TaoSetConvergedReason(PetscTAO,PetscTAOConvergedReason)
    PetscErrorCode TaoGetConvergedReason(PetscTAO,PetscTAOConvergedReason*)
    PetscErrorCode TaoLogConvergenceHistory(PetscTAO,PetscReal,PetscReal,PetscReal,PetscInt)
    PetscErrorCode TaoGetSolutionStatus(PetscTAO,PetscInt*,
                                        PetscReal*,PetscReal*,
                                        PetscReal*,PetscReal*,
                                        PetscTAOConvergedReason*)

    PetscErrorCode TaoSetMonitor(PetscTAO,PetscTaoMonitor,void*,PetscTaoMonitorDestroy)
    PetscErrorCode TaoCancelMonitors(PetscTAO)

    PetscErrorCode TaoComputeObjective(PetscTAO,PetscVec,PetscReal*)
    PetscErrorCode TaoComputeResidual(PetscTAO,PetscVec,PetscVec)
    PetscErrorCode TaoComputeGradient(PetscTAO,PetscVec,PetscVec)
    PetscErrorCode TaoComputeObjectiveAndGradient(PetscTAO,PetscVec,PetscReal*,PetscVec)
    PetscErrorCode TaoComputeConstraints(PetscTAO,PetscVec,PetscVec)
    PetscErrorCode TaoComputeDualVariables(PetscTAO,PetscVec,PetscVec)
    PetscErrorCode TaoComputeVariableBounds(PetscTAO)
    PetscErrorCode TaoComputeHessian(PetscTAO,PetscVec,PetscMat,PetscMat)
    PetscErrorCode TaoComputeJacobian(PetscTAO,PetscVec,PetscMat,PetscMat)

    PetscErrorCode TaoSetSolution(PetscTAO,PetscVec)
    PetscErrorCode TaoSetConstraintsVec(PetscTAO,PetscVec)
    PetscErrorCode TaoSetVariableBounds(PetscTAO,PetscVec,PetscVec)

    PetscErrorCode TaoGetSolution(PetscTAO,PetscVec*)
    PetscErrorCode TaoSetGradientNorm(PetscTAO,PetscMat)
    PetscErrorCode TaoGetGradientNorm(PetscTAO,PetscMat*)
    PetscErrorCode TaoLMVMSetH0(PetscTAO,PetscMat)
    PetscErrorCode TaoLMVMGetH0(PetscTAO,PetscMat*)
    PetscErrorCode TaoLMVMGetH0KSP(PetscTAO,PetscKSP*)
    PetscErrorCode TaoGetVariableBounds(PetscTAO,PetscVec*,PetscVec*)

    PetscErrorCode TaoSetObjective(PetscTAO,PetscTaoObjective*,void*)
    PetscErrorCode TaoSetGradient(PetscTAO,PetscVec,PetscTaoGradient*,void*)
    PetscErrorCode TaoSetObjectiveAndGradient(PetscTAO,PetscVec,PetscTaoObjGrad*,void*)
    PetscErrorCode TaoSetHessian(PetscTAO,PetscMat,PetscMat,PetscTaoHessian*,void*)
    PetscErrorCode TaoGetObjective(PetscTAO,PetscTaoObjective**,void**)
    PetscErrorCode TaoGetGradient(PetscTAO,PetscVec*,PetscTaoGradient**,void**)
    PetscErrorCode TaoGetObjectiveAndGradient(PetscTAO,PetscVec*,PetscTaoObjGrad**,void**)
    PetscErrorCode TaoGetHessian(PetscTAO,PetscMat*,PetscMat*,PetscTaoHessian**,void**)
    PetscErrorCode TaoSetResidualRoutine(PetscTAO,PetscVec,PetscTaoResidual,void*)
    PetscErrorCode TaoSetVariableBoundsRoutine(PetscTAO,PetscTaoVarBounds*,void*)
    PetscErrorCode TaoSetConstraintsRoutine(PetscTAO,PetscVec,PetscTaoConstraints*,void*)
    PetscErrorCode TaoSetJacobianRoutine(PetscTAO,PetscMat,PetscMat,PetscTaoJacobian*,void*)
    PetscErrorCode TaoSetJacobianResidualRoutine(PetscTAO,PetscMat,PetscMat,PetscTaoJacobianResidual*,void*)

    PetscErrorCode TaoSetStateDesignIS(PetscTAO,PetscIS,PetscIS)
    PetscErrorCode TaoSetJacobianStateRoutine(PetscTAO,PetscMat,PetscMat,PetscMat,PetscTaoJacobianState*,void*)
    PetscErrorCode TaoSetJacobianDesignRoutine(PetscTAO,PetscMat,PetscTaoJacobianDesign*,void*)

    PetscErrorCode TaoSetEqualityConstraintsRoutine(PetscTAO,PetscVec,PetscTaoEqualityConstraints*,void*)
    PetscErrorCode TaoSetJacobianEqualityRoutine(PetscTAO,PetscMat,PetscMat,PetscTaoJacobianEquality*,void*)
    PetscErrorCode TaoSetUpdate(PetscTAO,PetscTaoUpdateFunction*,void*)

    PetscErrorCode TaoSetInitialTrustRegionRadius(PetscTAO,PetscReal)

    PetscErrorCode TaoGetKSP(PetscTAO,PetscKSP*)

    PetscErrorCode TaoBRGNGetSubsolver(PetscTAO,PetscTAO*)
    PetscErrorCode TaoBRGNSetRegularizerObjectiveAndGradientRoutine(PetscTAO,PetscTaoRegularizerObjGrad*,void*)
    PetscErrorCode TaoBRGNSetRegularizerHessianRoutine(PetscTAO,PetscMat,PetscTaoRegularizerHessian*,void*)
    PetscErrorCode TaoBRGNSetRegularizerWeight(PetscTAO,PetscReal)
    PetscErrorCode TaoBRGNSetL1SmoothEpsilon(PetscTAO,PetscReal)
    PetscErrorCode TaoBRGNSetDictionaryMatrix(PetscTAO,PetscMat)
    PetscErrorCode TaoBRGNGetDampingVector(PetscTAO,PetscVec*)

    PetscErrorCode TaoPythonSetType(PetscTAO,char[])
    PetscErrorCode TaoPythonGetType(PetscTAO,char*[])

# --------------------------------------------------------------------

cdef inline TAO ref_TAO(PetscTAO tao):
    cdef TAO ob = <TAO> TAO()
    ob.tao = tao
    PetscINCREF(ob.obj)
    return ob

# --------------------------------------------------------------------

cdef PetscErrorCode TAO_Objective(PetscTAO _tao,
                       PetscVec _x, PetscReal *_f,
                       void *ctx) except PETSC_ERR_PYTHON with gil:

    cdef TAO tao = ref_TAO(_tao)
    cdef Vec x   = ref_Vec(_x)
    context = tao.get_attr("__objective__")
    if context is None and ctx != NULL: context = <object>ctx
    assert context is not None and type(context) is tuple # sanity check
    (objective, args, kargs) = context
    retv = objective(tao, x, *args, **kargs)
    _f[0] = asReal(retv)
    return PETSC_SUCCESS

cdef PetscErrorCode TAO_Residual(PetscTAO _tao,
                      PetscVec _x, PetscVec _r,
                      void *ctx) except PETSC_ERR_PYTHON with gil:

    cdef TAO tao = ref_TAO(_tao)
    cdef Vec x   = ref_Vec(_x)
    cdef Vec r   = ref_Vec(_r)
    context = tao.get_attr("__residual__")
    if context is None and ctx != NULL: context = <object>ctx
    assert context is not None and type(context) is tuple # sanity check
    (residual, args, kargs) = context
    residual(tao, x, r, *args, **kargs)
    return PETSC_SUCCESS

cdef PetscErrorCode TAO_Gradient(PetscTAO _tao,
                      PetscVec _x, PetscVec _g,
                      void *ctx) except PETSC_ERR_PYTHON with gil:

    cdef TAO tao = ref_TAO(_tao)
    cdef Vec x   = ref_Vec(_x)
    cdef Vec g   = ref_Vec(_g)
    context = tao.get_attr("__gradient__")
    if context is None and ctx != NULL: context = <object>ctx
    assert context is not None and type(context) is tuple # sanity check
    (gradient, args, kargs) = context
    gradient(tao, x, g, *args, **kargs)
    return PETSC_SUCCESS


cdef PetscErrorCode TAO_ObjGrad(PetscTAO _tao,
                     PetscVec _x, PetscReal *_f, PetscVec _g,
                     void *ctx) except PETSC_ERR_PYTHON with gil:

    cdef TAO tao = ref_TAO(_tao)
    cdef Vec x   = ref_Vec(_x)
    cdef Vec g   = ref_Vec(_g)
    context = tao.get_attr("__objgrad__")
    if context is None and ctx != NULL: context = <object>ctx
    assert context is not None and type(context) is tuple # sanity check
    (objgrad, args, kargs) = context
    retv = objgrad(tao, x, g, *args, **kargs)
    _f[0] = asReal(retv)
    return PETSC_SUCCESS

cdef PetscErrorCode TAO_BRGNRegObjGrad(PetscTAO _tao,
                     PetscVec _x, PetscReal *_f, PetscVec _g,
                     void *ctx) except PETSC_ERR_PYTHON with gil:

    cdef TAO tao = ref_TAO(_tao)
    cdef Vec x   = ref_Vec(_x)
    cdef Vec g   = ref_Vec(_g)
    context = tao.get_attr("__brgnregobjgrad__")
    if context is None and ctx != NULL: context = <object>ctx
    assert context is not None and type(context) is tuple # sanity check
    (objgrad, args, kargs) = context
    retv = objgrad(tao, x, g, *args, **kargs)
    _f[0] = asReal(retv)
    return PETSC_SUCCESS

cdef PetscErrorCode TAO_Constraints(PetscTAO _tao,
                         PetscVec _x, PetscVec _r,
                         void *ctx) except PETSC_ERR_PYTHON with gil:

    cdef TAO tao = ref_TAO(_tao)
    cdef Vec x   = ref_Vec(_x)
    cdef Vec r   = ref_Vec(_r)
    context = tao.get_attr("__constraints__")
    if context is None and ctx != NULL: context = <object>ctx
    assert context is not None and type(context) is tuple # sanity check
    (constraints, args, kargs) = context
    constraints(tao, x, r, *args, **kargs)
    return PETSC_SUCCESS

cdef PetscErrorCode TAO_VarBounds(PetscTAO _tao,
                       PetscVec _xl, PetscVec _xu,
                       void *ctx) except PETSC_ERR_PYTHON with gil:

    cdef TAO tao = ref_TAO(_tao)
    cdef Vec xl  = ref_Vec(_xl)
    cdef Vec xu  = ref_Vec(_xu)
    context = tao.get_attr("__varbounds__")
    if context is None and ctx != NULL: context = <object>ctx
    assert context is not None and type(context) is tuple # sanity check
    (varbounds, args, kargs) = context
    varbounds(tao, xl, xu, *args, **kargs)
    return PETSC_SUCCESS

cdef PetscErrorCode TAO_Hessian(PetscTAO _tao,
                     PetscVec _x,
                     PetscMat _H,
                     PetscMat _P,
                     void* ctx) except PETSC_ERR_PYTHON with gil:
    cdef TAO tao = ref_TAO(_tao)
    cdef Vec x   = ref_Vec(_x)
    cdef Mat H   = ref_Mat(_H)
    cdef Mat P   = ref_Mat(_P)
    context = tao.get_attr("__hessian__")
    if context is None and ctx != NULL: context = <object>ctx
    assert context is not None and type(context) is tuple # sanity check
    (hessian, args, kargs) = context
    hessian(tao, x, H, P, *args, **kargs)
    return PETSC_SUCCESS

cdef PetscErrorCode TAO_BRGNRegHessian(PetscTAO _tao,
                     PetscVec  _x,
                     PetscMat  _H,
                     void* ctx) except PETSC_ERR_PYTHON with gil:
    cdef TAO tao = ref_TAO(_tao)
    cdef Vec x   = ref_Vec(_x)
    cdef Mat H   = ref_Mat(_H)
    context = tao.get_attr("__brgnreghessian__")
    if context is None and ctx != NULL: context = <object>ctx
    assert context is not None and type(context) is tuple # sanity check
    (hessian, args, kargs) = context
    hessian(tao, x, H, *args, **kargs)
    return PETSC_SUCCESS

cdef PetscErrorCode TAO_Jacobian(PetscTAO _tao,
                      PetscVec  _x,
                      PetscMat  _J,
                      PetscMat  _P,
                      void* ctx) except PETSC_ERR_PYTHON with gil:
    cdef TAO tao = ref_TAO(_tao)
    cdef Vec x   = ref_Vec(_x)
    cdef Mat J   = ref_Mat(_J)
    cdef Mat P   = ref_Mat(_P)
    context = tao.get_attr("__jacobian__")
    if context is None and ctx != NULL: context = <object>ctx
    assert context is not None and type(context) is tuple # sanity check
    (jacobian, args, kargs) = context
    jacobian(tao, x, J, P, *args, **kargs)
    return PETSC_SUCCESS

cdef PetscErrorCode TAO_JacobianResidual(PetscTAO _tao,
                      PetscVec  _x,
                      PetscMat  _J,
                      PetscMat  _P,
                      void* ctx) except PETSC_ERR_PYTHON with gil:
    cdef TAO tao = ref_TAO(_tao)
    cdef Vec x   = ref_Vec(_x)
    cdef Mat J   = ref_Mat(_J)
    cdef Mat P   = ref_Mat(_P)
    context = tao.get_attr("__jacobian_residual__")
    if context is None and ctx != NULL: context = <object>ctx
    assert context is not None and type(context) is tuple # sanity check
    (jacobian, args, kargs) = context
    jacobian(tao, x, J, P, *args, **kargs)
    return PETSC_SUCCESS

cdef PetscErrorCode TAO_JacobianState(PetscTAO _tao,
                           PetscVec  _x,
                           PetscMat  _J,
                           PetscMat  _P,
                           PetscMat  _I,
                           void* ctx) except PETSC_ERR_PYTHON with gil:
    cdef TAO tao = ref_TAO(_tao)
    cdef Vec x   = ref_Vec(_x)
    cdef Mat J   = ref_Mat(_J)
    cdef Mat P   = ref_Mat(_P)
    cdef Mat I   = ref_Mat(_I)
    context = tao.get_attr("__jacobian_state__")
    if context is None and ctx != NULL: context = <object>ctx
    assert context is not None and type(context) is tuple # sanity check
    (jacobian, args, kargs) = context
    jacobian(tao, x, J, P, I, *args, **kargs)
    return PETSC_SUCCESS

cdef PetscErrorCode TAO_JacobianDesign(PetscTAO _tao,
                            PetscVec  _x,
                            PetscMat  _J,
                            void* ctx) except PETSC_ERR_PYTHON with gil:
    cdef TAO tao = ref_TAO(_tao)
    cdef Vec x   = ref_Vec(_x)
    cdef Mat J   = ref_Mat(_J)
    context = tao.get_attr("__jacobian_design__")
    if context is None and ctx != NULL: context = <object>ctx
    assert context is not None and type(context) is tuple # sanity check
    (jacobian, args, kargs) = context
    jacobian(tao, x, J, *args, **kargs)
    return PETSC_SUCCESS

cdef PetscErrorCode TAO_EqualityConstraints(PetscTAO _tao,
                                 PetscVec  _x,
                                 PetscVec  _c,
                                 void* ctx) except PETSC_ERR_PYTHON with gil:
    cdef TAO tao = ref_TAO(_tao)
    cdef Vec x   = ref_Vec(_x)
    cdef Vec c   = ref_Vec(_c)
    context = tao.get_attr("__equality_constraints__")
    if context is None and ctx != NULL: context = <object>ctx
    assert context is not None and type(context) is tuple # sanity check
    (f, args, kargs) = context
    f(tao, x, c, *args, **kargs)
    return PETSC_SUCCESS

cdef PetscErrorCode TAO_JacobianEquality(PetscTAO _tao,
                              PetscVec  _x,
                              PetscMat  _J,
                              PetscMat  _P,
                              void* ctx) except PETSC_ERR_PYTHON with gil:
    cdef TAO tao = ref_TAO(_tao)
    cdef Vec x   = ref_Vec(_x)
    cdef Mat J   = ref_Mat(_J)
    cdef Mat P   = ref_Mat(_P)
    context = tao.get_attr("__jacobian_equality__")
    if context is None and ctx != NULL: context = <object>ctx
    assert context is not None and type(context) is tuple # sanity check
    (jacobian, args, kargs) = context
    jacobian(tao, x, J, P, *args, **kargs)
    return PETSC_SUCCESS

# ctx is unused
cdef PetscErrorCode TAO_Update(
    PetscTAO _tao,
    PetscInt its,
    void* ctx) except PETSC_ERR_PYTHON with gil:
    cdef TAO tao = ref_TAO(_tao)
    cdef object context = tao.get_attr('__update__')
    assert context is not None and type(context) is tuple # sanity check
    (update, args, kargs) = context
    update(tao, toInt(its), *args, **kargs)
    return PETSC_SUCCESS


cdef PetscErrorCode TAO_Converged(PetscTAO _tao,
                       void* ctx) except PETSC_ERR_PYTHON with gil:
    # call first the default convergence test
    CHKERR( TaoDefaultConvergenceTest(_tao, NULL) )
    # call next the user-provided convergence test
    cdef TAO tao = ref_TAO(_tao)
    (converged, args, kargs) = tao.get_attr('__converged__')
    reason = converged(tao, *args, **kargs)
    if reason is None:  return PETSC_SUCCESS
    # handle value of convergence reason
    cdef PetscTAOConvergedReason creason = TAO_CONTINUE_ITERATING
    if reason is False or reason == -1:
        creason = TAO_DIVERGED_USER
    elif reason is True or reason == 1:
        creason = TAO_CONVERGED_USER
    else:
        creason = reason
        assert creason >= TAO_DIVERGED_USER
        assert creason <= TAO_CONVERGED_USER
    CHKERR( TaoSetConvergedReason(_tao, creason) )
    return PETSC_SUCCESS

cdef PetscErrorCode TAO_Monitor(PetscTAO _tao,
                     void* ctx) except PETSC_ERR_PYTHON with gil:
    cdef TAO tao = ref_TAO(_tao)
    cdef object monitorlist = tao.get_attr('__monitor__')
    if monitorlist is None: return PETSC_SUCCESS
    for (monitor, args, kargs) in monitorlist:
        monitor(tao, *args, **kargs)
    return PETSC_SUCCESS

# --------------------------------------------------------------------
