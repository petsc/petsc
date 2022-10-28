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

    ctypedef int (*PetscTaoMonitorDestroy)(void**)
    ctypedef int PetscTaoConvergenceTest(PetscTAO,void*) except PETSC_ERR_PYTHON
    ctypedef int PetscTaoMonitor(PetscTAO,void*) except PETSC_ERR_PYTHON
    ctypedef int PetscTaoObjective(PetscTAO,PetscVec,PetscReal*,void*) except PETSC_ERR_PYTHON
    ctypedef int PetscTaoResidual(PetscTAO,PetscVec,PetscVec,void*) except PETSC_ERR_PYTHON
    ctypedef int PetscTaoGradient(PetscTAO,PetscVec,PetscVec,void*) except PETSC_ERR_PYTHON
    ctypedef int PetscTaoObjGrad(PetscTAO,PetscVec,PetscReal*,PetscVec,void*) except PETSC_ERR_PYTHON
    ctypedef int PetscTaoRegularizerObjGrad(PetscTAO,PetscVec,PetscReal*,PetscVec,void*) except PETSC_ERR_PYTHON
    ctypedef int PetscTaoVarBounds(PetscTAO,PetscVec,PetscVec,void*) except PETSC_ERR_PYTHON
    ctypedef int PetscTaoConstraints(PetscTAO,PetscVec,PetscVec,void*) except PETSC_ERR_PYTHON
    ctypedef int PetscTaoEqualityConstraints(PetscTAO,PetscVec,PetscVec,void*) except PETSC_ERR_PYTHON
    ctypedef int PetscTaoHessian(PetscTAO,PetscVec,PetscMat,PetscMat,void*) except PETSC_ERR_PYTHON
    ctypedef int PetscTaoRegularizerHessian(PetscTAO,PetscVec,PetscMat,void*) except PETSC_ERR_PYTHON
    ctypedef int PetscTaoJacobian(PetscTAO,PetscVec,PetscMat,PetscMat,void*) except PETSC_ERR_PYTHON
    ctypedef int PetscTaoJacobianResidual(PetscTAO,PetscVec,PetscMat,PetscMat,void*) except PETSC_ERR_PYTHON
    ctypedef int PetscTaoJacobianState(PetscTAO,PetscVec,PetscMat,PetscMat,PetscMat,void*) except PETSC_ERR_PYTHON
    ctypedef int PetscTaoJacobianDesign(PetscTAO,PetscVec,PetscMat,void*) except PETSC_ERR_PYTHON
    ctypedef int PetscTaoJacobianEquality(PetscTAO,PetscVec,PetscMat,PetscMat,void*) except PETSC_ERR_PYTHON
    ctypedef int PetscTaoUpdateFunction(PetscTAO,PetscInt,void*) except PETSC_ERR_PYTHON


    int TaoMonitor(PetscTAO,PetscInt,PetscReal,PetscReal,PetscReal,PetscReal)
    int TaoView(PetscTAO,PetscViewer)
    int TaoDestroy(PetscTAO*)
    int TaoCreate(MPI_Comm,PetscTAO*)
    int TaoSetOptionsPrefix(PetscTAO,char[])
    int TaoAppendOptionsPrefix(PetscTAO,char[])
    int TaoGetOptionsPrefix(PetscTAO,char*[])
    int TaoSetFromOptions(PetscTAO)
    int TaoSetType(PetscTAO,PetscTAOType)
    int TaoGetType(PetscTAO,PetscTAOType*)

    int TaoSetUp(PetscTAO)
    int TaoSolve(PetscTAO)

    int TaoSetTolerances(PetscTAO,PetscReal,PetscReal,PetscReal)
    int TaoGetTolerances(PetscTAO,PetscReal*,PetscReal*,PetscReal*)
    int TaoSetConstraintTolerances(PetscTAO,PetscReal,PetscReal)
    int TaoGetConstraintTolerances(PetscTAO,PetscReal*,PetscReal*)

    int TaoSetFunctionLowerBound(PetscTAO,PetscReal)
    int TaoSetMaximumIterations(PetscTAO,PetscInt)
    int TaoGetMaximumIterations(PetscTAO,PetscInt*)
    int TaoSetMaximumFunctionEvaluations(PetscTAO,PetscInt)
    int TaoGetMaximumFunctionEvaluations(PetscTAO,PetscInt*)
    int TaoSetIterationNumber(PetscTAO,PetscInt)
    int TaoGetIterationNumber(PetscTAO,PetscInt*)

    int TaoSetTrustRegionTolerance(PetscTAO,PetscReal)
    int TaoGetInitialTrustRegionRadius(PetscTAO,PetscReal*)
    int TaoGetTrustRegionRadius(PetscTAO,PetscReal*)
    int TaoSetTrustRegionRadius(PetscTAO,PetscReal)

    int TaoDefaultConvergenceTest(PetscTAO,void*) except PETSC_ERR_PYTHON
    int TaoSetConvergenceTest(PetscTAO,PetscTaoConvergenceTest*, void*)
    int TaoSetConvergedReason(PetscTAO,PetscTAOConvergedReason)
    int TaoGetConvergedReason(PetscTAO,PetscTAOConvergedReason*)
    int TaoGetSolutionStatus(PetscTAO,PetscInt*,
                             PetscReal*,PetscReal*,
                             PetscReal*,PetscReal*,
                             PetscTAOConvergedReason*)

    int TaoSetMonitor(PetscTAO,PetscTaoMonitor,void*,PetscTaoMonitorDestroy)
    int TaoCancelMonitors(PetscTAO)

    int TaoComputeObjective(PetscTAO,PetscVec,PetscReal*)
    int TaoComputeResidual(PetscTAO,PetscVec,PetscVec)
    int TaoComputeGradient(PetscTAO,PetscVec,PetscVec)
    int TaoComputeObjectiveAndGradient(PetscTAO,PetscVec,PetscReal*,PetscVec)
    int TaoComputeConstraints(PetscTAO,PetscVec,PetscVec)
    int TaoComputeDualVariables(PetscTAO,PetscVec,PetscVec)
    int TaoComputeVariableBounds(PetscTAO)
    int TaoComputeHessian(PetscTAO,PetscVec,PetscMat,PetscMat)
    int TaoComputeJacobian(PetscTAO,PetscVec,PetscMat,PetscMat)

    int TaoSetSolution(PetscTAO,PetscVec)
    int TaoSetConstraintsVec(PetscTAO,PetscVec)
    int TaoSetVariableBounds(PetscTAO,PetscVec,PetscVec)

    int TaoGetSolution(PetscTAO,PetscVec*)
    int TaoSetGradientNorm(PetscTAO,PetscMat)
    int TaoGetGradientNorm(PetscTAO,PetscMat*)
    int TaoLMVMSetH0(PetscTAO,PetscMat)
    int TaoLMVMGetH0(PetscTAO,PetscMat*)
    int TaoLMVMGetH0KSP(PetscTAO,PetscKSP*)
    int TaoGetVariableBounds(PetscTAO,PetscVec*,PetscVec*)

    int TaoSetObjective(PetscTAO,PetscTaoObjective*,void*)
    int TaoSetGradient(PetscTAO,PetscVec,PetscTaoGradient*,void*)
    int TaoSetObjectiveAndGradient(PetscTAO,PetscVec,PetscTaoObjGrad*,void*)
    int TaoSetHessian(PetscTAO,PetscMat,PetscMat,PetscTaoHessian*,void*)
    int TaoGetObjective(PetscTAO,PetscTaoObjective**,void**)
    int TaoGetGradient(PetscTAO,PetscVec*,PetscTaoGradient**,void**)
    int TaoGetObjectiveAndGradient(PetscTAO,PetscVec*,PetscTaoObjGrad**,void**)
    int TaoGetHessian(PetscTAO,PetscMat*,PetscMat*,PetscTaoHessian**,void**)
    int TaoSetResidualRoutine(PetscTAO,PetscVec,PetscTaoResidual,void*)
    int TaoSetVariableBoundsRoutine(PetscTAO,PetscTaoVarBounds*,void*)
    int TaoSetConstraintsRoutine(PetscTAO,PetscVec,PetscTaoConstraints*,void*)
    int TaoSetJacobianRoutine(PetscTAO,PetscMat,PetscMat,PetscTaoJacobian*,void*)
    int TaoSetJacobianResidualRoutine(PetscTAO,PetscMat,PetscMat,PetscTaoJacobianResidual*,void*)

    int TaoSetStateDesignIS(PetscTAO,PetscIS,PetscIS)
    int TaoSetJacobianStateRoutine(PetscTAO,PetscMat,PetscMat,PetscMat,PetscTaoJacobianState*,void*)
    int TaoSetJacobianDesignRoutine(PetscTAO,PetscMat,PetscTaoJacobianDesign*,void*)

    int TaoSetEqualityConstraintsRoutine(PetscTAO,PetscVec,PetscTaoEqualityConstraints*,void*)
    int TaoSetJacobianEqualityRoutine(PetscTAO,PetscMat,PetscMat,PetscTaoJacobianEquality*,void*)
    int TaoSetUpdate(PetscTAO,PetscTaoUpdateFunction*,void*)

    int TaoSetInitialTrustRegionRadius(PetscTAO,PetscReal)

    int TaoGetKSP(PetscTAO,PetscKSP*)

    int TaoBRGNGetSubsolver(PetscTAO,PetscTAO*)
    int TaoBRGNSetRegularizerObjectiveAndGradientRoutine(PetscTAO,PetscTaoRegularizerObjGrad*,void*)
    int TaoBRGNSetRegularizerHessianRoutine(PetscTAO,PetscMat,PetscTaoRegularizerHessian*,void*)
    int TaoBRGNSetRegularizerWeight(PetscTAO,PetscReal)
    int TaoBRGNSetL1SmoothEpsilon(PetscTAO,PetscReal)
    int TaoBRGNSetDictionaryMatrix(PetscTAO,PetscMat)
    int TaoBRGNGetDampingVector(PetscTAO,PetscVec*)

    int TaoPythonSetType(PetscTAO,char[])
    int TaoPythonGetType(PetscTAO,char*[])

cdef extern from "libpetsc4py.h":
    int TaoPythonSetContext(PetscTAO,void*)
    int TaoPythonGetContext(PetscTAO,void**)

# --------------------------------------------------------------------

cdef inline TAO ref_TAO(PetscTAO tao):
    cdef TAO ob = <TAO> TAO()
    ob.tao = tao
    PetscINCREF(ob.obj)
    return ob

# --------------------------------------------------------------------

cdef int TAO_Objective(PetscTAO _tao,
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
    return 0

cdef int TAO_Residual(PetscTAO _tao,
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
    return 0

cdef int TAO_Gradient(PetscTAO _tao,
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
    return 0


cdef int TAO_ObjGrad(PetscTAO _tao,
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
    return 0

cdef int TAO_BRGNRegObjGrad(PetscTAO _tao,
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
    return 0

cdef int TAO_Constraints(PetscTAO _tao,
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
    return 0

cdef int TAO_VarBounds(PetscTAO _tao,
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
    return 0

cdef int TAO_Hessian(PetscTAO _tao,
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
    return 0

cdef int TAO_BRGNRegHessian(PetscTAO _tao,
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
    return 0

cdef int TAO_Jacobian(PetscTAO _tao,
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
    return 0

cdef int TAO_JacobianResidual(PetscTAO _tao,
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
    return 0

cdef int TAO_JacobianState(PetscTAO _tao,
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
    return 0

cdef int TAO_JacobianDesign(PetscTAO _tao,
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
    return 0

cdef int TAO_EqualityConstraints(PetscTAO _tao,
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
    return 0

cdef int TAO_JacobianEquality(PetscTAO _tao,
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
    return 0

# ctx is unused
cdef int TAO_Update(
    PetscTAO _tao,
    PetscInt its,
    void* ctx) except PETSC_ERR_PYTHON with gil:
    cdef TAO tao = ref_TAO(_tao)
    cdef object context = tao.get_attr('__update__')
    assert context is not None and type(context) is tuple # sanity check
    (update, args, kargs) = context
    update(tao, toInt(its), *args, **kargs)
    return 0


cdef int TAO_Converged(PetscTAO _tao,
                       void* ctx) except PETSC_ERR_PYTHON with gil:
    # call first the default convergence test
    CHKERR( TaoDefaultConvergenceTest(_tao, NULL) )
    # call next the user-provided convergence test
    cdef TAO tao = ref_TAO(_tao)
    (converged, args, kargs) = tao.get_attr('__converged__')
    reason = converged(tao, *args, **kargs)
    if reason is None:  return 0
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
    return 0

cdef int TAO_Monitor(PetscTAO _tao,
                     void* ctx) except PETSC_ERR_PYTHON with gil:
    cdef TAO tao = ref_TAO(_tao)
    cdef object monitorlist = tao.get_attr('__monitor__')
    if monitorlist is None: return 0
    for (monitor, args, kargs) in monitorlist:
        monitor(tao, *args, **kargs)
    return 0

# --------------------------------------------------------------------
