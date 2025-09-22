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
        # iterating
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

    ctypedef PetscErrorCode (*PetscTaoMonitorDestroy)(void**)
    ctypedef PetscErrorCode PetscTaoConvergenceTest(PetscTAO, void*) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode PetscTaoMonitor(PetscTAO, void*) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode PetscTaoObjective(PetscTAO, PetscVec, PetscReal*, void*) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode PetscTaoResidual(PetscTAO, PetscVec, PetscVec, void*) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode PetscTaoGradient(PetscTAO, PetscVec, PetscVec, void*) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode PetscTaoObjGrad(PetscTAO, PetscVec, PetscReal*, PetscVec, void*) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode PetscTaoRegularizerObjGrad(PetscTAO, PetscVec, PetscReal*, PetscVec, void*) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode PetscTaoVarBounds(PetscTAO, PetscVec, PetscVec, void*) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode PetscTaoConstraints(PetscTAO, PetscVec, PetscVec, void*) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode PetscTaoEqualityConstraints(PetscTAO, PetscVec, PetscVec, void*) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode PetscTaoInequalityConstraints(PetscTAO, PetscVec, PetscVec, void*) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode PetscTaoHessian(PetscTAO, PetscVec, PetscMat, PetscMat, void*) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode PetscTaoRegularizerHessian(PetscTAO, PetscVec, PetscMat, void*) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode PetscTaoJacobian(PetscTAO, PetscVec, PetscMat, PetscMat, void*) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode PetscTaoJacobianResidual(PetscTAO, PetscVec, PetscMat, PetscMat, void*) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode PetscTaoJacobianState(PetscTAO, PetscVec, PetscMat, PetscMat, PetscMat, void*) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode PetscTaoJacobianDesign(PetscTAO, PetscVec, PetscMat, void*) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode PetscTaoJacobianEquality(PetscTAO, PetscVec, PetscMat, PetscMat, void*) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode PetscTaoJacobianInequality(PetscTAO, PetscVec, PetscMat, PetscMat, void*) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode PetscTaoUpdateFunction(PetscTAO, PetscInt, void*) except PETSC_ERR_PYTHON

    ctypedef enum PetscTAOBNCGType "TaoBNCGType":
        TAO_BNCG_GD
        TAO_BNCG_PCGD
        TAO_BNCG_HS
        TAO_BNCG_FR
        TAO_BNCG_PRP
        TAO_BNCG_PRP_PLUS
        TAO_BNCG_DY
        TAO_BNCG_HZ
        TAO_BNCG_DK
        TAO_BNCG_KD
        TAO_BNCG_SSML_BFGS
        TAO_BNCG_SSML_DFP
        TAO_BNCG_SSML_BRDN

    ctypedef enum PetscTAOALMMType "TaoALMMType":
        TAO_ALMM_CLASSIC
        TAO_ALMM_PHR

    PetscErrorCode TaoMonitor(PetscTAO, PetscInt, PetscReal, PetscReal, PetscReal, PetscReal)
    PetscErrorCode TaoView(PetscTAO, PetscViewer)
    PetscErrorCode TaoDestroy(PetscTAO*)
    PetscErrorCode TaoCreate(MPI_Comm, PetscTAO*)
    PetscErrorCode TaoSetOptionsPrefix(PetscTAO, char[])
    PetscErrorCode TaoAppendOptionsPrefix(PetscTAO, char[])
    PetscErrorCode TaoGetOptionsPrefix(PetscTAO, char*[])
    PetscErrorCode TaoSetFromOptions(PetscTAO)
    PetscErrorCode TaoSetType(PetscTAO, PetscTAOType)
    PetscErrorCode TaoGetType(PetscTAO, PetscTAOType*)

    PetscErrorCode TaoSetUp(PetscTAO)
    PetscErrorCode TaoSolve(PetscTAO)

    PetscErrorCode TaoSetTolerances(PetscTAO, PetscReal, PetscReal, PetscReal)
    PetscErrorCode TaoParametersInitialize(PetscTAO)
    PetscErrorCode TaoGetTolerances(PetscTAO, PetscReal*, PetscReal*, PetscReal*)
    PetscErrorCode TaoSetConstraintTolerances(PetscTAO, PetscReal, PetscReal)
    PetscErrorCode TaoGetConstraintTolerances(PetscTAO, PetscReal*, PetscReal*)

    PetscErrorCode TaoSetFunctionLowerBound(PetscTAO, PetscReal)
    PetscErrorCode TaoSetMaximumIterations(PetscTAO, PetscInt)
    PetscErrorCode TaoGetMaximumIterations(PetscTAO, PetscInt*)
    PetscErrorCode TaoSetMaximumFunctionEvaluations(PetscTAO, PetscInt)
    PetscErrorCode TaoGetMaximumFunctionEvaluations(PetscTAO, PetscInt*)
    PetscErrorCode TaoSetIterationNumber(PetscTAO, PetscInt)
    PetscErrorCode TaoGetIterationNumber(PetscTAO, PetscInt*)

    PetscErrorCode TaoSetTrustRegionTolerance(PetscTAO, PetscReal)
    PetscErrorCode TaoGetInitialTrustRegionRadius(PetscTAO, PetscReal*)
    PetscErrorCode TaoGetTrustRegionRadius(PetscTAO, PetscReal*)
    PetscErrorCode TaoSetTrustRegionRadius(PetscTAO, PetscReal)

    PetscErrorCode TaoDefaultConvergenceTest(PetscTAO, void*) except PETSC_ERR_PYTHON
    PetscErrorCode TaoSetConvergenceTest(PetscTAO, PetscTaoConvergenceTest*, void*)
    PetscErrorCode TaoSetConvergedReason(PetscTAO, PetscTAOConvergedReason)
    PetscErrorCode TaoGetConvergedReason(PetscTAO, PetscTAOConvergedReason*)
    PetscErrorCode TaoLogConvergenceHistory(PetscTAO, PetscReal, PetscReal, PetscReal, PetscInt)
    PetscErrorCode TaoGetSolutionStatus(PetscTAO, PetscInt*,
                                        PetscReal*, PetscReal*,
                                        PetscReal*, PetscReal*,
                                        PetscTAOConvergedReason*)

    PetscErrorCode TaoMonitorSet(PetscTAO, PetscTaoMonitor, void*, PetscTaoMonitorDestroy)
    PetscErrorCode TaoMonitorCancel(PetscTAO)

    PetscErrorCode TaoComputeObjective(PetscTAO, PetscVec, PetscReal*)
    PetscErrorCode TaoComputeResidual(PetscTAO, PetscVec, PetscVec)
    PetscErrorCode TaoComputeGradient(PetscTAO, PetscVec, PetscVec)
    PetscErrorCode TaoComputeObjectiveAndGradient(PetscTAO, PetscVec, PetscReal*, PetscVec)
    PetscErrorCode TaoComputeConstraints(PetscTAO, PetscVec, PetscVec)
    PetscErrorCode TaoComputeDualVariables(PetscTAO, PetscVec, PetscVec)
    PetscErrorCode TaoComputeVariableBounds(PetscTAO)
    PetscErrorCode TaoComputeHessian(PetscTAO, PetscVec, PetscMat, PetscMat)
    PetscErrorCode TaoComputeJacobian(PetscTAO, PetscVec, PetscMat, PetscMat)

    PetscErrorCode TaoSetSolution(PetscTAO, PetscVec)
    PetscErrorCode TaoSetConstraintsVec(PetscTAO, PetscVec)
    PetscErrorCode TaoSetVariableBounds(PetscTAO, PetscVec, PetscVec)

    PetscErrorCode TaoGetSolution(PetscTAO, PetscVec*)
    PetscErrorCode TaoSetGradientNorm(PetscTAO, PetscMat)
    PetscErrorCode TaoGetGradientNorm(PetscTAO, PetscMat*)
    PetscErrorCode TaoLMVMSetH0(PetscTAO, PetscMat)
    PetscErrorCode TaoLMVMGetH0(PetscTAO, PetscMat*)
    PetscErrorCode TaoLMVMGetH0KSP(PetscTAO, PetscKSP*)
    PetscErrorCode TaoBNCGGetType(PetscTAO, PetscTAOBNCGType*)
    PetscErrorCode TaoBNCGSetType(PetscTAO, PetscTAOBNCGType)
    PetscErrorCode TaoGetVariableBounds(PetscTAO, PetscVec*, PetscVec*)

    PetscErrorCode TaoSetObjective(PetscTAO, PetscTaoObjective*, void*)
    PetscErrorCode TaoSetGradient(PetscTAO, PetscVec, PetscTaoGradient*, void*)
    PetscErrorCode TaoSetObjectiveAndGradient(PetscTAO, PetscVec, PetscTaoObjGrad*, void*)
    PetscErrorCode TaoSetHessian(PetscTAO, PetscMat, PetscMat, PetscTaoHessian*, void*)
    PetscErrorCode TaoGetObjective(PetscTAO, PetscTaoObjective**, void**)
    PetscErrorCode TaoGetGradient(PetscTAO, PetscVec*, PetscTaoGradient**, void**)
    PetscErrorCode TaoGetObjectiveAndGradient(PetscTAO, PetscVec*, PetscTaoObjGrad**, void**)
    PetscErrorCode TaoGetHessian(PetscTAO, PetscMat*, PetscMat*, PetscTaoHessian**, void**)
    PetscErrorCode TaoSetResidualRoutine(PetscTAO, PetscVec, PetscTaoResidual, void*)
    PetscErrorCode TaoSetVariableBoundsRoutine(PetscTAO, PetscTaoVarBounds*, void*)
    PetscErrorCode TaoSetConstraintsRoutine(PetscTAO, PetscVec, PetscTaoConstraints*, void*)
    PetscErrorCode TaoSetJacobianRoutine(PetscTAO, PetscMat, PetscMat, PetscTaoJacobian*, void*)
    PetscErrorCode TaoSetJacobianResidualRoutine(PetscTAO, PetscMat, PetscMat, PetscTaoJacobianResidual*, void*)
    PetscErrorCode TaoSetStateDesignIS(PetscTAO, PetscIS, PetscIS)
    PetscErrorCode TaoSetJacobianStateRoutine(PetscTAO, PetscMat, PetscMat, PetscMat, PetscTaoJacobianState*, void*)
    PetscErrorCode TaoSetJacobianDesignRoutine(PetscTAO, PetscMat, PetscTaoJacobianDesign*, void*)
    PetscErrorCode TaoGetLMVMMatrix(PetscTAO, PetscMat*)
    PetscErrorCode TaoSetLMVMMatrix(PetscTAO, PetscMat)

    PetscErrorCode TaoSetEqualityConstraintsRoutine(PetscTAO, PetscVec, PetscTaoEqualityConstraints*, void*)
    PetscErrorCode TaoGetEqualityConstraintsRoutine(PetscTAO, PetscVec*, PetscTaoEqualityConstraints**, void**)
    PetscErrorCode TaoSetJacobianEqualityRoutine(PetscTAO, PetscMat, PetscMat, PetscTaoJacobianEquality*, void*)
    PetscErrorCode TaoGetJacobianEqualityRoutine(PetscTAO, PetscMat*, PetscMat*, PetscTaoJacobianEquality**, void**)
    PetscErrorCode TaoSetInequalityConstraintsRoutine(PetscTAO, PetscVec, PetscTaoInequalityConstraints*, void*)
    PetscErrorCode TaoGetInequalityConstraintsRoutine(PetscTAO, PetscVec*, PetscTaoInequalityConstraints**, void**)
    PetscErrorCode TaoSetJacobianInequalityRoutine(PetscTAO, PetscMat, PetscMat, PetscTaoJacobianInequality*, void*)
    PetscErrorCode TaoGetJacobianInequalityRoutine(PetscTAO, PetscMat*, PetscMat*, PetscTaoJacobianInequality**, void**)
    PetscErrorCode TaoSetUpdate(PetscTAO, PetscTaoUpdateFunction*, void*)

    PetscErrorCode TaoSetInitialTrustRegionRadius(PetscTAO, PetscReal)

    PetscErrorCode TaoGetKSP(PetscTAO, PetscKSP*)
    PetscErrorCode TaoGetLineSearch(PetscTAO, PetscTAOLineSearch*)

    PetscErrorCode TaoALMMGetSubsolver(PetscTAO, PetscTAO*)
    PetscErrorCode TaoALMMSetSubsolver(PetscTAO, PetscTAO)
    PetscErrorCode TaoALMMGetType(PetscTAO, PetscTAOALMMType*)
    PetscErrorCode TaoALMMSetType(PetscTAO, PetscTAOALMMType)

    PetscErrorCode TaoBRGNGetSubsolver(PetscTAO, PetscTAO*)
    PetscErrorCode TaoBRGNSetRegularizerObjectiveAndGradientRoutine(PetscTAO, PetscTaoRegularizerObjGrad*, void*)
    PetscErrorCode TaoBRGNSetRegularizerHessianRoutine(PetscTAO, PetscMat, PetscTaoRegularizerHessian*, void*)
    PetscErrorCode TaoBRGNSetRegularizerWeight(PetscTAO, PetscReal)
    PetscErrorCode TaoBRGNSetL1SmoothEpsilon(PetscTAO, PetscReal)
    PetscErrorCode TaoBRGNSetDictionaryMatrix(PetscTAO, PetscMat)
    PetscErrorCode TaoBRGNGetDampingVector(PetscTAO, PetscVec*)

    PetscErrorCode TaoPythonSetType(PetscTAO, char[])
    PetscErrorCode TaoPythonGetType(PetscTAO, char*[])

    ctypedef const char* PetscTAOLineSearchType "TaoLineSearchType"
    PetscTAOLineSearchType TAOLINESEARCHUNIT
    PetscTAOLineSearchType TAOLINESEARCHARMIJO
    PetscTAOLineSearchType TAOLINESEARCHOWARMIJO
    PetscTAOLineSearchType TAOLINESEARCHGPCG
    PetscTAOLineSearchType TAOLINESEARCHMT
    PetscTAOLineSearchType TAOLINESEARCHIPM

    ctypedef enum PetscTAOLineSearchConvergedReason "TaoLineSearchConvergedReason":
        # failed
        TAOLINESEARCH_FAILED_INFORNAN
        TAOLINESEARCH_FAILED_BADPARAMETER
        TAOLINESEARCH_FAILED_ASCENT
        # continue
        TAOLINESEARCH_CONTINUE_ITERATING
        # success
        TAOLINESEARCH_SUCCESS
        TAOLINESEARCH_SUCCESS_USER
        # halted
        TAOLINESEARCH_HALTED_OTHER
        TAOLINESEARCH_HALTED_MAXFCN
        TAOLINESEARCH_HALTED_UPPERBOUND
        TAOLINESEARCH_HALTED_LOWERBOUND
        TAOLINESEARCH_HALTED_RTOL
        TAOLINESEARCH_HALTED_USER

    ctypedef PetscErrorCode PetscTaoLineSearchObjective(PetscTAOLineSearch, PetscVec, PetscReal*, void*) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode PetscTaoLineSearchGradient(PetscTAOLineSearch, PetscVec, PetscVec, void*) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode PetscTaoLineSearchObjGrad(PetscTAOLineSearch, PetscVec, PetscReal*, PetscVec, void*) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode PetscTaoLineSearchObjGTS(PetscTaoLineSearch, PetscVec, PetscVec, PetscReal*, PetscReal*, void*) except PETSC_ERR_PYTHON

    PetscErrorCode TaoLineSearchCreate(MPI_Comm, PetscTAOLineSearch*)
    PetscErrorCode TaoLineSearchDestroy(PetscTAOLineSearch*)
    PetscErrorCode TaoLineSearchView(PetscTAOLineSearch, PetscViewer)
    PetscErrorCode TaoLineSearchSetType(PetscTAOLineSearch, PetscTAOLineSearchType)
    PetscErrorCode TaoLineSearchGetType(PetscTAOLineSearch, PetscTAOLineSearchType*)
    PetscErrorCode TaoLineSearchSetOptionsPrefix(PetscTAOLineSearch, char[])
    PetscErrorCode TaoLineSearchGetOptionsPrefix(PetscTAOLineSearch, char*[])
    PetscErrorCode TaoLineSearchSetFromOptions(PetscTAOLineSearch)
    PetscErrorCode TaoLineSearchSetUp(PetscTAOLineSearch)
    PetscErrorCode TaoLineSearchUseTaoRoutines(PetscTAOLineSearch, PetscTAO)
    PetscErrorCode TaoLineSearchSetObjectiveRoutine(PetscTAOLineSearch, PetscTaoLineSearchObjective, void*)
    PetscErrorCode TaoLineSearchSetGradientRoutine(PetscTAOLineSearch, PetscTaoLineSearchGradient, void*)
    PetscErrorCode TaoLineSearchSetObjectiveAndGradientRoutine(PetscTAOLineSearch, PetscTaoLineSearchObjGrad, void*)
    PetscErrorCode TaoLineSearchApply(PetscTAOLineSearch, PetscVec, PetscReal*, PetscVec, PetscVec, PetscReal*, PetscTAOLineSearchConvergedReason*)
    PetscErrorCode TaoLineSearchSetInitialStepLength(PetscTAOLineSearch, PetscReal)

# --------------------------------------------------------------------

cdef inline TAO ref_TAO(PetscTAO tao):
    cdef TAO ob = <TAO> TAO()
    ob.tao = tao
    CHKERR(PetscINCREF(ob.obj))
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
                                      PetscMat  _Jp,
                                      PetscMat  _Ji,
                                      void* ctx) except PETSC_ERR_PYTHON with gil:
    cdef TAO tao = ref_TAO(_tao)
    cdef Vec x   = ref_Vec(_x)
    cdef Mat J   = ref_Mat(_J)
    cdef Mat Jp  = ref_Mat(_Jp)
    cdef Mat Ji  = ref_Mat(_Ji)
    context = tao.get_attr("__jacobian_state__")
    if context is None and ctx != NULL: context = <object>ctx
    assert context is not None and type(context) is tuple # sanity check
    (jacobian, args, kargs) = context
    jacobian(tao, x, J, Jp, Ji, *args, **kargs)
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

cdef PetscErrorCode TAO_InequalityConstraints(PetscTAO _tao,
                                              PetscVec  _x,
                                              PetscVec  _c,
                                              void* ctx) except PETSC_ERR_PYTHON with gil:
    cdef TAO tao = ref_TAO(_tao)
    cdef Vec x   = ref_Vec(_x)
    cdef Vec c   = ref_Vec(_c)
    context = tao.get_attr("__inequality_constraints__")
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

cdef PetscErrorCode TAO_JacobianInequality(PetscTAO _tao,
                                           PetscVec  _x,
                                           PetscMat  _J,
                                           PetscMat  _P,
                                           void* ctx) except PETSC_ERR_PYTHON with gil:
    cdef TAO tao = ref_TAO(_tao)
    cdef Vec x   = ref_Vec(_x)
    cdef Mat J   = ref_Mat(_J)
    cdef Mat P   = ref_Mat(_P)
    context = tao.get_attr("__jacobian_inequality__")
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
    (update, args, kargs) = tao.get_attr('__update__')
    update(tao, toInt(its), *args, **kargs)
    return PETSC_SUCCESS

cdef PetscErrorCode TAO_Converged(PetscTAO _tao,
                                  void* ctx) except PETSC_ERR_PYTHON with gil:
    # call first the default convergence test
    CHKERR(TaoDefaultConvergenceTest(_tao, NULL))
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
    CHKERR(TaoSetConvergedReason(_tao, creason))
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

cdef inline TAOLineSearch ref_TAOLS(PetscTAOLineSearch taols):
    cdef TAOLineSearch ob = <TAOLineSearch> TAOLineSearch()
    ob.taols = taols
    CHKERR(PetscINCREF(ob.obj))
    return ob

# --------------------------------------------------------------------

cdef PetscErrorCode TAOLS_Objective(PetscTAOLineSearch _ls,
                                    PetscVec _x,
                                    PetscReal *_f,
                                    void *ctx) except PETSC_ERR_PYTHON with gil:

    cdef TAOLineSearch ls = ref_TAOLS(_ls)
    cdef Vec x   = ref_Vec(_x)
    (objective, args, kargs) = ls.get_attr("__objective__")
    retv = objective(ls, x, *args, **kargs)
    _f[0] = asReal(retv)
    return PETSC_SUCCESS

cdef PetscErrorCode TAOLS_Gradient(PetscTAOLineSearch _ls,
                                   PetscVec _x,
                                   PetscVec _g,
                                   void *ctx) except PETSC_ERR_PYTHON with gil:

    cdef TAOLineSearch ls = ref_TAOLS(_ls)
    cdef Vec x   = ref_Vec(_x)
    cdef Vec g   = ref_Vec(_g)
    (gradient, args, kargs) = ls.get_attr("__gradient__")
    gradient(ls, x, g, *args, **kargs)
    return PETSC_SUCCESS


cdef PetscErrorCode TAOLS_ObjGrad(PetscTAOLineSearch _ls,
                                  PetscVec _x,
                                  PetscReal *_f,
                                  PetscVec _g,
                                  void *ctx) except PETSC_ERR_PYTHON with gil:

    cdef TAOLineSearch ls = ref_TAOLS(_ls)
    cdef Vec x   = ref_Vec(_x)
    cdef Vec g   = ref_Vec(_g)
    (objgrad, args, kargs) = ls.get_attr("__objgrad__")
    retv = objgrad(ls, x, g, *args, **kargs)
    _f[0] = asReal(retv)
    return PETSC_SUCCESS
