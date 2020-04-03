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
    PetscTAOType TAOLCL
    PetscTAOType TAOSSILS
    PetscTAOType TAOSSFLS
    PetscTAOType TAOASILS
    PetscTAOType TAOASFLS
    PetscTAOType TAOIPM

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

    int TaoView(PetscTAO,PetscViewer)
    int TaoDestroy(PetscTAO*)
    int TaoCreate(MPI_Comm,PetscTAO*)
    int TaoSetOptionsPrefix(PetscTAO, char[])
    int TaoGetOptionsPrefix(PetscTAO, char*[])
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
    int TaoSetMaximumIterates(PetscTAO, PetscInt)
    int TaoSetMaximumFunctionEvaluations(PetscTAO, PetscInt)

    int TaoSetTrustRegionTolerance(PetscTAO,PetscReal)
    int TaoGetInitialTrustRegionRadius(PetscTAO,PetscReal*)
    int TaoGetTrustRegionRadius(PetscTAO,PetscReal*)
    int TaoSetTrustRegionRadius(PetscTAO,PetscReal)

    ctypedef int TaoConvergenceTest(PetscTAO,void*) except PETSC_ERR_PYTHON
    int TaoDefaultConvergenceTest(PetscTAO tao,void *dummy) except PETSC_ERR_PYTHON
    int TaoSetConvergenceTest(PetscTAO, TaoConvergenceTest*, void*)
    int TaoSetConvergedReason(PetscTAO,PetscTAOConvergedReason)
    int TaoGetConvergedReason(PetscTAO,PetscTAOConvergedReason*)
    int TaoGetSolutionStatus(PetscTAO,PetscInt*,
                             PetscReal*,PetscReal*,
                             PetscReal*,PetscReal*,
                             PetscTAOConvergedReason*)

    ctypedef int TaoMonitor(PetscTAO,void*) except PETSC_ERR_PYTHON
    ctypedef int (*TaoMonitorDestroy)(void**)
    int TaoSetMonitor(PetscTAO,TaoMonitor,void*,TaoMonitorDestroy)
    int TaoCancelMonitors(PetscTAO)





    int TaoComputeObjective(PetscTAO,PetscVec,PetscReal*)
    int TaoComputeResidual(PetscTAO,PetscVec,PetscVec)
    int TaoComputeGradient(PetscTAO,PetscVec,PetscVec)
    int TaoComputeObjectiveAndGradient(PetscTAO,PetscVec,PetscReal*,PetscVec)
    int TaoComputeConstraints(PetscTAO,PetscVec,PetscVec)
    int TaoComputeDualVariables(PetscTAO,PetscVec,PetscVec)
    int TaoComputeVariableBounds(PetscTAO)
    int TaoComputeHessian (PetscTAO,PetscVec,PetscMat,PetscMat)
    int TaoComputeJacobian(PetscTAO,PetscVec,PetscMat,PetscMat)

    int TaoSetInitialVector(PetscTAO,PetscVec)
    int TaoSetConstraintsVec(PetscTAO,PetscVec)
    int TaoSetVariableBounds(PetscTAO,PetscVec,PetscVec)
    int TaoSetHessianMat(PetscTAO,PetscMat,PetscMat)
    int TaoSetJacobianMat(PetscTAO,PetscMat,PetscMat)

    int TaoGetSolutionVector(PetscTAO,PetscVec*)
    int TaoGetGradientVector(PetscTAO,PetscVec*)
    int TaoSetGradientNorm(PetscTAO,PetscMat)
    int TaoGetGradientNorm(PetscTAO,PetscMat*)
    int TaoLMVMSetH0(PetscTAO,PetscMat)
    int TaoLMVMGetH0(PetscTAO,PetscMat*)
    int TaoLMVMGetH0KSP(PetscTAO,PetscKSP*)
    int TaoGetVariableBounds(PetscTAO,PetscVec*,PetscVec*)
    #int TaoGetConstraintsVec(PetscTAO,PetscVec*)
    #int TaoGetVariableBoundVecs(PetscTAO,PetscVec*,PetscVec*)
    #int TaoGetHessianMat(PetscTAO,PetscMat*,PetscMat*)
    #int TaoGetJacobianMat(PetscTAO,PetscMat*,PetscMat*)

    ctypedef int TaoObjective(PetscTAO,PetscVec,PetscReal*,void*) except PETSC_ERR_PYTHON
    ctypedef int TaoResidual(PetscTAO,PetscVec,PetscVec,void*) except PETSC_ERR_PYTHON
    ctypedef int TaoGradient(PetscTAO,PetscVec,PetscVec,void*) except PETSC_ERR_PYTHON
    ctypedef int TaoObjGrad(PetscTAO,PetscVec,PetscReal*,PetscVec,void*) except PETSC_ERR_PYTHON
    ctypedef int TaoVarBounds(PetscTAO,PetscVec,PetscVec,void*) except PETSC_ERR_PYTHON
    ctypedef int TaoConstraints(PetscTAO,PetscVec,PetscVec,void*) except PETSC_ERR_PYTHON
    ctypedef int TaoHessian(PetscTAO,PetscVec,
                            PetscMat,PetscMat,
                            void*) except PETSC_ERR_PYTHON
    ctypedef int TaoJacobian(PetscTAO,PetscVec,
                             PetscMat,PetscMat,
                             void*) except PETSC_ERR_PYTHON
    ctypedef int TaoJacobianState(PetscTAO,PetscVec,
                                  PetscMat,PetscMat,PetscMat,
                                  void*) except PETSC_ERR_PYTHON
    ctypedef int TaoJacobianDesign(PetscTAO,PetscVec,PetscMat,
                                   void*) except PETSC_ERR_PYTHON

    int TaoSetObjectiveRoutine(PetscTAO,TaoObjective*,void*)
    int TaoSetResidualRoutine(PetscTAO,PetscVec,TaoResidual,void*)
    int TaoSetGradientRoutine(PetscTAO,TaoGradient*,void*)
    int TaoSetObjectiveAndGradientRoutine(PetscTAO,TaoObjGrad*,void*)
    int TaoSetVariableBoundsRoutine(PetscTAO,TaoVarBounds*,void*)
    int TaoSetConstraintsRoutine(PetscTAO,PetscVec,TaoConstraints*,void*)
    int TaoSetHessianRoutine(PetscTAO,PetscMat,PetscMat,TaoHessian*,void*)
    int TaoSetJacobianRoutine(PetscTAO,PetscMat,PetscMat,TaoJacobian*,void*)

    int TaoSetStateDesignIS(PetscTAO,PetscIS,PetscIS)
    int TaoSetJacobianStateRoutine(PetscTAO,PetscMat,PetscMat,PetscMat,TaoJacobianState*,void*)
    int TaoSetJacobianDesignRoutine(PetscTAO,PetscMat,TaoJacobianDesign*,void*)

    int TaoSetInitialTrustRegionRadius(PetscTAO,PetscReal)

    int TaoGetKSP(PetscTAO,PetscKSP*)

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
    (objective, args, kargs) = tao.get_attr("__objective__")
    retv = objective(tao, x, *args, **kargs)
    _f[0] = asReal(retv)
    return 0

cdef int TAO_Residual(PetscTAO _tao,
                      PetscVec _x, PetscVec _r,
                      void *ctx) except PETSC_ERR_PYTHON with gil:

    cdef TAO tao = ref_TAO(_tao)
    cdef Vec x   = ref_Vec(_x)
    cdef Vec r   = ref_Vec(_r)
    (residual, args, kargs) = tao.get_attr("__residual__")
    residual(tao, x, r, *args, **kargs)
    return 0

cdef int TAO_Gradient(PetscTAO _tao,
                      PetscVec _x, PetscVec _g,
                      void *ctx) except PETSC_ERR_PYTHON with gil:

    cdef TAO tao = ref_TAO(_tao)
    cdef Vec x   = ref_Vec(_x)
    cdef Vec g   = ref_Vec(_g)
    (gradient, args, kargs) = tao.get_attr("__gradient__")
    gradient(tao, x, g, *args, **kargs)
    return 0


cdef int TAO_ObjGrad(PetscTAO _tao,
                     PetscVec _x, PetscReal *_f, PetscVec _g,
                     void *ctx) except PETSC_ERR_PYTHON with gil:

    cdef TAO tao = ref_TAO(_tao)
    cdef Vec x   = ref_Vec(_x)
    cdef Vec g   = ref_Vec(_g)
    (objgrad, args, kargs) = tao.get_attr("__objgrad__")
    retv = objgrad(tao, x, g, *args, **kargs)
    _f[0] = asReal(retv)
    return 0

cdef int TAO_Constraints(PetscTAO _tao,
                         PetscVec _x, PetscVec _r,
                         void *ctx) except PETSC_ERR_PYTHON with gil:

    cdef TAO tao = ref_TAO(_tao)
    cdef Vec x   = ref_Vec(_x)
    cdef Vec r   = ref_Vec(_r)
    (constraints, args, kargs) = tao.get_attr("__constraints__")
    constraints(tao, x, r, *args, **kargs)
    return 0

cdef int TAO_VarBounds(PetscTAO _tao,
                       PetscVec _xl, PetscVec _xu,
                       void *ctx) except PETSC_ERR_PYTHON with gil:

    cdef TAO tao = ref_TAO(_tao)
    cdef Vec xl  = ref_Vec(_xl)
    cdef Vec xu  = ref_Vec(_xu)
    (varbounds, args, kargs) = tao.get_attr("__varbounds__")
    varbounds(tao, xl, xu, *args, **kargs)
    return 0

cdef int TAO_Hessian(PetscTAO _tao,
                     PetscVec  _x,
                     PetscMat  _H,
                     PetscMat  _P,
                     void* ctx) except PETSC_ERR_PYTHON with gil:
    cdef TAO tao = ref_TAO(_tao)
    cdef Vec x   = ref_Vec(_x)
    cdef Mat H   = ref_Mat(_H)
    cdef Mat P   = ref_Mat(_P)
    (hessian, args, kargs) = tao.get_attr("__hessian__")
    hessian(tao, x, H, P, *args, **kargs)
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
    (jacobian, args, kargs) = tao.get_attr("__jacobian__")
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
    (jacobian, args, kargs) = tao.get_attr("__jacobian_state__")
    jacobian(tao, x, J, P, I, *args, **kargs)
    return 0

cdef int TAO_JacobianDesign(PetscTAO _tao,
                            PetscVec  _x,
                            PetscMat  _J,
                            void* ctx) except PETSC_ERR_PYTHON with gil:
    cdef TAO tao = ref_TAO(_tao)
    cdef Vec x   = ref_Vec(_x)
    cdef Mat J   = ref_Mat(_J)
    (jacobian, args, kargs) = tao.get_attr("__jacobian_design__")
    jacobian(tao, x, J, *args, **kargs)
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
