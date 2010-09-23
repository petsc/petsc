# --------------------------------------------------------------------

cdef extern from "petscda.h" nogil:

    ctypedef enum PetscDAStencilType "DAStencilType":
        DA_STENCIL_STAR
        DA_STENCIL_BOX

    ctypedef enum PetscDAPeriodicType "DAPeriodicType":
        DA_PERIODIC_NONE  "DA_NONPERIODIC"
        DA_PERIODIC_X     "DA_XPERIODIC"
        DA_PERIODIC_Y     "DA_YPERIODIC"
        DA_PERIODIC_Z     "DA_ZPERIODIC"
        DA_PERIODIC_XY    "DA_XYPERIODIC"
        DA_PERIODIC_XZ    "DA_XZPERIODIC"
        DA_PERIODIC_YZ    "DA_YZPERIODIC"
        DA_PERIODIC_XYZ   "DA_XYZPERIODIC"
        DA_GHOSTED_XYZ    "DA_XYZGHOSTED"

    ctypedef enum PetscDAInterpolationType "DAInterpolationType":
        DA_INTERPOLATION_Q0 "DA_Q0"
        DA_INTERPOLATION_Q1 "DA_Q1"

    ctypedef enum PetscDAElementType "DAElementType":
        DA_ELEMENT_P1
        DA_ELEMENT_Q1

    int DAView(PetscDA,PetscViewer)
    int DADestroy(PetscDA)

    int DACreateND(MPI_Comm,
                   PetscInt,PetscInt,                # dim, dof
                   PetscInt,PetscInt,PetscInt,       # M, N, P
                   PetscInt,PetscInt,PetscInt,       # m, n, p
                   PetscInt[],PetscInt[],PetscInt[], # lx, ly, lz
                   PetscDAPeriodicType,              # periodicity
                   PetscDAStencilType,               # stencil type
                   PetscInt,                         # stencil width
                   PetscDA*)

    int DASetOptionsPrefix(PetscDA,char[])
    int DASetFromOptions(PetscDA)
    int DASetElementType(PetscDA,PetscDAElementType)
    int DASetInterpolationType(PetscDA,PetscDAInterpolationType)
    int DAGetInterpolation(PetscDA,PetscDA,PetscMat*,PetscVec*)
    int DAGetInfo(PetscDA,
                  PetscInt*,
                  PetscInt*,PetscInt*,PetscInt*,
                  PetscInt*,PetscInt*,PetscInt*,
                  PetscInt*,PetscInt*,
                  PetscDAPeriodicType*,PetscDAStencilType*)

    int DAGetCorners(PetscDA,
                     PetscInt*,PetscInt*,PetscInt*,
                     PetscInt*,PetscInt*,PetscInt*)
    int DAGetGhostCorners(PetscDA,
                          PetscInt*,PetscInt*,PetscInt*,
                          PetscInt*,PetscInt*,PetscInt*)

    int DASetUniformCoordinates(PetscDA,
                                PetscReal,PetscReal,
                                PetscReal,PetscReal,
                                PetscReal,PetscReal)
    int DASetCoordinates(PetscDA,PetscVec)
    int DAGetCoordinates(PetscDA,PetscVec*)
    int DAGetCoordinateDA(PetscDA,PetscDA*)
    int DAGetGhostedCoordinates(PetscDA,PetscVec*)

    int DACreateGlobalVector(PetscDA,PetscVec*)
    int DACreateNaturalVector(PetscDA,PetscVec*)
    int DACreateLocalVector(PetscDA,PetscVec*)

    int DAGlobalToLocalBegin(PetscDA,PetscVec,PetscInsertMode,PetscVec)
    int DAGlobalToLocalEnd(PetscDA,PetscVec,PetscInsertMode,PetscVec)
    int DAGlobalToNaturalBegin(PetscDA,PetscVec,PetscInsertMode,PetscVec)
    int DAGlobalToNaturalEnd(PetscDA,PetscVec,PetscInsertMode,PetscVec)
    int DANaturalToGlobalBegin(PetscDA,PetscVec,PetscInsertMode,PetscVec)
    int DANaturalToGlobalEnd(PetscDA,PetscVec,PetscInsertMode,PetscVec)
    int DALocalToLocalBegin(PetscDA,PetscVec,PetscInsertMode,PetscVec)
    int DALocalToLocalEnd(PetscDA,PetscVec,PetscInsertMode,PetscVec)
    int DALocalToGlobal(PetscDA,PetscVec,PetscInsertMode,PetscVec)
    int DALocalToGlobalBegin(PetscDA,PetscVec,PetscVec)
    int DALocalToGlobalEnd(PetscDA,PetscVec,PetscVec)

    int DAGetMatrix(PetscDA,PetscMatType,PetscMat*)

    int DAGetAO(PetscDA,PetscAO*)
    int DAGetISLocalToGlobalMapping(PetscDA,PetscLGMap*)
    int DAGetISLocalToGlobalMappingBlck(PetscDA,PetscLGMap*)
    int DAGetScatter(PetscDA,PetscScatter*,PetscScatter*,PetscScatter*)

    int DASetRefinementFactor(PetscDA,PetscInt,PetscInt,PetscInt)
    int DAGetRefinementFactor(PetscDA,PetscInt*,PetscInt*,PetscInt*)
    int DARefine(PetscDA,MPI_Comm,PetscDA*)

    #int DASetFieldName(PetscDA,PetscInt,const_char[])
    #int DAGetFieldName(PetscDA,PetscInt,char**)

# --------------------------------------------------------------------

cdef inline int DAGetDim(PetscDA da, PetscInt *dim) nogil:
     return DAGetInfo(da, dim,
                      NULL, NULL, NULL,
                      NULL, NULL, NULL,
                      NULL, NULL,
                      NULL, NULL)

cdef inline PetscInt asDims(dims,
                            PetscInt *_M, 
                            PetscInt *_N, 
                            PetscInt *_P) except -1:
    cdef PetscInt ndim = len(dims)
    cdef object M, N, P
    if ndim == 1: 
        M, = dims
    elif ndim == 2: 
        M, N = dims
    elif ndim == 3: 
        M, N, P = dims
    _M[0] = _N[0] = _P[0] = PETSC_DECIDE
    if ndim >= 1: _M[0] = asInt(M)
    if ndim >= 2: _N[0] = asInt(N)
    if ndim >= 3: _P[0] = asInt(P)
    return ndim

cdef inline tuple toDims(PetscInt dim,
                         PetscInt M,
                         PetscInt N, 
                         PetscInt P):
        if dim == 0:
            return ()
        elif dim == 1:
            return (toInt(M),)
        elif dim == 2:
            return (toInt(M), toInt(N))
        else:
            return (toInt(M), toInt(N), toInt(P))

# --------------------------------------------------------------------

cdef class _DA_Vec_array(object):

    cdef readonly ndarray array
    cdef readonly tuple   starts
    cdef readonly tuple   sizes

    def __cinit__(self, DA da not None, Vec vec not None):
        #
        cdef PetscInt dim, dof
        CHKERR( DAGetInfo(da.da,
                          &dim, NULL, NULL, NULL, NULL, NULL, NULL,
                          &dof, NULL, NULL, NULL) )
        cdef PetscInt lxs, lys, lzs, lxm, lym, lzm
        CHKERR( DAGetCorners(da.da,
                             &lxs, &lys, &lzs,
                             &lxm, &lym, &lzm) )
        cdef PetscInt gxs, gys, gzs, gxm, gym, gzm
        CHKERR( DAGetGhostCorners(da.da,
                                  &gxs, &gys, &gzs,
                                  &gxm, &gym, &gzm) )
        #
        cdef PetscInt n
        CHKERR( VecGetLocalSize(vec.vec, &n) )
        cdef PetscInt xs, ys, zs, xm, ym, zm
        if (n == lxm*lym*lzm*dof):
            xs, ys, zs = lxs, lys, lzs
            xm, ym, zm = lxm, lym, lzm
        elif (n == gxm*gym*gzm*dof):
            xs, ys, zs = gxs, gys, gzs
            xm, ym, zm = gxm, gym, gzm
        else:
            raise ValueError(
                "Vector local size %d is not compatible with DA local sizes %s"
                % (<Py_ssize_t>n, toDims(dim, lxm, lym, lzm)))
        #
        cdef tuple starts = toDims(dim, xs, ys, zs)
        cdef tuple sizes  = toDims(dim, xm, ym, zm)
        cdef PetscInt k = sizeof(PetscScalar)
        cdef tuple shape   = toDims(dim, xm, ym, zm)
        cdef tuple strides = toDims(dim, k*1, k*xm, k*xm*ym)
        if dof > 1:
            shape   += (<Py_ssize_t>dof,)
            strides += (<Py_ssize_t>(k*xm*ym*zm),)
        #
        self.array = asarray(vec)
        self.starts = starts
        self.sizes  = sizes
        self.array.shape = shape
        self.array.strides = strides

    def __getitem__(self, index):
        index = adjust_index_exp(self.starts, index)
        return self.array[index]

    def __setitem__(self, index, value):
        index = adjust_index_exp(self.starts, index)
        self.array[index] = value

cdef object adjust_index_exp(object starts, object index):
     if not isinstance(index, tuple):
         return adjust_index(starts[0], index)
     index = list(index)
     for i, start in enumerate(starts):
         index[i] = adjust_index(start, index[i])
     index = tuple(index)
     return index

cdef object adjust_index(object lbound, object index):
    if index is None:
        return index
    if index is Ellipsis:
        return index
    if isinstance(index, slice):
        start = index.start
        stop  = index.stop
        step  = index.step
        if start is not None: start -= lbound
        if stop  is not None: stop  -= lbound
        return slice(start, stop, step)
    try:
        return index - lbound
    except TypeError:
        return index

# --------------------------------------------------------------------
