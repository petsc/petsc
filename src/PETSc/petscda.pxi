# --------------------------------------------------------------------

cdef extern from * nogil:

    ctypedef enum PetscDAStencilType "DAStencilType":
        DA_STENCIL_STAR
        DA_STENCIL_BOX

    ctypedef enum PetscDABoundaryType "DABoundaryType":
        DA_NONPERIODIC
        DA_XPERIODIC
        DA_YPERIODIC
        DA_ZPERIODIC
        DA_XYPERIODIC
        DA_XZPERIODIC
        DA_YZPERIODIC
        DA_XYZPERIODIC
        DA_NONGHOSTED
        DA_XGHOSTED
        DA_YGHOSTED
        DA_ZGHOSTED
        DA_XYGHOSTED
        DA_XZGHOSTED
        DA_YZGHOSTED
        DA_XYZGHOSTED

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
                   PetscDABoundaryType,              # periodicity
                   PetscDAStencilType,               # stencil type
                   PetscInt,                         # stencil width
                   PetscDA*)

    int DASetOptionsPrefix(PetscDA,char[])
    int DASetFromOptions(PetscDA)
    int DASetInterpolationType(PetscDA,PetscDAInterpolationType)
    int DAGetInterpolation(PetscDA,PetscDA,PetscMat*,PetscVec*)
    int DAGetInjection(PetscDA,PetscDA,PetscScatter*)
    int DAGetAggregates(PetscDA,PetscDA,PetscMat*)
    int DAGetInfo(PetscDA,
                  PetscInt*,
                  PetscInt*,PetscInt*,PetscInt*,
                  PetscInt*,PetscInt*,PetscInt*,
                  PetscInt*,PetscInt*,
                  PetscDABoundaryType*,PetscDAStencilType*)

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
    int DAGetLocalToGlobalMapping(PetscDA,PetscLGMap*)
    int DAGetLocalToGlobalMappingBlock(PetscDA,PetscLGMap*)
    int DAGetScatter(PetscDA,PetscScatter*,PetscScatter*,PetscScatter*)

    int DASetRefinementFactor(PetscDA,PetscInt,PetscInt,PetscInt)
    int DAGetRefinementFactor(PetscDA,PetscInt*,PetscInt*,PetscInt*)
    int DARefine(PetscDA,MPI_Comm,PetscDA*)
    int DACoarsen(PetscDA,MPI_Comm,PetscDA*)

    int DASetElementType(PetscDA,PetscDAElementType)
    int DAGetElementType(PetscDA,PetscDAElementType*)
    int DAGetElements(PetscDA,PetscInt*,PetscInt*,const_PetscInt**)
    int DARestoreElements(PetscDA,PetscInt*,PetscInt*,const_PetscInt**)

    #int DASetFieldName(PetscDA,PetscInt,const_char[])
    #int DAGetFieldName(PetscDA,PetscInt,char**)

# --------------------------------------------------------------------

cdef enum:
    DA_PERIODIC_NONE = 0+0
    DA_PERIODIC_X    = 0+1
    DA_PERIODIC_Y    = 0+2
    DA_PERIODIC_Z    = 0+4

cdef enum:
    DA_GHOSTED_NONE  = 8+0
    DA_GHOSTED_X     = 8+1
    DA_GHOSTED_Y     = 8+2
    DA_GHOSTED_Z     = 8+4

cdef PetscDABoundaryType daboundary[8+8]
#
daboundary[DA_PERIODIC_NONE] = DA_NONPERIODIC
daboundary[DA_PERIODIC_X] = DA_XPERIODIC
daboundary[DA_PERIODIC_Y] = DA_YPERIODIC
daboundary[DA_PERIODIC_Z] = DA_ZPERIODIC
daboundary[DA_PERIODIC_X|DA_PERIODIC_Y] = DA_XYPERIODIC
daboundary[DA_PERIODIC_X|DA_PERIODIC_Z] = DA_XZPERIODIC
daboundary[DA_PERIODIC_Y|DA_PERIODIC_Z] = DA_YZPERIODIC
daboundary[DA_PERIODIC_X|DA_PERIODIC_Y|DA_PERIODIC_Z] = DA_XYZPERIODIC
#
daboundary[DA_GHOSTED_NONE] = DA_NONGHOSTED
daboundary[DA_GHOSTED_X] = DA_XGHOSTED
daboundary[DA_GHOSTED_Y] = DA_YGHOSTED
daboundary[DA_GHOSTED_Z] = DA_ZGHOSTED
daboundary[DA_GHOSTED_X|DA_GHOSTED_Y] = DA_XYGHOSTED
daboundary[DA_GHOSTED_X|DA_GHOSTED_Z] = DA_XZGHOSTED
daboundary[DA_GHOSTED_Y|DA_GHOSTED_Z] = DA_YZGHOSTED
daboundary[DA_GHOSTED_X|DA_GHOSTED_Y|DA_GHOSTED_Z] = DA_XYZGHOSTED

cdef inline PetscDABoundaryType DA_BOUNDARY(PetscInt dim,
                                            int x, int y, int z) nogil:
    cdef int flag = 0
    if dim >= 1 and x:
        if x == 1: flag |= DA_PERIODIC_X
        if x == 2: flag |= DA_GHOSTED_X
    if dim >= 2 and y:
        if y == 1: flag |= DA_PERIODIC_Y
        if y == 2: flag |= DA_GHOSTED_Y
    if dim >= 3 and z:
        if z == 1: flag |= DA_PERIODIC_Z
        if z == 2: flag |= DA_GHOSTED_Z
    return daboundary[flag]

cdef inline PetscDABoundaryType asBoundary(PetscInt dim, object boundary) \
    except <PetscDABoundaryType>(-1):
    if boundary is None:
        return DA_BOUNDARY(dim,0,0,0)
    if isinstance(boundary, str):
        if   boundary == 'periodic':
            return DA_BOUNDARY(dim,1,1,1)
        elif boundary == 'ghosted':
            return DA_BOUNDARY(dim,2,2,2)
        else:
            raise ValueError("unknown boundary type: %s" % boundary)
    if isinstance(boundary, int):
        return boundary
    cdef PetscInt pdim = len(boundary)
    cdef int x=0, y=0, z=0
    if   pdim == 1: (x,)    = boundary
    elif pdim == 2: (x,y)   = boundary
    elif pdim == 3: (x,y,z) = boundary
    return DA_BOUNDARY(dim,x,y,z)

cdef inline object toBoundary(PetscInt dim, PetscDABoundaryType ptype):
    cdef int x=0, y=0, z=0
    #
    if   ptype == DA_NONPERIODIC: pass
    elif ptype == DA_XYZPERIODIC: x = y = z = 1
    elif ptype == DA_XYPERIODIC:  x = y = 1
    elif ptype == DA_XZPERIODIC:  x = z = 1
    elif ptype == DA_YZPERIODIC:  y = z = 1
    elif ptype == DA_XPERIODIC:   x = 1
    elif ptype == DA_YPERIODIC:   y = 1
    elif ptype == DA_ZPERIODIC:   z = 1
    #
    if   ptype == DA_NONGHOSTED: pass
    elif ptype == DA_XYZGHOSTED: x = y = z = 2
    elif ptype == DA_XYGHOSTED:  x = y = 2
    elif ptype == DA_XZGHOSTED:  x = z = 2
    elif ptype == DA_YZGHOSTED:  y = z = 2
    elif ptype == DA_XGHOSTED:   x = 2
    elif ptype == DA_YGHOSTED:   y = 2
    elif ptype == DA_ZGHOSTED:   z = 2
    #
    if   dim == 1: return (x,)
    elif dim == 2: return (x,y)
    elif dim == 3: return (x,y,z)

cdef inline PetscDAStencilType asStencil(object stencil) \
    except <PetscDAStencilType>(-1):
    if stencil is None:
        return DA_STENCIL_BOX
    if isinstance(stencil, str):
        if   stencil == "star": return DA_STENCIL_STAR
        elif stencil == "box":  return DA_STENCIL_BOX
        else: raise ValueError("unknown stencil type: %s" % stencil)
    return stencil

cdef inline object toStencil(PetscDAStencilType stype):
    if   stype == DA_STENCIL_STAR: return "star"
    elif stype == DA_STENCIL_BOX:  return "box"

cdef inline PetscDAInterpolationType dainterpolationtype(object itype) \
    except <PetscDAInterpolationType>(-1):
    if (isinstance(itype, str)):
        if itype in ("q0", "Q0"): return DA_INTERPOLATION_Q0
        if itype in ("q1", "Q1"): return DA_INTERPOLATION_Q1
        else: raise ValueError("unknown interpolation type: %s" % itype)
    return itype

cdef inline PetscDAElementType daelementtype(object etype) \
    except <PetscDAElementType>(-1):
    if (isinstance(etype, str)):
        if etype in ("p1", "P1"): return DA_ELEMENT_P1
        if etype in ("q1", "Q1"): return DA_ELEMENT_Q1
        else: raise ValueError("unknown element type: %s" % etype)
    return etype

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
    if   ndim == 1: M, = dims
    elif ndim == 2: M, N = dims
    elif ndim == 3: M, N, P = dims
    if ndim >= 1: _M[0] = asInt(M)
    if ndim >= 2: _N[0] = asInt(N)
    if ndim >= 3: _P[0] = asInt(P)
    return ndim

cdef inline tuple toDims(PetscInt dim,
                         PetscInt M,
                         PetscInt N,
                         PetscInt P):
        if   dim == 1: return (toInt(M),)
        elif dim == 2: return (toInt(M), toInt(N))
        elif dim == 3: return (toInt(M), toInt(N), toInt(P))

# --------------------------------------------------------------------

cdef class _DA_Vec_array(object):

    cdef _Vec_buffer vecbuf
    cdef readonly tuple starts, sizes
    cdef tuple shape, strides
    cdef readonly ndarray array

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
        self.vecbuf = _Vec_buffer(vec)
        self.starts = starts
        self.sizes = sizes
        self.shape = shape
        self.strides = strides

    cdef int acquire(self) except -1:
        self.vecbuf.acquire()
        if self.array is None:
            self.array = asarray(self.vecbuf)
            self.array.shape = self.shape
            self.array.strides = self.strides
        return 0

    cdef int release(self) except -1:
        self.vecbuf.release()
        self.array = None
        return 0

    #

    def __getitem__(self, index):
        self.acquire()
        index = adjust_index_exp(self.starts, index)
        return self.array[index]

    def __setitem__(self, index, value):
        self.acquire()
        index = adjust_index_exp(self.starts, index)
        self.array[index] = value

    # 'with' statement (PEP 343)

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, t, v, tb):
        self.release()
        return None


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
