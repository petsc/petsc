# --------------------------------------------------------------------

cdef extern from * nogil:

    ctypedef PetscDM PetscDA "DM"

    ctypedef enum PetscDABoundaryType "DMDABoundaryType":
        DA_BOUNDARY_NONE     "DMDA_BOUNDARY_NONE"
        DA_BOUNDARY_GHOSTED  "DMDA_BOUNDARY_GHOSTED"
        DA_BOUNDARY_MIRROR   "DMDA_BOUNDARY_MIRROR"
        DA_BOUNDARY_PERIODIC "DMDA_BOUNDARY_PERIODIC"

    ctypedef enum PetscDAStencilType "DMDAStencilType":
        DA_STENCIL_STAR "DMDA_STENCIL_STAR"
        DA_STENCIL_BOX  "DMDA_STENCIL_BOX"

    ctypedef enum PetscDAInterpolationType "DMDAInterpolationType":
        DA_INTERPOLATION_Q0 "DMDA_Q0"
        DA_INTERPOLATION_Q1 "DMDA_Q1"

    ctypedef enum PetscDAElementType "DMDAElementType":
        DA_ELEMENT_P1 "DMDA_ELEMENT_P1"
        DA_ELEMENT_Q1 "DMDA_ELEMENT_Q1"

    int DACreateND(MPI_Comm,
                   PetscInt,PetscInt,                # dim, dof
                   PetscInt,PetscInt,PetscInt,       # M, N, P
                   PetscInt,PetscInt,PetscInt,       # m, n, p
                   PetscInt[],PetscInt[],PetscInt[], # lx, ly, lz
                   PetscDABoundaryType,              # bx
                   PetscDABoundaryType,              # by
                   PetscDABoundaryType,              # bz
                   PetscDAStencilType,               # stencil type
                   PetscInt,                         # stencil width
                   PetscDA*)
    
    int DASetDim"DMDASetDim"(PetscDA,PetscInt)
    int DASetDof"DMDASetDof"(PetscDA,PetscInt)
    int DASetSizes"DMDASetSizes"(PetscDA,PetscInt,PetscInt,PetscInt)
    int DASetNumProcs"DMDASetNumProcs"(PetscDA,PetscInt,PetscInt,PetscInt)
    int DASetBoundaryType"DMDASetBoundaryType"(PetscDA,PetscDABoundaryType,PetscDABoundaryType,PetscDABoundaryType)
    int DASetStencilType"DMDASetStencilType"(PetscDA,PetscDAStencilType)
    int DASetStencilWidth"DMDASetStencilWidth"(PetscDA,PetscInt)
    int DASetUp"DMSetUp"(PetscDA)

    int DAGetInfo"DMDAGetInfo"(
                  PetscDA,
                  PetscInt*,
                  PetscInt*,PetscInt*,PetscInt*,
                  PetscInt*,PetscInt*,PetscInt*,
                  PetscInt*,PetscInt*,
                  PetscDABoundaryType*,
                  PetscDABoundaryType*,
                  PetscDABoundaryType*,
                  PetscDAStencilType*)
    int DAGetCorners"DMDAGetCorners"(
                     PetscDA,
                     PetscInt*,PetscInt*,PetscInt*,
                     PetscInt*,PetscInt*,PetscInt*)
    int DAGetGhostCorners"DMDAGetGhostCorners"(
                          PetscDA,
                          PetscInt*,PetscInt*,PetscInt*,
                          PetscInt*,PetscInt*,PetscInt*)
    int DAGetOwnershipRanges"DMDAGetOwnershipRanges"(
                             PetscDA,
                             const_PetscInt*[],
                             const_PetscInt*[],
                             const_PetscInt*[])

    int DASetUniformCoordinates"DMDASetUniformCoordinates"(
                                PetscDA,
                                PetscReal,PetscReal,
                                PetscReal,PetscReal,
                                PetscReal,PetscReal)
    int DASetCoordinates"DMDASetCoordinates"(PetscDA,PetscVec)
    int DAGetCoordinates"DMDAGetCoordinates"(PetscDA,PetscVec*)
    int DAGetCoordinateDA"DMDAGetCoordinateDA"(PetscDA,PetscDA*)
    int DAGetGhostedCoordinates"DMDAGetGhostedCoordinates"(PetscDA,PetscVec*)
    int DAGetBoundingBox"DMDAGetBoundingBox"(PetscDM,PetscReal[],PetscReal[])
    int DAGetLocalBoundingBox"DMDAGetLocalBoundingBox"(PetscDM,PetscReal[],PetscReal[])

    int DACreateNaturalVector"DMDACreateNaturalVector"(PetscDA,PetscVec*)
    int DAGlobalToNaturalBegin"DMDAGlobalToNaturalBegin"(PetscDA,PetscVec,PetscInsertMode,PetscVec)
    int DAGlobalToNaturalEnd"DMDAGlobalToNaturalEnd"(PetscDA,PetscVec,PetscInsertMode,PetscVec)
    int DANaturalToGlobalBegin"DMDANaturalToGlobalBegin"(PetscDA,PetscVec,PetscInsertMode,PetscVec)
    int DANaturalToGlobalEnd"DMDANaturalToGlobalEnd"(PetscDA,PetscVec,PetscInsertMode,PetscVec)
    int DALocalToLocalBegin"DMDALocalToLocalBegin"(PetscDA,PetscVec,PetscInsertMode,PetscVec)
    int DALocalToLocalEnd"DMDALocalToLocalEnd"(PetscDA,PetscVec,PetscInsertMode,PetscVec)

    int DAGetAO"DMDAGetAO"(PetscDA,PetscAO*)
    int DAGetScatter"DMDAGetScatter"(PetscDA,PetscScatter*,PetscScatter*,PetscScatter*)

    int DASetRefinementFactor"DMDASetRefinementFactor"(PetscDA,PetscInt,PetscInt,PetscInt)
    int DAGetRefinementFactor"DMDAGetRefinementFactor"(PetscDA,PetscInt*,PetscInt*,PetscInt*)
    int DASetInterpolationType"DMDASetInterpolationType"(PetscDA,PetscDAInterpolationType)
    int DAGetInterpolationType"DMDAGetInterpolationType"(PetscDA,PetscDAInterpolationType*)
    int DASetElementType"DMDASetElementType"(PetscDA,PetscDAElementType)
    int DAGetElementType"DMDAGetElementType"(PetscDA,PetscDAElementType*)
    int DAGetElements"DMDAGetElements"(PetscDA,PetscInt*,PetscInt*,const_PetscInt**)
    int DARestoreElements"DMDARestoreElements"(PetscDA,PetscInt*,PetscInt*,const_PetscInt**)

    #int DASetFieldName"DMDASetFieldName"(PetscDA,PetscInt,const_char[])
    #int DAGetFieldName"DMDAGetFieldName"(PetscDA,PetscInt,const_char*[])

# --------------------------------------------------------------------

cdef inline PetscDABoundaryType asBoundaryType(object boundary) \
    except <PetscDABoundaryType>(-1):
    if boundary is None:
        return DA_BOUNDARY_NONE
    if isinstance(boundary, str):
        if boundary == 'none':
            return DA_BOUNDARY_NONE
        elif boundary == 'ghosted':
            return DA_BOUNDARY_GHOSTED
        elif boundary == 'mirror':
            return DA_BOUNDARY_MIRROR
        elif boundary == 'periodic':
            return DA_BOUNDARY_PERIODIC
        else:
            raise ValueError("unknown boundary type: %s" % boundary)
    return boundary

cdef inline PetscInt asBoundary(object boundary,
                                PetscDABoundaryType *_x,
                                PetscDABoundaryType *_y,
                                PetscDABoundaryType *_z) except? -1:
    cdef PetscInt dim = PETSC_DECIDE
    cdef object x, y, z
    if (boundary is None or
        isinstance(boundary, str) or
        isinstance(boundary, int)):
        _x[0] = _y[0] = _z[0] = asBoundaryType(boundary)
    else:
        boundary = tuple(boundary)
        dim = <PetscInt>len(boundary)
        if   dim == 0: pass
        elif dim == 1: (x,) = boundary
        elif dim == 2: (x, y) = boundary
        elif dim == 3: (x, y, z) = boundary
        if dim >= 1: _x[0] = asBoundaryType(x)
        if dim >= 2: _y[0] = asBoundaryType(y)
        if dim >= 3: _z[0] = asBoundaryType(z)
    return dim

cdef inline object toBoundary(PetscInt dim,
                              PetscDABoundaryType x,
                              PetscDABoundaryType y,
                              PetscDABoundaryType z):
    if   dim == 0: return ()
    elif dim == 1: return (x,)
    elif dim == 2: return (x, y)
    elif dim == 3: return (x, y, z)

cdef inline PetscDAStencilType asStencil(object stencil) \
    except <PetscDAStencilType>(-1):
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
                      NULL, NULL, NULL,
                      NULL)

cdef inline PetscInt asDims(dims,
                            PetscInt *_M,
                            PetscInt *_N,
                            PetscInt *_P) except? -1:
    cdef PetscInt dim = PETSC_DECIDE
    cdef object M, N, P
    dims = tuple(dims)
    dim = <PetscInt>len(dims)
    if   dim == 0: pass
    elif dim == 1: M, = dims
    elif dim == 2: M, N = dims
    elif dim == 3: M, N, P = dims
    if dim >= 1: _M[0] = asInt(M)
    if dim >= 2: _N[0] = asInt(N)
    if dim >= 3: _P[0] = asInt(P)
    return dim

cdef inline tuple toDims(PetscInt dim,
                         PetscInt M,
                         PetscInt N,
                         PetscInt P):
        if   dim == 0: return ()
        elif dim == 1: return (toInt(M),)
        elif dim == 2: return (toInt(M), toInt(N))
        elif dim == 3: return (toInt(M), toInt(N), toInt(P))

# --------------------------------------------------------------------

cdef class _DA_Vec_array(object):

    cdef _Vec_buffer vecbuf
    cdef readonly tuple starts, sizes
    cdef readonly tuple shape, strides
    cdef readonly ndarray array

    def __cinit__(self, DA da not None, Vec vec not None, bint DOF=False):
        #
        cdef PetscInt dim=0, dof=0
        CHKERR( DAGetInfo(da.dm,
                          &dim, NULL, NULL, NULL, NULL, NULL, NULL,
                          &dof, NULL, NULL, NULL, NULL, NULL) )
        cdef PetscInt lxs=0, lys=0, lzs=0
        cdef PetscInt lxm=0, lym=0, lzm=0
        CHKERR( DAGetCorners(da.dm,
                             &lxs, &lys, &lzs,
                             &lxm, &lym, &lzm) )
        cdef PetscInt gxs=0, gys=0, gzs=0
        cdef PetscInt gxm=0, gym=0, gzm=0
        CHKERR( DAGetGhostCorners(da.dm,
                                  &gxs, &gys, &gzs,
                                  &gxm, &gym, &gzm) )
        #
        cdef PetscInt n=0
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
        cdef Py_ssize_t k = <Py_ssize_t>sizeof(PetscScalar)
        cdef Py_ssize_t f = <Py_ssize_t>dof
        cdef Py_ssize_t d = <Py_ssize_t>dim
        cdef tuple shape   = toDims(dim, xm, ym, zm)
        cdef tuple strides = (k*f, k*f*xm, k*f*xm*ym)[:d]
        if DOF or f > 1: shape   += (f,)
        if DOF or f > 1: strides += (k,)
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

    def __exit__(self, *exc):
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
