# --------------------------------------------------------------------

cdef extern from * nogil:

    ctypedef enum PetscDMDAStencilType"DMDAStencilType":
        DMDA_STENCIL_STAR
        DMDA_STENCIL_BOX

    ctypedef enum PetscDMDAInterpolationType"DMDAInterpolationType":
        DMDA_INTERPOLATION_Q0 "DMDA_Q0"
        DMDA_INTERPOLATION_Q1 "DMDA_Q1"

    ctypedef enum PetscDMDAElementType"DMDAElementType":
        DMDA_ELEMENT_P1
        DMDA_ELEMENT_Q1

    PetscErrorCode DMDACreateND(MPI_Comm,
                     PetscInt,PetscInt,                # dim, dof
                     PetscInt,PetscInt,PetscInt,       # M, N, P
                     PetscInt,PetscInt,PetscInt,       # m, n, p
                     PetscInt[],PetscInt[],PetscInt[], # lx, ly, lz
                     PetscDMBoundaryType,              # bx
                     PetscDMBoundaryType,              # by
                     PetscDMBoundaryType,              # bz
                     PetscDMDAStencilType,             # stencil type
                     PetscInt,                         # stencil width
                     PetscDM*)
    
    PetscErrorCode DMDASetDof(PetscDM,PetscInt)
    PetscErrorCode DMDASetSizes(PetscDM,PetscInt,PetscInt,PetscInt)
    PetscErrorCode DMDASetNumProcs(PetscDM,PetscInt,PetscInt,PetscInt)
    PetscErrorCode DMDASetBoundaryType(PetscDM,PetscDMBoundaryType,PetscDMBoundaryType,PetscDMBoundaryType)
    PetscErrorCode DMDASetStencilType(PetscDM,PetscDMDAStencilType)
    PetscErrorCode DMDASetStencilWidth(PetscDM,PetscInt)

    PetscErrorCode DMDAGetInfo(PetscDM,
                    PetscInt*,
                    PetscInt*,PetscInt*,PetscInt*,
                    PetscInt*,PetscInt*,PetscInt*,
                    PetscInt*,PetscInt*,
                    PetscDMBoundaryType*,
                    PetscDMBoundaryType*,
                    PetscDMBoundaryType*,
                    PetscDMDAStencilType*)
    PetscErrorCode DMDAGetCorners(PetscDM,
                       PetscInt*,PetscInt*,PetscInt*,
                       PetscInt*,PetscInt*,PetscInt*)
    PetscErrorCode DMDAGetGhostCorners(PetscDM,
                            PetscInt*,PetscInt*,PetscInt*,
                            PetscInt*,PetscInt*,PetscInt*)
    PetscErrorCode DMDAGetOwnershipRanges(PetscDM,
                               const PetscInt*[],
                               const PetscInt*[],
                               const PetscInt*[])

    PetscErrorCode DMDASetUniformCoordinates(PetscDM,
                                  PetscReal,PetscReal,
                                  PetscReal,PetscReal,
                                  PetscReal,PetscReal)
    PetscErrorCode DMGetBoundingBox(PetscDM,PetscReal[],PetscReal[])
    PetscErrorCode DMGetLocalBoundingBox(PetscDM,PetscReal[],PetscReal[])

    PetscErrorCode DMDACreateNaturalVector(PetscDM,PetscVec*)
    PetscErrorCode DMDAGlobalToNaturalBegin(PetscDM,PetscVec,PetscInsertMode,PetscVec)
    PetscErrorCode DMDAGlobalToNaturalEnd(PetscDM,PetscVec,PetscInsertMode,PetscVec)
    PetscErrorCode DMDANaturalToGlobalBegin(PetscDM,PetscVec,PetscInsertMode,PetscVec)
    PetscErrorCode DMDANaturalToGlobalEnd(PetscDM,PetscVec,PetscInsertMode,PetscVec)

    PetscErrorCode DMDAGetAO(PetscDM,PetscAO*)
    PetscErrorCode DMDAGetScatter(PetscDM,PetscScatter*,PetscScatter*)

    PetscErrorCode DMDASetRefinementFactor(PetscDM,PetscInt,PetscInt,PetscInt)
    PetscErrorCode DMDAGetRefinementFactor(PetscDM,PetscInt*,PetscInt*,PetscInt*)
    PetscErrorCode DMDASetInterpolationType(PetscDM,PetscDMDAInterpolationType)
    PetscErrorCode DMDAGetInterpolationType(PetscDM,PetscDMDAInterpolationType*)
    PetscErrorCode DMDASetElementType(PetscDM,PetscDMDAElementType)
    PetscErrorCode DMDAGetElementType(PetscDM,PetscDMDAElementType*)
    PetscErrorCode DMDAGetElements(PetscDM,PetscInt*,PetscInt*,const PetscInt**)
    PetscErrorCode DMDARestoreElements(PetscDM,PetscInt*,PetscInt*,const PetscInt**)

    PetscErrorCode DMDASetFieldName(PetscDM,PetscInt,const char[])
    PetscErrorCode DMDAGetFieldName(PetscDM,PetscInt,const char*[])
    PetscErrorCode DMDASetCoordinateName(PetscDM,PetscInt,const char[])
    PetscErrorCode DMDAGetCoordinateName(PetscDM,PetscInt,const char*[])

# --------------------------------------------------------------------

cdef inline PetscDMDAStencilType asStencil(object stencil) \
    except <PetscDMDAStencilType>(-1):
    if isinstance(stencil, str):
        if   stencil == "star": return DMDA_STENCIL_STAR
        elif stencil == "box":  return DMDA_STENCIL_BOX
        else: raise ValueError("unknown stencil type: %s" % stencil)
    return stencil

cdef inline object toStencil(PetscDMDAStencilType stype):
    if   stype == DMDA_STENCIL_STAR: return "star"
    elif stype == DMDA_STENCIL_BOX:  return "box"

cdef inline PetscDMDAInterpolationType dainterpolationtype(object itype) \
    except <PetscDMDAInterpolationType>(-1):
    if (isinstance(itype, str)):
        if itype in ("q0", "Q0"): return DMDA_INTERPOLATION_Q0
        if itype in ("q1", "Q1"): return DMDA_INTERPOLATION_Q1
        else: raise ValueError("unknown interpolation type: %s" % itype)
    return itype

cdef inline PetscDMDAElementType daelementtype(object etype) \
    except <PetscDMDAElementType>(-1):
    if (isinstance(etype, str)):
        if etype in ("p1", "P1"): return DMDA_ELEMENT_P1
        if etype in ("q1", "Q1"): return DMDA_ELEMENT_Q1
        else: raise ValueError("unknown element type: %s" % etype)
    return etype

cdef inline PetscErrorCode DMDAGetDim(PetscDM da, PetscInt *dim) nogil:
     return DMDAGetInfo(da, dim,
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
    cdef object M=None, N=None, P=None
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

cdef inline tuple asOwnershipRanges(object ownership_ranges,
                                    PetscInt dim,
                                    PetscInt *m, PetscInt *n, PetscInt *p,
                                    PetscInt **_x,
                                    PetscInt **_y,
                                    PetscInt **_z):
    cdef object ranges = list(ownership_ranges)
    cdef PetscInt rdim = <PetscInt>len(ranges)
    cdef PetscInt nlx=0, nly=0, nlz=0
    if dim == PETSC_DECIDE: dim = rdim
    elif dim != rdim: raise ValueError(
        "number of dimensions %d and number ownership ranges %d" %
        (toInt(dim), toInt(rdim)))
    if dim >= 1: 
        ranges[0] = iarray_i(ranges[0], &nlx, _x)
        if m[0] == PETSC_DECIDE: m[0] = nlx
        elif m[0] != nlx: raise ValueError(
            "ownership range size %d and number or processors %d" %
            (toInt(nlx), toInt(m[0])))
    if dim >= 2:
        ranges[1] = iarray_i(ranges[1], &nly, _y)
        if n[0] == PETSC_DECIDE: n[0] = nly
        elif n[0] != nly: raise ValueError(
            "ownership range size %d and number or processors %d" %
            (toInt(nly), toInt(n[0])))
    if dim >= 3:
        ranges[2] = iarray_i(ranges[2], &nlz, _z)
        if p[0] == PETSC_DECIDE: p[0] = nlz
        elif p[0] != nlz: raise ValueError(
            "ownership range size %d and number or processors %d" %
             (toInt(nlz), toInt(p[0])))
    return tuple(ranges)

cdef inline tuple toOwnershipRanges(PetscInt dim,
                                    PetscInt m, PetscInt n, PetscInt p,
                                    const PetscInt *lx,
                                    const PetscInt *ly,
                                    const PetscInt *lz):
    # Returns tuple of arrays containing ownership ranges as Python arrays
    ranges = [array_i(m, lx)]
    if dim > 1:
        ranges.append(array_i(n, ly))
    if dim > 2:
        ranges.append(array_i(p, lz))
    return tuple(ranges)

# --------------------------------------------------------------------

cdef class _DMDA_Vec_array(object):

    cdef _Vec_buffer vecbuf
    cdef readonly tuple starts, sizes
    cdef readonly tuple shape, strides
    cdef readonly ndarray array

    def __cinit__(self, DMDA da, Vec vec, bint DOF=False):
        #
        cdef PetscInt dim=0, dof=0
        CHKERR( DMDAGetInfo(da.dm,
                            &dim, NULL, NULL, NULL, NULL, NULL, NULL,
                            &dof, NULL, NULL, NULL, NULL, NULL) )
        cdef PetscInt lxs=0, lys=0, lzs=0
        cdef PetscInt lxm=0, lym=0, lzm=0
        CHKERR( DMDAGetCorners(da.dm,
                               &lxs, &lys, &lzs,
                               &lxm, &lym, &lzm) )
        cdef PetscInt gxs=0, gys=0, gzs=0
        cdef PetscInt gxm=0, gym=0, gzm=0
        CHKERR( DMDAGetGhostCorners(da.dm,
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
                "Vector local size %d is not compatible "
                "with DMDA local sizes %s"
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
