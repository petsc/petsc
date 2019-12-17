# --------------------------------------------------------------------

cdef extern from * nogil:

    ctypedef enum PetscDMStagStencilType"DMStagStencilType":
        DMSTAG_STENCIL_STAR
        DMSTAG_STENCIL_BOX
        DMSTAG_STENCIL_NONE

    ctypedef enum PetscDMStagStencilLocation"DMStagStencilLocation":
        DMSTAG_NULL_LOCATION
        DMSTAG_BACK_DOWN_LEFT
        DMSTAG_BACK_DOWN
        DMSTAG_BACK_DOWN_RIGHT
        DMSTAG_BACK_LEFT
        DMSTAG_BACK
        DMSTAG_BACK_RIGHT
        DMSTAG_BACK_UP_LEFT
        DMSTAG_BACK_UP
        DMSTAG_BACK_UP_RIGHT
        DMSTAG_DOWN_LEFT
        DMSTAG_DOWN
        DMSTAG_DOWN_RIGHT
        DMSTAG_LEFT
        DMSTAG_ELEMENT
        DMSTAG_RIGHT
        DMSTAG_UP_LEFT
        DMSTAG_UP
        DMSTAG_UP_RIGHT
        DMSTAG_FRONT_DOWN_LEFT
        DMSTAG_FRONT_DOWN
        DMSTAG_FRONT_DOWN_RIGHT
        DMSTAG_FRONT_LEFT
        DMSTAG_FRONT
        DMSTAG_FRONT_RIGHT
        DMSTAG_FRONT_UP_LEFT
        DMSTAG_FRONT_UP
        DMSTAG_FRONT_UP_RIGHT


    int DMStagCreate1d(MPI_Comm,PetscDMBoundaryType,PetscInt,PetscInt,PetscInt,PetscDMStagStencilType,PetscInt,const_PetscInt[],PetscDM*)
    int DMStagCreate2d(MPI_Comm,PetscDMBoundaryType,PetscDMBoundaryType,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscDMStagStencilType,PetscInt,const_PetscInt[],const_PetscInt[],PetscDM*)
    int DMStagCreate3d(MPI_Comm,PetscDMBoundaryType,PetscDMBoundaryType,PetscDMBoundaryType,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscDMStagStencilType,PetscInt,const_PetscInt[],const_PetscInt[],const_PetscInt[],PetscDM*)


    int DMStagGetCorners(PetscDM,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*)
    int DMStagGetGhostCorners(PetscDM,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*)
    int DMStagGetLocalSizes(PetscDM,PetscInt*,PetscInt*,PetscInt*)
    int DMStagGetEntriesPerElement(PetscDM,PetscInt*)

    int DMStagGetDOF(PetscDM,PetscInt*,PetscInt*,PetscInt*,PetscInt*)
    int DMStagGetNumRanks(PetscDM,PetscInt*,PetscInt*,PetscInt*)
    int DMStagGetGlobalSizes(PetscDM,PetscInt*,PetscInt*,PetscInt*)
    int DMStagGetBoundaryTypes(PetscDM,PetscDMBoundaryType*,PetscDMBoundaryType*,PetscDMBoundaryType*)
    int DMStagGetStencilWidth(PetscDM,PetscInt*)
    int DMStagGetStencilType(PetscDM,PetscDMStagStencilType*)
    int DMStagGetOwnershipRanges(PetscDM,const_PetscInt*[],const_PetscInt*[],const_PetscInt*[])

    int DMStagSetDOF(PetscDM,PetscInt,PetscInt,PetscInt,PetscInt)
    int DMStagSetNumRanks(PetscDM,PetscInt,PetscInt,PetscInt)    
    int DMStagSetGlobalSizes(PetscDM,PetscInt,PetscInt,PetscInt)
    int DMStagSetBoundaryTypes(PetscDM,PetscDMBoundaryType,PetscDMBoundaryType,PetscDMBoundaryType)
    int DMStagSetStencilWidth(PetscDM,PetscInt)
    int DMStagSetStencilType(PetscDM,PetscDMStagStencilType)
    int DMStagSetOwnershipRanges(PetscDM,const_PetscInt[],const_PetscInt[],const_PetscInt[])

    int DMStagGetLocationSlot(PetscDM,PetscDMStagStencilLocation,PetscInt,PetscInt*)
    int DMStagGetLocationDOF(PetscDM,PetscDMStagStencilLocation,PetscInt*)
    int DMStagGetProductCoordinateLocationSlot(PetscDM,PetscDMStagStencilLocation,PetscInt*)

    int DMStagGetIsFirstRank(PetscDM,PetscBool*,PetscBool*,PetscBool*)
    int DMStagGetIsLastRank(PetscDM,PetscBool*,PetscBool*,PetscBool*)
    
    int DMStagSetUniformCoordinatesExplicit(PetscDM,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal)
    int DMStagSetUniformCoordinatesProduct(PetscDM,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal)
    int DMStagSetCoordinateDMType(PetscDM,PetscDMType)
    int DMStagSetUniformCoordinates(PetscDM,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal)
        
    int DMStagCreateCompatibleDMStag(PetscDM,PetscInt,PetscInt,PetscInt,PetscInt,PetscDM*)
    int DMStagVecSplitToDMDA(PetscDM,PetscVec,PetscDMStagStencilLocation,PetscInt,PetscDM*,PetscVec*)
    int DMStagMigrateVec(PetscDM,PetscVec,PetscDM,PetscVec)

# --------------------------------------------------------------------

cdef inline PetscDMStagStencilType asStagStencil(object stencil) \
    except <PetscDMStagStencilType>(-1):
    if isinstance(stencil, str):
        if   stencil == "star":  return DMSTAG_STENCIL_STAR
        elif stencil == "box":   return DMSTAG_STENCIL_BOX
        elif stencil == "none":  return DMSTAG_STENCIL_NONE
        else: raise ValueError("unknown stencil type: %s" % stencil)
    return stencil

cdef inline object toStagStencil(PetscDMStagStencilType stype):
    if   stype == DMSTAG_STENCIL_STAR:  return "star"
    elif stype == DMSTAG_STENCIL_BOX:   return "box"
    elif stype == DMSTAG_STENCIL_NONE:  return "none"

cdef inline PetscDMStagStencilLocation asStagStencilLocation(object stencil_location) \
    except <PetscDMStagStencilLocation>(-1):
    if isinstance(stencil_location, str):
        if   stencil_location == "null":                return DMSTAG_NULL_LOCATION
        elif stencil_location == "back_down_left":      return DMSTAG_BACK_DOWN_LEFT
        elif stencil_location == "back_down":           return DMSTAG_BACK_DOWN
        elif stencil_location == "back_down_right":     return DMSTAG_BACK_DOWN_RIGHT
        elif stencil_location == "back_left":           return DMSTAG_BACK_LEFT
        elif stencil_location == "back":                return DMSTAG_BACK
        elif stencil_location == "back_right":          return DMSTAG_BACK_RIGHT
        elif stencil_location == "back_up_left":        return DMSTAG_BACK_UP_LEFT
        elif stencil_location == "back_up":             return DMSTAG_BACK_UP
        elif stencil_location == "back_up_right":       return DMSTAG_BACK_UP_RIGHT
        elif stencil_location == "down_left":           return DMSTAG_DOWN_LEFT
        elif stencil_location == "down":                return DMSTAG_DOWN
        elif stencil_location == "down_right":          return DMSTAG_DOWN_RIGHT
        elif stencil_location == "left":                return DMSTAG_LEFT
        elif stencil_location == "element":             return DMSTAG_ELEMENT
        elif stencil_location == "right":               return DMSTAG_RIGHT
        elif stencil_location == "up_left":             return DMSTAG_UP_LEFT
        elif stencil_location == "up":                  return DMSTAG_UP
        elif stencil_location == "up_right":            return DMSTAG_UP_RIGHT
        elif stencil_location == "front_down_left":     return DMSTAG_FRONT_DOWN_LEFT
        elif stencil_location == "front_down":          return DMSTAG_FRONT_DOWN
        elif stencil_location == "front_down_right":    return DMSTAG_FRONT_DOWN_RIGHT
        elif stencil_location == "front_left":          return DMSTAG_FRONT_LEFT
        elif stencil_location == "front":               return DMSTAG_FRONT
        elif stencil_location == "front_right":         return DMSTAG_FRONT_RIGHT
        elif stencil_location == "front_up_left":       return DMSTAG_FRONT_UP_LEFT
        elif stencil_location == "front_up":            return DMSTAG_FRONT_UP
        elif stencil_location == "front_up_right":      return DMSTAG_FRONT_UP_RIGHT
        else: raise ValueError("unknown stencil location type: %s" % stencil_location)
    return stencil_location


cdef inline PetscInt asStagDims(dims,
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

cdef inline tuple toStagDims(PetscInt dim,
                         PetscInt M,
                         PetscInt N,
                         PetscInt P):
    if   dim == 0: return ()
    elif dim == 1: return (toInt(M),)
    elif dim == 2: return (toInt(M), toInt(N))
    elif dim == 3: return (toInt(M), toInt(N), toInt(P))

cdef inline PetscInt asDofs(dofs,
                            PetscInt *_dof0,
                            PetscInt *_dof1,
                            PetscInt *_dof2,
                            PetscInt *_dof3) except? -1:
    cdef PetscInt ndofs = PETSC_DECIDE
    cdef object dof0=None, dof1=None, dof2=None, dof3=None
    dofs = tuple(dofs)
    ndofs = <PetscInt>len(dofs)
    if ndofs == 2:   dof0, dof1 = dofs
    elif ndofs == 3: dof0, dof1, dof2 = dofs
    elif ndofs == 4: dof0, dof1, dof2, dof3 = dofs
    if ndofs >= 2: _dof0[0] = asInt(dof0)
    if ndofs >= 2: _dof1[0] = asInt(dof1)
    if ndofs >= 3: _dof2[0] = asInt(dof2)
    if ndofs >= 4: _dof3[0] = asInt(dof3)
    return ndofs
    
cdef inline tuple toDofs(PetscInt ndofs,
                         PetscInt dof0,
                         PetscInt dof1,
                         PetscInt dof2,
                         PetscInt dof3):
    if ndofs == 2: return (toInt(dof0), toInt(dof1))
    elif ndofs == 3: return (toInt(dof0), toInt(dof1), toInt(dof2))
    elif ndofs == 4: return (toInt(dof0), toInt(dof1), toInt(dof2), toInt(dof3))

cdef inline tuple asStagOwnershipRanges(object ownership_ranges,
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


cdef inline tuple toStagOwnershipRanges(PetscInt dim,
                                    PetscInt m, PetscInt n, PetscInt p,
                                    const_PetscInt *lx,
                                    const_PetscInt *ly,
                                    const_PetscInt *lz):
    # Returns tuple of arrays containing ownership ranges as Python arrays
    ranges = [array_i(m, lx)]
    if dim > 1:
        ranges.append(array_i(n, ly))
    if dim > 2:
        ranges.append(array_i(p, lz))
    return tuple(ranges)

cdef inline object toStagBoundary(PetscDMBoundaryType btype):
    if   btype == DM_BOUNDARY_NONE:       return "none"
    elif btype == DM_BOUNDARY_PERIODIC:   return "periodic"
    elif btype == DM_BOUNDARY_GHOSTED:    return "ghosted"
    
cdef inline tuple toStagBoundaryTypes(PetscInt dim, PetscDMBoundaryType btx, PetscDMBoundaryType bty, PetscDMBoundaryType btz):
    if dim == 1: return (toStagBoundary(btx), )
    if dim == 2: return (toStagBoundary(btx), toStagBoundary(bty))
    if dim == 3: return (toStagBoundary(btx), toStagBoundary(bty), toStagBoundary(btz))

# --------------------------------------------------------------------
