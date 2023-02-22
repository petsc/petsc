#---------------------------------------------------------------------

cdef inline int setref(void *d, void *s) except -1:
    cdef PetscObject *dest  = <PetscObject*> d
    cdef PetscObject source = <PetscObject>  s
    CHKERR( PetscINCREF(&source) )
    dest[0] = source
    return 0

#---------------------------------------------------------------------

# -- Error --

cdef api int PyPetscError_Set(PetscErrorCode ierr):
    return SETERR(ierr)

# -- Comm --

cdef api object PyPetscComm_New(MPI_Comm arg):
    cdef Comm retv = Comm()
    retv.comm = arg
    return retv

cdef api MPI_Comm PyPetscComm_Get(object arg) except *:
    cdef MPI_Comm retv = MPI_COMM_NULL
    cdef Comm ob = <Comm?> arg
    retv = ob.comm
    return retv

cdef api MPI_Comm* PyPetscComm_GetPtr(object arg) except NULL:
    cdef MPI_Comm *retv = NULL
    cdef Comm ob = <Comm?> arg
    retv = &ob.comm
    return retv

# -- Object --

cdef api object PyPetscObject_New(PetscObject arg):
    cdef Object retv = subtype_Object(arg)()
    setref(&retv.obj[0], arg)
    return retv

cdef api PetscObject PyPetscObject_Get(object arg) except ? NULL:
    cdef PetscObject retv = NULL
    cdef Object ob = <Object?> arg
    retv = ob.obj[0]
    return retv

cdef api PetscObject* PyPetscObject_GetPtr(object arg) except NULL:
    cdef PetscObject *retv = NULL
    cdef Object ob = <Object?> arg
    retv = ob.obj
    return retv

# -- Viewer --

cdef api object PyPetscViewer_New(PetscViewer arg):
    cdef Viewer retv = Viewer()
    setref(&retv.vwr, arg)
    return retv

cdef api PetscViewer PyPetscViewer_Get(object arg) except ? NULL:
    cdef PetscViewer retv = NULL
    cdef Viewer ob = <Viewer?> arg
    retv = ob.vwr
    return retv

# -- Random --

cdef api object PyPetscRandom_New(PetscRandom arg):
    cdef Random retv = Random()
    setref(&retv.rnd, arg)
    return retv

cdef api PetscRandom PyPetscRandom_Get(object arg) except ? NULL:
    cdef PetscRandom retv = NULL
    cdef Random ob = <Random?> arg
    retv = ob.rnd
    return retv

# -- Device --

cdef api Device PyPetscDevice_New(PetscDevice arg):
    cdef Device ret = Device()
    ret.device = arg
    return ret

cdef api PetscDevice PyPetscDevice_Get(object arg) except ? NULL:
    cdef Device      obj = <Device?>arg
    cdef PetscDevice ret = obj.device
    return ret

# -- DeviceContext --

cdef api DeviceContext PyPetscDeviceContext_New(PetscDeviceContext arg):
    cdef DeviceContext retv = DeviceContext()
    setref(&retv.dctx, arg)
    return retv

cdef api PetscDeviceContext PyPetscDeviceContext_Get(object arg) except ? NULL:
    cdef DeviceContext      obj = <DeviceContext?>arg
    cdef PetscDeviceContext ret = obj.dctx
    return ret

# -- IS --

cdef api object PyPetscIS_New(PetscIS arg):
    cdef IS retv = IS()
    setref(&retv.iset, arg)
    return retv

cdef api PetscIS PyPetscIS_Get(object arg) except? NULL:
    cdef PetscIS retv = NULL
    cdef IS ob = <IS?> arg
    retv = ob.iset
    return retv

# -- LGMap --

cdef api object PyPetscLGMap_New(PetscLGMap arg):
    cdef LGMap retv = LGMap()
    setref(&retv.lgm, arg)
    return retv

cdef api PetscLGMap PyPetscLGMap_Get(object arg) except ? NULL:
    cdef PetscLGMap retv = NULL
    cdef LGMap ob = <LGMap?> arg
    retv = ob.lgm
    return retv

# -- SF --

cdef api object PyPetscSF_New(PetscSF arg):
    cdef SF retv = SF()
    setref(&retv.sf, arg)
    return retv

cdef api PetscSF PyPetscSF_Get(object arg) except? NULL:
    cdef PetscSF retv = NULL
    cdef SF ob = <SF?> arg
    retv = ob.sf
    return retv

# -- Vec --

cdef api object PyPetscVec_New(PetscVec arg):
    cdef Vec retv = Vec()
    setref(&retv.vec, arg)
    return retv

cdef api PetscVec PyPetscVec_Get(object arg) except ? NULL:
    cdef PetscVec retv = NULL
    cdef Vec ob = <Vec?> arg
    retv = ob.vec
    return retv

# -- Scatter --

cdef api object PyPetscScatter_New(PetscScatter arg):
    cdef Scatter retv = Scatter()
    setref(&retv.sct, arg)
    return retv

cdef api PetscScatter PyPetscScatter_Get(object arg) except ? NULL:
    cdef PetscScatter retv = NULL
    cdef Scatter ob = <Scatter?> arg
    retv = ob.sct
    return retv

# -- Section --

cdef api object PyPetscSection_New(PetscSection arg):
    cdef Section retv = Section()
    setref(&retv.sec, arg)
    return retv

cdef api PetscSection PyPetscSection_Get(object arg) except ? NULL:
    cdef PetscSection retv = NULL
    cdef Section ob = <Section?> arg
    retv = ob.sec
    return retv

# -- Mat --

cdef api object PyPetscMat_New(PetscMat arg):
    cdef Mat retv = Mat()
    setref(&retv.mat, arg)
    return retv

cdef api PetscMat PyPetscMat_Get(object arg) except ? NULL:
    cdef PetscMat retv = NULL
    cdef Mat ob = <Mat?> arg
    retv = ob.mat
    return retv

# -- MatPartitioning --

cdef api object PyPetscMatPartitioning_New(PetscMatPartitioning arg):
    cdef MatPartitioning retv = MatPartitioning()
    setref(&retv.part, arg)
    return retv

cdef api PetscMatPartitioning PyPetscMatPartitioning_Get(object arg) except ? NULL:
    cdef PetscMatPartitioning retv = NULL
    cdef MatPartitioning ob = <MatPartitioning?> arg
    retv = ob.part
    return retv

# -- PC --

cdef api object PyPetscPC_New(PetscPC arg):
    cdef PC retv = PC()
    setref(&retv.pc, arg)
    return retv

cdef api PetscPC PyPetscPC_Get(object arg) except ? NULL:
    cdef PetscPC retv = NULL
    cdef PC ob = <PC?> arg
    retv = ob.pc
    return retv

# -- KSP --

cdef api object PyPetscKSP_New(PetscKSP arg):
    cdef KSP retv = KSP()
    setref(&retv.ksp, arg)
    return retv

cdef api PetscKSP PyPetscKSP_Get(object arg) except ? NULL:
    cdef PetscKSP retv = NULL
    cdef KSP ob = <KSP?> arg
    retv = ob.ksp
    return retv

# -- SNES --

cdef api object PyPetscSNES_New(PetscSNES arg):
    cdef SNES retv = SNES()
    setref(&retv.snes, arg)
    return retv

cdef api PetscSNES PyPetscSNES_Get(object arg) except ? NULL:
    cdef PetscSNES retv = NULL
    cdef SNES ob = <SNES?> arg
    retv = ob.snes
    return retv

# -- TS --

cdef api object PyPetscTS_New(PetscTS arg):
    cdef TS retv = TS()
    setref(&retv.ts, arg)
    return retv

cdef api PetscTS PyPetscTS_Get(object arg) except ? NULL:
    cdef PetscTS retv = NULL
    cdef TS ob = <TS?> arg
    retv = ob.ts
    return retv

# -- TAO --

cdef api object PyPetscTAO_New(PetscTAO arg):
    cdef TAO retv = TAO()
    setref(&retv.tao, arg)
    return retv

cdef api PetscTAO PyPetscTAO_Get(object arg) except ? NULL:
    cdef PetscTAO retv = NULL
    cdef TAO ob = <TAO?> arg
    retv = ob.tao
    return retv

# -- AO --

cdef api object PyPetscAO_New(PetscAO arg):
    cdef AO retv = AO()
    setref(&retv.ao, arg)
    return retv

cdef api PetscAO PyPetscAO_Get(object arg) except ? NULL:
    cdef PetscAO retv = NULL
    cdef AO ob = <AO?> arg
    retv = ob.ao
    return retv

# -- DM --

cdef api object PyPetscDM_New(PetscDM arg):
    cdef DM retv = subtype_DM(arg)()
    setref(&retv.dm, arg)
    return retv

cdef api PetscDM PyPetscDM_Get(object arg) except ? NULL:
    cdef PetscDM retv = NULL
    cdef DM ob = <DM?> arg
    retv = ob.dm
    return retv

# -- DS --

cdef api object PyPetscDS_New(PetscDS arg):
    cdef DS retv = DS()
    setref(&retv.ds, arg)
    return retv

cdef api PetscDS PyPetscDS_Get(object arg) except ? NULL:
    cdef PetscDS retv = NULL
    cdef DS ob = <DS?> arg
    retv = ob.ds
    return retv

# -- Partitioner --

cdef api object PyPetscPartitioner_New(PetscPartitioner arg):
    cdef Partitioner retv = Partitioner()
    setref(&retv.part, arg)
    return retv

cdef api PetscPartitioner PyPetscPartitioner_Get(object arg) except ? NULL:
    cdef PetscPartitioner retv = NULL
    cdef Partitioner ob = <Partitioner?> arg
    retv = ob.part
    return retv

# -- FE --

cdef api object PyPetscFE_New(PetscFE arg):
    cdef FE retv = FE()
    setref(&retv.fe, arg)
    return retv

cdef api PetscFE PyPetscFE_Get(object arg) except ? NULL:
    cdef PetscFE retv = NULL
    cdef FE ob = <FE?> arg
    retv = ob.fe
    return retv

# -- QUAD --

cdef api object PyPetscQuad_New(PetscQuadrature arg):
    cdef Quad retv = Quad()
    setref(&retv.quad, arg)
    return retv

cdef api PetscQuadrature PyPetscQuad_Get(object arg) except ? NULL:
    cdef PetscQuadrature retv = NULL
    cdef Quad ob = <Quad?> arg
    retv = ob.quad
    return retv

# -- DMLabel --

cdef api object PyPetscDMLabel_New(PetscDMLabel arg):
    cdef DMLabel retv = DMLabel()
    setref(&retv.dmlabel, arg)
    return retv

cdef api PetscDMLabel PyPetscDMLabel_Get(object arg) except ? NULL:
    cdef PetscDMLabel retv = NULL
    cdef DMLabel ob = <DMLabel?> arg
    retv = ob.dmlabel
    return retv


# -- SPACE --

cdef api object PyPetscSpace_New(PetscSpace arg):
    cdef Space retv = Space()
    setref(&retv.space, arg)
    return retv

cdef api PetscSpace PyPetscSpace_Get(object arg) except ? NULL:
    cdef PetscSpace retv = NULL
    cdef Space ob = <Space?> arg
    retv = ob.space
    return retv

# -- DUALSPACE --

cdef api object PyPetscDualSpace_New(PetscDualSpace arg):
    cdef DualSpace retv = DualSpace()
    setref(&retv.dualspace, arg)
    return retv

cdef api PetscDualSpace PyPetscDualSpace_Get(object arg) except ? NULL:
    cdef PetscDualSpace retv = NULL
    cdef DualSpace ob = <DualSpace?> arg
    retv = ob.dualspace
    return retv


#---------------------------------------------------------------------
