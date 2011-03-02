#---------------------------------------------------------------------

cdef extern from * nogil:
    int PyPetscINCREF"PetscObjectReference"(PetscObject)

cdef int addref(void *o) except -1:
    if o == NULL: return 0
    CHKERR( PyPetscINCREF(<PetscObject>o) )
    return 0

#---------------------------------------------------------------------

# -- Error --

cdef api int PyPetscError_Set(int ierr):
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
    addref(arg); retv.vwr = arg
    return retv

cdef api PetscViewer PyPetscViewer_Get(object arg) except ? NULL:
    cdef PetscViewer retv = NULL
    cdef Viewer ob = <Viewer?> arg
    retv = ob.vwr
    return retv

# -- Random --

cdef api object PyPetscRandom_New(PetscRandom arg):
    cdef Random retv = Random()
    addref(arg); retv.rnd = arg
    return retv

cdef api PetscRandom PyPetscRandom_Get(object arg) except ? NULL:
    cdef PetscRandom retv = NULL
    cdef Random ob = <Random?> arg
    retv = ob.rnd
    return retv

# -- IS --

cdef api object PyPetscIS_New(PetscIS arg):
    cdef IS retv = IS()
    addref(arg); retv.iset = arg
    return retv

cdef api PetscIS PyPetscIS_Get(object arg) except? NULL:
    cdef PetscIS retv = NULL
    cdef IS ob = <IS?> arg
    retv = ob.iset
    return retv

# -- LGMap --

cdef api object PyPetscLGMap_New(PetscLGMap arg):
    cdef LGMap retv = LGMap()
    addref(arg); retv.lgm = arg
    return retv

cdef api PetscLGMap PyPetscLGMap_Get(object arg) except ? NULL:
    cdef PetscLGMap retv = NULL
    cdef LGMap ob = <LGMap?> arg
    retv = ob.lgm
    return retv

# -- Vec --

cdef api object PyPetscVec_New(PetscVec arg):
    cdef Vec retv = Vec()
    addref(arg); retv.vec = arg
    return retv

cdef api PetscVec PyPetscVec_Get(object arg) except ? NULL:
    cdef PetscVec retv = NULL
    cdef Vec ob = <Vec?> arg
    retv = ob.vec
    return retv

# -- Scatter --

cdef api object PyPetscScatter_New(PetscScatter arg):
    cdef Scatter retv = Scatter()
    addref(arg); retv.sct = arg
    return retv

cdef api PetscScatter PyPetscScatter_Get(object arg) except ? NULL:
    cdef PetscScatter retv = NULL
    cdef Scatter ob = <Scatter?> arg
    retv = ob.sct
    return retv

# -- Mat --

cdef api object PyPetscMat_New(PetscMat arg):
    cdef Mat retv = Mat()
    addref(arg); retv.mat = arg
    return retv

cdef api PetscMat PyPetscMat_Get(object arg) except ? NULL:
    cdef PetscMat retv = NULL
    cdef Mat ob = <Mat?> arg
    retv = ob.mat
    return retv

# -- PC --

cdef api object PyPetscPC_New(PetscPC arg):
    cdef PC retv = PC()
    addref(arg); retv.pc = arg
    return retv

cdef api PetscPC PyPetscPC_Get(object arg) except ? NULL:
    cdef PetscPC retv = NULL
    cdef PC ob = <PC?> arg
    retv = ob.pc
    return retv

# -- KSP --

cdef api object PyPetscKSP_New(PetscKSP arg):
    cdef KSP retv = KSP()
    addref(arg); retv.ksp = arg
    return retv

cdef api PetscKSP PyPetscKSP_Get(object arg) except ? NULL:
    cdef PetscKSP retv = NULL
    cdef KSP ob = <KSP?> arg
    retv = ob.ksp
    return retv

# -- SNES --

cdef api object PyPetscSNES_New(PetscSNES arg):
    cdef SNES retv = SNES()
    addref(arg); retv.snes = arg
    return retv

cdef api PetscSNES PyPetscSNES_Get(object arg) except ? NULL:
    cdef PetscSNES retv = NULL
    cdef SNES ob = <SNES?> arg
    retv = ob.snes
    return retv

# -- TS --

cdef api object PyPetscTS_New(PetscTS arg):
    cdef TS retv = TS()
    addref(arg); retv.ts = arg
    return retv

cdef api PetscTS PyPetscTS_Get(object arg) except ? NULL:
    cdef PetscTS retv = NULL
    cdef TS ob = <TS?> arg
    retv = ob.ts
    return retv

# -- AO --

cdef api object PyPetscAO_New(PetscAO arg):
    cdef AO retv = AO()
    addref(arg); retv.ao = arg
    return retv

cdef api PetscAO PyPetscAO_Get(object arg) except ? NULL:
    cdef PetscAO retv = NULL
    cdef AO ob = <AO?> arg
    retv = ob.ao
    return retv

# -- DM --

cdef api object PyPetscDM_New(PetscDM arg):
    cdef DM retv = subtype_DM(arg)()#DM()
    addref(arg); retv.dm[0] = arg
    return retv

cdef api PetscDM PyPetscDM_Get(object arg) except ? NULL:
    cdef PetscDM retv = NULL
    cdef DM ob = <DM?> arg
    retv = ob.dm[0]
    return retv

# -- DA --

cdef api object PyPetscDA_New(PetscDA arg):
    cdef DA retv = DA()
    addref(arg); retv.da = arg
    return retv

cdef api PetscDA PyPetscDA_Get(object arg) except ? NULL:
    cdef PetscDA retv = NULL
    cdef DA ob = <DA?> arg
    retv = ob.da
    return retv

#---------------------------------------------------------------------
