# --------------------------------------------------------------------

cdef extern from "mpi.h":
    ctypedef struct _p_MPI_Comm
    ctypedef _p_MPI_Comm* MPI_Comm

cdef extern from "petsc.h":
    ctypedef struct _p_PetscObject
    ctypedef _p_PetscObject* PetscObject

cdef extern from "petscviewer.h":
    struct _p_PetscViewer
    ctypedef _p_PetscViewer* PetscViewer

cdef extern from "petscsys.h":
    struct _p_PetscRandom
    ctypedef _p_PetscRandom* PetscRandom

cdef extern from "petscis.h":
    struct _p_IS
    ctypedef _p_IS* PetscIS "IS"
    struct _p_ISLocalToGlobalMapping
    ctypedef _p_ISLocalToGlobalMapping* PetscLGMap "ISLocalToGlobalMapping"

cdef extern from "petscvec.h":
    struct _p_Vec
    ctypedef _p_Vec* PetscVec "Vec"
    struct _p_VecScatter
    ctypedef _p_VecScatter* PetscScatter "VecScatter"

cdef extern from "petscmat.h":
    struct _p_Mat
    ctypedef _p_Mat* PetscMat "Mat"
    struct _p_MatNullSpace
    ctypedef _p_MatNullSpace* PetscNullSpace "MatNullSpace"

cdef extern from "petscpc.h":
    struct _p_PC
    ctypedef _p_PC* PetscPC "PC"

cdef extern from "petscksp.h":
    struct _p_KSP
    ctypedef _p_KSP* PetscKSP "KSP"

cdef extern from "petscsnes.h":
    struct _p_SNES
    ctypedef _p_SNES* PetscSNES "SNES"

cdef extern from "petscts.h":
    struct _p_TS
    ctypedef _p_TS* PetscTS "TS"

cdef extern from "petscao.h":
    struct _p_AO
    ctypedef _p_AO* PetscAO "AO"

cdef extern from "petscda.h":
    struct _p_DA
    ctypedef _p_DA* PetscDA "DA"

# --------------------------------------------------------------------

cdef public api class Comm [type PyPetscComm_Type, object PyPetscCommObject]:
    cdef MPI_Comm comm
    cdef int isdup

cdef public api class Object [type PyPetscObject_Type, object PyPetscObjectObject]:
    cdef PetscObject oval
    cdef PetscObject *obj
    cdef long inc_ref(self) except -1
    cdef long dec_ref(self) except -1
    cdef object get_attr(self, char name[])
    cdef object set_attr(self, char name[], object attr)
    cdef object get_dict(self)

cdef public api class Viewer(Object) [type PyPetscViewer_Type, object PyPetscViewerObject]:
    cdef PetscViewer vwr

cdef public api class Random(Object) [type PyPetscRandom_Type, object PyPetscRandomObject]:
    cdef PetscRandom rnd

cdef public api class IS(Object) [type PyPetscIS_Type, object PyPetscISObject]:
    cdef PetscIS iset

cdef public api class LGMap(Object) [type PyPetscLGMap_Type, object PyPetscLGMapObject]:
    cdef PetscLGMap lgm

cdef public api class Vec(Object) [type PyPetscVec_Type, object PyPetscVecObject]:
    cdef PetscVec vec

cdef public api class Scatter(Object) [type PyPetscScatter_Type, object PyPetscScatterObject]:
    cdef PetscScatter sct

cdef public api class Mat(Object) [type PyPetscMat_Type, object PyPetscMatObject]:
    cdef PetscMat mat

cdef public api class NullSpace(Object) [type PyPetscNullSpace_Type, object PyPetscNullSpaceObject]:
    cdef PetscNullSpace nsp

cdef public api class PC(Object) [type PyPetscPC_Type, object PyPetscPCObject]:
    cdef PetscPC pc

cdef public api class KSP(Object) [type PyPetscKSP_Type, object PyPetscKSPObject]:
    cdef PetscKSP ksp

cdef public api class SNES(Object) [type PyPetscSNES_Type, object PyPetscSNESObject]:
    cdef PetscSNES snes

cdef public api class TS(Object) [type PyPetscTS_Type, object PyPetscTSObject]:
    cdef PetscTS ts

cdef public api class AO(Object) [type PyPetscAO_Type, object PyPetscAOObject]:
    cdef PetscAO ao

cdef public api class DA(Object) [type PyPetscDA_Type, object PyPetscDAObject]:
    cdef PetscDA da

# --------------------------------------------------------------------

cdef extern from "petsc.h":
    ctypedef int PetscCookie

cdef int  RegisterPyType(PetscCookie, type) except -1
cdef type CookieToPyType(PetscCookie)

# --------------------------------------------------------------------
