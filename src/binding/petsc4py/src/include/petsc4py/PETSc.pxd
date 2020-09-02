# Author:  Lisandro Dalcin
# Contact: dalcinl@gmail.com

# --------------------------------------------------------------------

cdef extern from "petsc.h":

    ctypedef struct _p_MPI_Comm
    ctypedef _p_MPI_Comm* MPI_Comm

    ctypedef struct _p_MPI_Op
    ctypedef _p_MPI_Op* MPI_Op

    ctypedef struct _p_MPI_Datatype
    ctypedef _p_MPI_Datatype* MPI_Datatype

    ctypedef struct _p_PetscObject
    ctypedef _p_PetscObject* PetscObject

    struct _p_PetscViewer
    ctypedef _p_PetscViewer* PetscViewer

    struct _p_PetscRandom
    ctypedef _p_PetscRandom* PetscRandom

    struct _p_IS
    ctypedef _p_IS* PetscIS "IS"

    struct _p_ISLocalToGlobalMapping
    ctypedef _p_ISLocalToGlobalMapping* PetscLGMap "ISLocalToGlobalMapping"

    struct _p_PetscSF
    ctypedef _p_PetscSF* PetscSF "PetscSF"

    struct _p_Vec
    ctypedef _p_Vec* PetscVec "Vec"

    struct _p_VecScatter
    ctypedef _p_VecScatter* PetscScatter "VecScatter"

    struct _p_PetscFE
    ctypedef _p_PetscFE* PetscFE "PetscFE"

    struct _p_PetscQuadrature
    ctypedef _p_PetscQuadrature* PetscQuadrature "PetscQuadrature"

    struct _p_DMLabel
    ctypedef _p_DMLabel* PetscDMLabel "DMLabel"

    struct _p_PetscSection
    ctypedef _p_PetscSection* PetscSection

    struct _p_Mat
    ctypedef _p_Mat* PetscMat "Mat"

    struct _p_MatNullSpace
    ctypedef _p_MatNullSpace* PetscNullSpace "MatNullSpace"

    struct _p_PC
    ctypedef _p_PC* PetscPC "PC"

    struct _p_KSP
    ctypedef _p_KSP* PetscKSP "KSP"

    struct _p_SNES
    ctypedef _p_SNES* PetscSNES "SNES"

    struct _p_SNESLineSearch
    ctypedef _p_SNESLineSearch* PetscSNESLineSearch "SNESLineSearch"

    struct _p_TS
    ctypedef _p_TS* PetscTS "TS"

    struct _p_TAO "_p_Tao"
    ctypedef _p_TAO* PetscTAO "Tao"

    struct _p_AO
    ctypedef _p_AO* PetscAO "AO"

    struct _p_DM
    ctypedef _p_DM* PetscDM "DM"

    struct _p_PetscDS
    ctypedef _p_PetscDS* PetscDS

    struct _p_PetscPartitioner
    ctypedef _p_PetscPartitioner* PetscPartitioner "PetscPartitioner"

# --------------------------------------------------------------------

ctypedef public api class Comm [
    type   PyPetscComm_Type,
    object PyPetscCommObject,
    ]:
    cdef MPI_Comm comm
    cdef int isdup
    cdef object base

ctypedef public api class Object [
    type   PyPetscObject_Type,
    object PyPetscObjectObject,
    ]:
    cdef __weakref__
    cdef __dummy__
    cdef PetscObject oval
    cdef PetscObject *obj
    cdef object get_attr(self, char name[])
    cdef object set_attr(self, char name[], object attr)
    cdef object get_dict(self)

ctypedef public api class Viewer(Object) [
    type   PyPetscViewer_Type,
    object PyPetscViewerObject,
    ]:
    cdef PetscViewer vwr

ctypedef public api class Random(Object) [
    type   PyPetscRandom_Type,
    object PyPetscRandomObject,
    ]:
    cdef PetscRandom rnd

ctypedef public api class IS(Object) [
    type   PyPetscIS_Type,
    object PyPetscISObject,
    ]:
    cdef PetscIS iset

ctypedef public api class LGMap(Object) [
    type   PyPetscLGMap_Type,
    object PyPetscLGMapObject,
    ]:
    cdef PetscLGMap lgm

ctypedef public api class SF(Object) [
    type   PyPetscSF_Type,
    object PyPetscSFObject,
    ]:
    cdef PetscSF sf

ctypedef public api class Vec(Object) [
    type   PyPetscVec_Type,
    object PyPetscVecObject,
    ]:
    cdef PetscVec vec

ctypedef public api class FE(Object) [
    type   PyPetscFE_Type,
    object PyPetscFEObject,
    ]:
    cdef PetscFE fe

ctypedef public api class Quad(Object) [
    type   PyPetscQuad_Type,
    object PyPetscQuadObject,
    ]:
    cdef PetscQuadrature quad

ctypedef public api class Scatter(Object) [
    type   PyPetscScatter_Type,
    object PyPetscScatterObject,
    ]:
    cdef PetscScatter sct

ctypedef public api class Section(Object) [
    type   PyPetscSection_Type,
    object PyPetscSectionObject,
    ]:
    cdef PetscSection sec

ctypedef public api class Mat(Object) [
    type   PyPetscMat_Type,
    object PyPetscMatObject,
    ]:
    cdef PetscMat mat

ctypedef public api class NullSpace(Object) [
    type   PyPetscNullSpace_Type,
    object PyPetscNullSpaceObject,
    ]:
    cdef PetscNullSpace nsp

ctypedef public api class PC(Object) [
    type   PyPetscPC_Type,
    object PyPetscPCObject,
    ]:
    cdef PetscPC pc

ctypedef public api class KSP(Object) [
    type   PyPetscKSP_Type,
    object PyPetscKSPObject,
    ]:
    cdef PetscKSP ksp

ctypedef public api class SNES(Object) [
    type   PyPetscSNES_Type,
    object PyPetscSNESObject,
    ]:
    cdef PetscSNES snes

ctypedef public api class TS(Object) [
    type   PyPetscTS_Type,
    object PyPetscTSObject,
    ]:
    cdef PetscTS ts

ctypedef public api class TAO(Object) [
    type   PyPetscTAO_Type,
    object PyPetscTAOObject,
    ]:
    cdef PetscTAO tao

ctypedef public api class AO(Object) [
    type   PyPetscAO_Type,
    object PyPetscAOObject,
    ]:
    cdef PetscAO ao

ctypedef public api class DM(Object) [
    type   PyPetscDM_Type,
    object PyPetscDMObject,
    ]:
    cdef PetscDM dm

ctypedef public api class DS(Object) [
    type   PyPetscDS_Type,
    object PyPetscDSObject,
    ]:
    cdef PetscDS ds

ctypedef public api class Partitioner(Object) [
    type PyPetscPartitioner_Type,
    object PyPetscPartitionerObject,
    ]:
    cdef PetscPartitioner part

ctypedef public api class DMLabel(Object) [
    type   PyPetscDMLabel_Type,
    object PyPetscDMLabelObject,
    ]:
    cdef PetscDMLabel dmlabel
# --------------------------------------------------------------------

cdef MPI_Comm GetComm(object, MPI_Comm) except *
cdef MPI_Comm GetCommDefault()

cdef int  PyPetscType_Register(int, type) except -1
cdef type PyPetscType_Lookup(int)

# --------------------------------------------------------------------
