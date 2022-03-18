# DLPack interface

cdef extern from "Python.h":
    ctypedef void (*PyCapsule_Destructor)(object)
    bint PyCapsule_IsValid(object, const char*)
    void* PyCapsule_GetPointer(object, const char*) except? NULL
    int PyCapsule_SetName(object, const char*) except -1
    object PyCapsule_New(void*, const char*, PyCapsule_Destructor)
    int PyCapsule_CheckExact(object)

cdef extern from "stdlib.h" nogil:
    ctypedef signed long int64_t
    ctypedef unsigned long long uint64_t
    ctypedef unsigned char uint8_t
    ctypedef unsigned short uint16_t
    void free(void* ptr)
    void* malloc(size_t size)

cdef struct DLDataType:
    uint8_t code
    uint8_t bits
    uint16_t lanes

cdef enum PetscDLDeviceType:
    kDLCPU = <unsigned int>1
    kDLCUDA = <unsigned int>2
    kDLCUDAHost = <unsigned int>3
    #kDLOpenCL = <unsigned int>4
    #kDLVulkan = <unsigned int>7
    #kDLMetal = <unsigned int>8
    #kDLVPI = <unsigned int>9
    kDLROCM = <unsigned int>10
    kDLROCMHost = <unsigned int>11
    #kDLExtDev = <unsigned int>12
    kDLCUDAManaged = <unsigned int>13
    #kDLOneAPI = <unsigned int>14

ctypedef struct DLContext:
    PetscDLDeviceType device_type
    int device_id

cdef enum DLDataTypeCode:
    kDLInt = <unsigned int>0
    kDLUInt = <unsigned int>1
    kDLFloat = <unsigned int>2

cdef struct DLTensor:
    void* data
    DLContext ctx
    int ndim
    DLDataType dtype
    int64_t* shape
    int64_t* strides
    uint64_t byte_offset

ctypedef int (*dlpack_manager_del_obj)(void*) nogil

cdef struct DLManagedTensor:
    DLTensor dl_tensor
    void* manager_ctx
    void (*manager_deleter)(DLManagedTensor*) nogil
    dlpack_manager_del_obj del_obj

cdef void pycapsule_deleter(object dltensor):
    cdef DLManagedTensor* dlm_tensor = NULL
    try:
        dlm_tensor = <DLManagedTensor *>PyCapsule_GetPointer(dltensor, 'used_dltensor')
        return # we do not call a used capsule's deleter
    except Exception:
        dlm_tensor = <DLManagedTensor *>PyCapsule_GetPointer(dltensor, 'dltensor')
    manager_deleter(dlm_tensor)

cdef void manager_deleter(DLManagedTensor* tensor) nogil:
    if tensor.manager_ctx is NULL:
        return
    free(tensor.dl_tensor.shape)
    if tensor.del_obj is not NULL:
        tensor.del_obj(&tensor.manager_ctx)
    free(tensor)

# --------------------------------------------------------------------
