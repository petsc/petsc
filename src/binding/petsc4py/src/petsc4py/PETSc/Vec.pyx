# --------------------------------------------------------------------

class VecType(object):
    """The vector type."""
    SEQ        = S_(VECSEQ)
    MPI        = S_(VECMPI)
    STANDARD   = S_(VECSTANDARD)
    SHARED     = S_(VECSHARED)
    SEQVIENNACL= S_(VECSEQVIENNACL)
    MPIVIENNACL= S_(VECMPIVIENNACL)
    VIENNACL   = S_(VECVIENNACL)
    SEQCUDA    = S_(VECSEQCUDA)
    MPICUDA    = S_(VECMPICUDA)
    CUDA       = S_(VECCUDA)
    SEQHIP     = S_(VECSEQHIP)
    MPIHIP     = S_(VECMPIHIP)
    HIP        = S_(VECHIP)
    NEST       = S_(VECNEST)
    SEQKOKKOS  = S_(VECSEQKOKKOS)
    MPIKOKKOS  = S_(VECMPIKOKKOS)
    KOKKOS     = S_(VECKOKKOS)


class VecOption(object):
    """Vector assembly option."""
    IGNORE_OFF_PROC_ENTRIES = VEC_IGNORE_OFF_PROC_ENTRIES
    IGNORE_NEGATIVE_INDICES = VEC_IGNORE_NEGATIVE_INDICES

# --------------------------------------------------------------------


cdef class Vec(Object):
    """A vector object.

    See Also
    --------
    petsc.Vec

    """

    Type = VecType
    Option = VecOption

    #

    def __cinit__(self):
        self.obj = <PetscObject*> &self.vec
        self.vec = NULL

    # unary operations

    def __pos__(self):
        return vec_pos(self)

    def __neg__(self):
        return vec_neg(self)

    def __abs__(self):
        return vec_abs(self)

    # inplace binary operations

    def __iadd__(self, other):
        return vec_iadd(self, other)

    def __isub__(self, other):
        return vec_isub(self, other)

    def __imul__(self, other):
        return vec_imul(self, other)

    def __idiv__(self, other):
        return vec_idiv(self, other)

    def __itruediv__(self, other):
        return vec_idiv(self, other)

    # binary operations

    def __add__(self, other):
        return vec_add(self, other)

    def __radd__(self, other):
        return vec_radd(self, other)

    def __sub__(self, other):
        return vec_sub(self, other)

    def __rsub__(self, other):
        return vec_rsub(self, other)

    def __mul__(self, other):
        return vec_mul(self, other)

    def __rmul__(self, other):
        return vec_rmul(self, other)

    def __div__(self, other):
        return vec_div(self, other)

    def __rdiv__(self, other):
        return vec_rdiv(self, other)

    def __truediv__(self, other):
        return vec_div(self, other)

    def __rtruediv__(self, other):
        return vec_rdiv(self, other)

    def __matmul__(self, other):
        return vec_matmul(self, other)

    def __getitem__(self, i):
        return vec_getitem(self, i)

    def __setitem__(self, i, v):
        vec_setitem(self, i, v)

    # buffer interface (PEP 3118)

    def __getbuffer__(self, Py_buffer *view, int flags):
        cdef _Vec_buffer buf = _Vec_buffer(self)
        buf.acquirebuffer(view, flags)

    def __releasebuffer__(self, Py_buffer *view):
        cdef _Vec_buffer buf = <_Vec_buffer>(view.obj)
        buf.releasebuffer(view)
        <void>self # unused

    # 'with' statement (PEP 343)

    def __enter__(self):
        cdef _Vec_buffer buf = _Vec_buffer(self)
        self.set_attr('__buffer__', buf)
        return buf.enter()

    def __exit__(self, *exc):
        cdef _Vec_buffer buf = self.get_attr('__buffer__')
        self.set_attr('__buffer__', None)
        return buf.exit()

    #

    def view(self, Viewer viewer=None) -> None:
        """Display the vector.

        Collective.

        Parameters
        ----------
        viewer
            A `Viewer` instance or `None` for the default viewer.

        See Also
        --------
        load, petsc.VecView

        """
        cdef PetscViewer vwr = NULL
        if viewer is not None: vwr = viewer.vwr
        CHKERR(VecView(self.vec, vwr))

    def destroy(self) -> Self:
        """Destroy the vector.

        Collective.

        See Also
        --------
        create, petsc.VecDestroy

        """
        CHKERR(VecDestroy(&self.vec))
        return self

    def create(self, comm: Comm | None = None) -> Self:
        """Create a vector object.

        Collective.

        After creation the vector type can then be set with `setType`.

        Parameters
        ----------
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        destroy, petsc.VecCreate

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscVec newvec = NULL
        CHKERR(VecCreate(ccomm, &newvec))
        CHKERR(PetscCLEAR(self.obj)); self.vec = newvec
        return self

    def setType(self, vec_type: Type | str) -> None:
        """Set the vector type.

        Collective.

        Parameters
        ----------
        vec_type
            The vector type.

        See Also
        --------
        create, getType, petsc.VecSetType

        """
        cdef PetscVecType cval = NULL
        vec_type = str2bytes(vec_type, &cval)
        CHKERR(VecSetType(self.vec, cval))

    def setSizes(
        self,
        size: LayoutSizeSpec,
        bsize: int | None = None) -> None:
        """Set the local and global sizes of the vector.

        Collective.

        Parameters
        ----------
        size
            Vector size.
        bsize
            Vector block size. If `None`, ``bsize = 1``.

        See Also
        --------
        getSizes, petsc.VecSetSizes

        """
        cdef PetscInt bs=0, n=0, N=0
        Vec_Sizes(size, bsize, &bs, &n, &N)
        CHKERR(VecSetSizes(self.vec, n, N))
        if bs != PETSC_DECIDE:
            CHKERR(VecSetBlockSize(self.vec, bs))

    #

    # FIXME the comm argument is hideous.
    def createSeq(
        self,
        size: LayoutSizeSpec,
        bsize: int | None = None,
        comm: Comm | None = None) -> Self:
        """Create a sequential `Type.SEQ` vector.

        Collective.

        Parameters
        ----------
        size
            Vector size.
        bsize
            Vector block size. If `None`, ``bsize = 1``.
        comm
            MPI communicator, defaults to `COMM_SELF`.

        See Also
        --------
        createMPI, petsc.VecCreateSeq

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_SELF)
        cdef PetscInt bs=0, n=0, N=0
        Vec_Sizes(size, bsize, &bs, &n, &N)
        Sys_Layout(ccomm, bs, &n, &N)
        if bs == PETSC_DECIDE: bs = 1
        cdef PetscVec newvec = NULL
        CHKERR(VecCreate(ccomm, &newvec))
        CHKERR(VecSetSizes(newvec, n, N))
        CHKERR(VecSetBlockSize(newvec, bs))
        CHKERR(VecSetType(newvec, VECSEQ))
        CHKERR(PetscCLEAR(self.obj)); self.vec = newvec
        return self

    def createMPI(
        self,
        size: LayoutSizeSpec,
        bsize: int | None = None,
        comm: Comm | None = None) -> Self:
        """Create a parallel `Type.MPI` vector.

        Collective.

        Parameters
        ----------
        size
            Vector size.
        bsize
            Vector block size. If `None`, ``bsize = 1``.
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        createSeq, petsc.VecCreateMPI

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscInt bs=0, n=0, N=0
        Vec_Sizes(size, bsize, &bs, &n, &N)
        Sys_Layout(ccomm, bs, &n, &N)
        if bs == PETSC_DECIDE: bs = 1
        cdef PetscVec newvec = NULL
        CHKERR(VecCreate(ccomm, &newvec))
        CHKERR(VecSetSizes(newvec, n, N))
        CHKERR(VecSetBlockSize(newvec, bs))
        CHKERR(VecSetType(newvec, VECMPI))
        CHKERR(PetscCLEAR(self.obj)); self.vec = newvec
        return self

    def createWithArray(
        self,
        array: Sequence[Scalar],
        size: LayoutSizeSpec | None = None,
        bsize: int | None = None,
        comm: Comm | None = None) -> Self:
        """Create a vector using a provided array.

        Collective.

        This method will create either a `Type.SEQ` or `Type.MPI`
        depending on the size of the communicator.

        Parameters
        ----------
        array
            Array to store the vector values. Must be at least as large as
            the local size of the vector.
        size
            Vector size.
        bsize
            Vector block size. If `None`, ``bsize = 1``.
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        petsc.VecCreateSeqWithArray, petsc.VecCreateMPIWithArray

        """
        cdef PetscInt na=0
        cdef PetscScalar *sa=NULL
        array = iarray_s(array, &na, &sa)
        if size is None: size = (toInt(na), toInt(PETSC_DECIDE))
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscInt bs=0, n=0, N=0
        Vec_Sizes(size, bsize, &bs, &n, &N)
        Sys_Layout(ccomm, bs, &n, &N)
        if bs == PETSC_DECIDE: bs = 1
        if na < n:  raise ValueError(
            "array size %d and vector local size %d block size %d" %
            (toInt(na), toInt(n), toInt(bs)))
        cdef PetscVec newvec = NULL
        if comm_size(ccomm) == 1:
            CHKERR(VecCreateSeqWithArray(ccomm, bs, N, sa, &newvec))
        else:
            CHKERR(VecCreateMPIWithArray(ccomm, bs, n, N, sa, &newvec))
        CHKERR(PetscCLEAR(self.obj)); self.vec = newvec
        self.set_attr('__array__', array)
        return self

    def createCUDAWithArrays(
        self,
        cpuarray: Sequence[Scalar] | None = None,
        cudahandle: Any | None = None,  # FIXME What type is appropriate here?
        size: LayoutSizeSpec | None = None,
        bsize: int | None = None,
        comm: Comm | None = None) -> Self:
        """Create a `Type.CUDA` vector with optional arrays.

        Collective.

        Parameters
        ----------
        cpuarray
            Host array. Will be lazily allocated if not provided.
        cudahandle
            Address of the array on the GPU. Will be lazily allocated if
            not provided.
        size
            Vector size.
        bsize
            Vector block size. If `None`, ``bsize = 1``.
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        petsc.VecCreateSeqCUDAWithArrays, petsc.VecCreateMPICUDAWithArrays

        """
        cdef PetscInt na=0
        cdef PetscScalar *sa=NULL
        cdef PetscScalar *gpuarray = NULL
        if cudahandle:
            gpuarray = <PetscScalar*>(<Py_uintptr_t>cudahandle)
        if cpuarray is not None:
            cpuarray = iarray_s(cpuarray, &na, &sa)

        if size is None: size = (toInt(na), toInt(PETSC_DECIDE))
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscInt bs=0, n=0, N=0
        Vec_Sizes(size, bsize, &bs, &n, &N)
        Sys_Layout(ccomm, bs, &n, &N)
        if bs == PETSC_DECIDE: bs = 1
        if na < n:  raise ValueError(
            "array size %d and vector local size %d block size %d" %
            (toInt(na), toInt(n), toInt(bs)))
        cdef PetscVec newvec = NULL
        if comm_size(ccomm) == 1:
            CHKERR(VecCreateSeqCUDAWithArrays(ccomm, bs, N, sa, gpuarray, &newvec))
        else:
            CHKERR(VecCreateMPICUDAWithArrays(ccomm, bs, n, N, sa, gpuarray, &newvec))
        CHKERR(PetscCLEAR(self.obj)); self.vec = newvec

        if cpuarray is not None:
            self.set_attr('__array__', cpuarray)
        return self

    def createHIPWithArrays(
        self,
        cpuarray: Sequence[Scalar] | None = None,
        hiphandle: Any | None = None,  # FIXME What type is appropriate here?
        size: LayoutSizeSpec | None = None,
        bsize: int | None = None,
        comm: Comm | None = None) -> Self:
        """Create a `Type.HIP` vector with optional arrays.

        Collective.

        Parameters
        ----------
        cpuarray
            Host array. Will be lazily allocated if not provided.
        hiphandle
            Address of the array on the GPU. Will be lazily allocated if
            not provided.
        size
            Vector size.
        bsize
            Vector block size. If `None`, ``bsize = 1``.
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        petsc.VecCreateSeqHIPWithArrays, petsc.VecCreateMPIHIPWithArrays

        """
        cdef PetscInt na=0
        cdef PetscScalar *sa=NULL
        cdef PetscScalar *gpuarray = NULL
        if hiphandle:
            gpuarray = <PetscScalar*>(<Py_uintptr_t>hiphandle)
        if cpuarray is not None:
            cpuarray = iarray_s(cpuarray, &na, &sa)

        if size is None: size = (toInt(na), toInt(PETSC_DECIDE))
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscInt bs=0, n=0, N=0
        Vec_Sizes(size, bsize, &bs, &n, &N)
        Sys_Layout(ccomm, bs, &n, &N)
        if bs == PETSC_DECIDE: bs = 1
        if na < n:  raise ValueError(
            "array size %d and vector local size %d block size %d" %
            (toInt(na), toInt(n), toInt(bs)))
        cdef PetscVec newvec = NULL
        if comm_size(ccomm) == 1:
            CHKERR(VecCreateSeqHIPWithArrays(ccomm, bs, N, sa, gpuarray, &newvec))
        else:
            CHKERR(VecCreateMPIHIPWithArrays(ccomm, bs, n, N, sa, gpuarray, &newvec))
        CHKERR(PetscCLEAR(self.obj)); self.vec = newvec

        if cpuarray is not None:
            self.set_attr('__array__', cpuarray)
        return self

    def createViennaCLWithArrays(
        self,
        cpuarray: Sequence[Scalar] | None = None,
        viennaclvechandle: Any | None = None,  # FIXME What type is appropriate here?
        size: LayoutSizeSpec | None = None,
        bsize: int | None = None,
        comm: Comm | None = None) -> Self:
        """Create a `Type.VIENNACL` vector with optional arrays.

        Collective.

        Parameters
        ----------
        cpuarray
            Host array. Will be lazily allocated if not provided.
        viennaclvechandle
            Address of the array on the GPU. Will be lazily allocated if
            not provided.
        size
            Vector size.
        bsize
            Vector block size. If `None`, ``bsize = 1``.
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        petsc.VecCreateSeqViennaCLWithArrays
        petsc.VecCreateMPIViennaCLWithArrays

        """
        cdef PetscInt na=0
        cdef PetscScalar *sa=NULL
        cdef PetscScalar *vclvec = NULL
        if viennaclvechandle:
            vclvec = <PetscScalar*>(<Py_uintptr_t>viennaclvechandle)
        if cpuarray is not None:
            cpuarray = iarray_s(cpuarray, &na, &sa)

        if size is None: size = (toInt(na), toInt(PETSC_DECIDE))
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscInt bs=0, n=0, N=0
        Vec_Sizes(size, bsize, &bs, &n, &N)
        Sys_Layout(ccomm, bs, &n, &N)
        if bs == PETSC_DECIDE: bs = 1
        if na < n:
            raise ValueError("array size %d and vector local size %d block size %d" % (toInt(na), toInt(n), toInt(bs)))
        cdef PetscVec newvec = NULL
        if comm_size(ccomm) == 1:
            CHKERR(VecCreateSeqViennaCLWithArrays(ccomm, bs, N, sa, vclvec, &newvec))
        else:
            CHKERR(VecCreateMPIViennaCLWithArrays(ccomm, bs, n, N, sa, vclvec, &newvec))
        CHKERR(PetscCLEAR(self.obj)); self.vec = newvec

        if cpuarray is not None:
            self.set_attr('__array__', cpuarray)
        return self

    # FIXME: object? Do we need to specify it? Can't we just use Any?
    def createWithDLPack(
        self,
        object dltensor,
        size: LayoutSizeSpec | None = None,
        bsize: int | None = None,
        comm: Comm | None = None) -> Self:
        """Create a vector wrapping a DLPack object, sharing the same memory.

        Collective.

        This operation does not modify the storage of the original tensor and
        should be used with contiguous tensors only. If the tensor is stored in
        row-major order (e.g. PyTorch tensors), the resulting vector will look
        like an unrolled tensor using row-major order.

        The resulting vector type will be one of `Type.SEQ`, `Type.MPI`,
        `Type.SEQCUDA`, `Type.MPICUDA`, `Type.SEQHIP` or
        `Type.MPIHIP` depending on the type of ``dltensor`` and the number
        of processes in the communicator.

        Parameters
        ----------
        dltensor
            Either an object with a ``__dlpack__`` method or a DLPack tensor object.
        size
            Vector size.
        bsize
            Vector block size. If `None`, ``bsize = 1``.
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        """
        cdef DLManagedTensor* ptr = NULL
        cdef int bits = 0
        cdef PetscInt nz = 1
        cdef int64_t ndim = 0
        cdef int64_t* shape = NULL
        cdef int64_t* strides = NULL
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscInt bs = 0, n = 0, N = 0

        if not PyCapsule_CheckExact(dltensor):
            dltensor = dltensor.__dlpack__()

        if PyCapsule_IsValid(dltensor, 'dltensor'):
            ptr = <DLManagedTensor*>PyCapsule_GetPointer(dltensor, 'dltensor')
            bits = ptr.dl_tensor.dtype.bits
            if bits != 8*sizeof(PetscScalar):
                raise TypeError("Tensor dtype = {} does not match PETSc precision".format(ptr.dl_tensor.dtype))
            ndim = ptr.dl_tensor.ndim
            shape = ptr.dl_tensor.shape
            for s in shape[:ndim]:
                nz = nz*s
            strides = ptr.dl_tensor.strides
            PyCapsule_SetName(dltensor, 'used_dltensor')
        else:
            raise ValueError("Expect a dltensor field, pycapsule.PyCapsule can only be consumed once")
        if size is None: size = (toInt(nz), toInt(PETSC_DECIDE))
        Vec_Sizes(size, bsize, &bs, &n, &N)
        Sys_Layout(ccomm, bs, &n, &N)
        if bs == PETSC_DECIDE: bs = 1
        if nz < n:  raise ValueError(
            "array size %d and vector local size %d block size %d" %
            (toInt(nz), toInt(n), toInt(bs)))
        cdef PetscVec newvec = NULL
        cdef PetscDLDeviceType dltype = ptr.dl_tensor.ctx.device_type
        if dltype in [kDLCUDA, kDLCUDAManaged]:
            if comm_size(ccomm) == 1:
                CHKERR(VecCreateSeqCUDAWithArray(ccomm, bs, N, <PetscScalar*>(ptr.dl_tensor.data), &newvec))
            else:
                CHKERR(VecCreateMPICUDAWithArray(ccomm, bs, n, N, <PetscScalar*>(ptr.dl_tensor.data), &newvec))
        elif dltype in [kDLCPU, kDLCUDAHost, kDLROCMHost]:
            if comm_size(ccomm) == 1:
                CHKERR(VecCreateSeqWithArray(ccomm, bs, N, <PetscScalar*>(ptr.dl_tensor.data), &newvec))
            else:
                CHKERR(VecCreateMPIWithArray(ccomm, bs, n, N, <PetscScalar*>(ptr.dl_tensor.data), &newvec))
        elif dltype == kDLROCM:
            if comm_size(ccomm) == 1:
                CHKERR(VecCreateSeqHIPWithArray(ccomm, bs, N, <PetscScalar*>(ptr.dl_tensor.data), &newvec))
            else:
                CHKERR(VecCreateMPIHIPWithArray(ccomm, bs, n, N, <PetscScalar*>(ptr.dl_tensor.data), &newvec))
        else:
            raise TypeError("Device type {} not supported".format(dltype))

        CHKERR(PetscCLEAR(self.obj)); self.vec = newvec
        self.set_attr('__array__', dltensor)
        cdef int64_t* shape_arr = NULL
        cdef int64_t* strides_arr = NULL
        cdef object s1 = oarray_p(empty_p(<PetscInt>ndim), NULL, <void**>&shape_arr)
        cdef object s2 = oarray_p(empty_p(<PetscInt>ndim), NULL, <void**>&strides_arr)
        for i in range(ndim):
            shape_arr[i] = shape[i]
            strides_arr[i] = strides[i]
        self.set_attr('__dltensor_ctx__', (ptr.dl_tensor.ctx.device_type, ptr.dl_tensor.ctx.device_id, ndim, s1, s2))
        if ptr.manager_deleter != NULL:
            ptr.manager_deleter(ptr) # free the manager
        return self

    def attachDLPackInfo(
        self,
        Vec vec=None,
        object dltensor=None) -> Self:
        """Attach tensor information from another vector or DLPack tensor.

        Logically collective.

        This tensor information is required when converting a `Vec` to a
        DLPack object.

        Parameters
        ----------
        vec
            Vector with attached tensor information. This is typically created
            by calling `createWithDLPack`.
        dltensor
            DLPack tensor. This will only be used if ``vec`` is `None`.

        Notes
        -----
        This operation does not copy any data from ``vec`` or ``dltensor``.

        See Also
        --------
        clearDLPackInfo, createWithDLPack

        """
        cdef object ctx = None
        cdef DLManagedTensor* ptr = NULL
        cdef int64_t* shape_arr = NULL
        cdef int64_t* strides_arr = NULL
        cdef object s1 = None, s2 = None

        if vec is None and dltensor is None:
            raise ValueError('Missing input parameters')
        if vec is not None:
            ctx = (<Object>vec).get_attr('__dltensor_ctx__')
            if ctx is None:
                raise ValueError('Input vector has no tensor information')
            self.set_attr('__dltensor_ctx__', ctx)
        else:
            if PyCapsule_IsValid(dltensor, 'dltensor'):
                ptr = <DLManagedTensor*>PyCapsule_GetPointer(dltensor, 'dltensor')
            elif PyCapsule_IsValid(dltensor, 'used_dltensor'):
                ptr = <DLManagedTensor*>PyCapsule_GetPointer(dltensor, 'used_dltensor')
            else:
                raise ValueError("Expect a dltensor or used_dltensor field")
            bits = ptr.dl_tensor.dtype.bits
            if bits != 8*sizeof(PetscScalar):
                raise TypeError("Tensor dtype = {} does not match PETSc precision".format(ptr.dl_tensor.dtype))
            ndim = ptr.dl_tensor.ndim
            shape = ptr.dl_tensor.shape
            strides = ptr.dl_tensor.strides
            s1 = oarray_p(empty_p(ndim), NULL, <void**>&shape_arr)
            s2 = oarray_p(empty_p(ndim), NULL, <void**>&strides_arr)
            for i in range(ndim):
                shape_arr[i] = shape[i]
                strides_arr[i] = strides[i]
            self.set_attr('__dltensor_ctx__', (ptr.dl_tensor.ctx.device_type, ptr.dl_tensor.ctx.device_id, ndim, s1, s2))
        return self

    def clearDLPackInfo(self) -> Self:
        """Clear tensor information.

        Logically collective.

        See Also
        --------
        attachDLPackInfo, createWithDLPack

        """
        self.set_attr('__dltensor_ctx__', None)
        return self

    # TODO Stream
    def __dlpack__(self, stream=-1):
        return self.toDLPack('rw')

    def __dlpack_device__(self):
        (dltype, devId, _, _, _) = vec_get_dlpack_ctx(self)
        return (dltype, devId)

    def toDLPack(self, mode: AccessModeSpec = 'rw') -> Any:
        """Return a DLPack `PyCapsule` wrapping the vector data.

        Collective.

        Parameters
        ----------
        mode
            Access mode for the vector.

        Returns
        -------
        `PyCapsule`
            Capsule of a DLPack tensor wrapping a `Vec`.

        Notes
        -----
        It is important that the access mode is respected by the consumer
        as this is not enforced internally.

        See Also
        --------
        createWithDLPack

        """
        if mode is None: mode = 'rw'
        if mode not in ['rw', 'r', 'w']:
            raise ValueError("Invalid mode: expected 'rw', 'r', or 'w'")

        cdef int64_t ndim = 0
        (device_type, device_id, ndim, shape, strides) = vec_get_dlpack_ctx(self)
        hostmem = (device_type == kDLCPU)

        cdef DLManagedTensor* dlm_tensor = <DLManagedTensor*>malloc(sizeof(DLManagedTensor))
        cdef DLTensor* dl_tensor = &dlm_tensor.dl_tensor
        cdef PetscScalar *a = NULL
        cdef int64_t* shape_strides = NULL
        dl_tensor.byte_offset = 0

        # DLPack does not currently play well with our get/restore model
        # Call restore right-away and hope that the consumer will do the right thing
        # and not modify memory requested with read access
        # By restoring now, we guarantee the sanity of the ObjectState
        if mode == 'w':
            if hostmem:
                CHKERR(VecGetArrayWrite(self.vec, <PetscScalar**>&a))
                CHKERR(VecRestoreArrayWrite(self.vec, NULL))
            else:
                CHKERR(VecGetArrayWriteAndMemType(self.vec, <PetscScalar**>&a, NULL))
                CHKERR(VecRestoreArrayWriteAndMemType(self.vec, NULL))
        elif mode == 'r':
            if hostmem:
                CHKERR(VecGetArrayRead(self.vec, <const PetscScalar**>&a))
                CHKERR(VecRestoreArrayRead(self.vec, NULL))
            else:
                CHKERR(VecGetArrayReadAndMemType(self.vec, <const PetscScalar**>&a, NULL))
                CHKERR(VecRestoreArrayReadAndMemType(self.vec, NULL))
        else:
            if hostmem:
                CHKERR(VecGetArray(self.vec, <PetscScalar**>&a))
                CHKERR(VecRestoreArray(self.vec, NULL))
            else:
                CHKERR(VecGetArrayAndMemType(self.vec, <PetscScalar**>&a, NULL))
                CHKERR(VecRestoreArrayAndMemType(self.vec, NULL))
        dl_tensor.data = <void *>a

        cdef DLContext* ctx = &dl_tensor.ctx
        ctx.device_type = device_type
        ctx.device_id = device_id
        shape_strides = <int64_t*>malloc(sizeof(int64_t)*2*ndim)
        for i in range(ndim):
            shape_strides[i] = shape[i]
        for i in range(ndim):
            shape_strides[i+ndim] = strides[i]
        dl_tensor.ndim = <int>ndim
        dl_tensor.shape = shape_strides
        dl_tensor.strides = shape_strides + ndim

        cdef DLDataType* dtype = &dl_tensor.dtype
        dtype.code = <uint8_t>DLDataTypeCode.kDLFloat
        if sizeof(PetscScalar) == 8:
            dtype.bits = <uint8_t>64
        elif sizeof(PetscScalar) == 4:
            dtype.bits = <uint8_t>32
        else:
            raise ValueError('Unsupported PetscScalar type')
        dtype.lanes = <uint16_t>1
        dlm_tensor.manager_ctx = <void *>self.vec
        CHKERR(PetscObjectReference(<PetscObject>self.vec))
        dlm_tensor.manager_deleter = manager_deleter
        dlm_tensor.del_obj = <dlpack_manager_del_obj>PetscDEALLOC
        return PyCapsule_New(dlm_tensor, 'dltensor', pycapsule_deleter)

    def createGhost(
        self,
        ghosts: Sequence[int],
        size: LayoutSizeSpec,
        bsize: int | None = None,
        comm: Comm | None = None) -> Self:
        """Create a parallel vector with ghost padding on each processor.

        Collective.

        Parameters
        ----------
        ghosts
            Global indices of ghost points.
        size
            Vector size.
        bsize
            Vector block size. If `None`, ``bsize = 1``.
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        createGhostWithArray, petsc.VecCreateGhost, petsc.VecCreateGhostBlock

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscInt ng=0, *ig=NULL
        ghosts = iarray_i(ghosts, &ng, &ig)
        cdef PetscInt bs=0, n=0, N=0
        Vec_Sizes(size, bsize, &bs, &n, &N)
        Sys_Layout(ccomm, bs, &n, &N)
        cdef PetscVec newvec = NULL
        if bs == PETSC_DECIDE:
            CHKERR(VecCreateGhost(
                    ccomm, n, N, ng, ig, &newvec))
        else:
            CHKERR(VecCreateGhostBlock(
                    ccomm, bs, n, N, ng, ig, &newvec))
        CHKERR(PetscCLEAR(self.obj)); self.vec = newvec
        return self

    def createGhostWithArray(
        self,
        ghosts: Sequence[int],
        array: Sequence[Scalar],
        size: LayoutSizeSpec | None = None,
        bsize: int | None = None,
        comm: Comm | None = None) -> Self:
        """Create a parallel vector with ghost padding and provided arrays.

        Collective.

        Parameters
        ----------
        ghosts
            Global indices of ghost points.
        array
            Array to store the vector values. Must be at least as large as
            the local size of the vector (including ghost points).
        size
            Vector size.
        bsize
            Vector block size. If `None`, ``bsize = 1``.
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        createGhost, petsc.VecCreateGhostWithArray
        petsc.VecCreateGhostBlockWithArray

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscInt ng=0, *ig=NULL
        ghosts = iarray_i(ghosts, &ng, &ig)
        cdef PetscInt na=0
        cdef PetscScalar *sa=NULL
        array = oarray_s(array, &na, &sa)
        cdef PetscInt b = 1 if bsize is None else asInt(bsize)
        if size is None: size = (toInt(na-ng*b), toInt(PETSC_DECIDE))
        cdef PetscInt bs=0, n=0, N=0
        Vec_Sizes(size, bsize, &bs, &n, &N)
        Sys_Layout(ccomm, bs, &n, &N)
        if na < (n+ng*b): raise ValueError(
            "ghosts size %d, array size %d, and "
            "vector local size %d block size %d" %
            (toInt(ng), toInt(na), toInt(n), toInt(b)))
        cdef PetscVec newvec = NULL
        if bs == PETSC_DECIDE:
            CHKERR(VecCreateGhostWithArray(
                    ccomm, n, N, ng, ig, sa, &newvec))
        else:
            CHKERR(VecCreateGhostBlockWithArray(
                    ccomm, bs, n, N, ng, ig, sa, &newvec))
        CHKERR(PetscCLEAR(self.obj)); self.vec = newvec
        self.set_attr('__array__', array)
        return self

    def createShared(
        self,
        size: LayoutSizeSpec,
        bsize: int | None = None,
        comm: Comm | None = None) -> Self:
        """Create a `Type.SHARED` vector that uses shared memory.

        Collective.

        Parameters
        ----------
        size
            Vector size.
        bsize
            Vector block size. If `None`, ``bsize = 1``.
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        petsc.VecCreateShared

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscInt bs=0, n=0, N=0
        Vec_Sizes(size, bsize, &bs, &n, &N)
        Sys_Layout(ccomm, bs, &n, &N)
        cdef PetscVec newvec = NULL
        CHKERR(VecCreateShared(ccomm, n, N, &newvec))
        CHKERR(PetscCLEAR(self.obj)); self.vec = newvec
        if bs != PETSC_DECIDE:
            CHKERR(VecSetBlockSize(self.vec, bs))
        return self

    def createNest(
        self,
        vecs: Sequence[Vec],
        isets: Sequence[IS] | None = None,
        comm: Comm | None = None) -> Self:
        """Create a `Type.NEST` vector containing multiple nested subvectors.

        Collective.

        Parameters
        ----------
        vecs
            Iterable of subvectors.
        isets
            Iterable of index sets for each nested subvector.
            Defaults to contiguous ordering.
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        petsc.VecCreateNest

        """
        vecs = list(vecs)
        if isets:
            isets = list(isets)
            assert len(isets) == len(vecs)
        else:
            isets = None
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef Py_ssize_t i, m = len(vecs)
        cdef PetscInt n = <PetscInt>m
        cdef PetscVec *cvecs  = NULL
        cdef PetscIS  *cisets = NULL
        cdef object unused1, unused2
        unused1 = oarray_p(empty_p(n), NULL, <void**>&cvecs)
        for i from 0 <= i < m: cvecs[i] = (<Vec?>vecs[i]).vec
        if isets is not None:
            unused2 = oarray_p(empty_p(n), NULL, <void**>&cisets)
            for i from 0 <= i < m: cisets[i] = (<IS?>isets[i]).iset
        cdef PetscVec newvec = NULL
        CHKERR(VecCreateNest(ccomm, n, cisets, cvecs, &newvec))
        CHKERR(PetscCLEAR(self.obj)); self.vec = newvec
        return self

    #

    def setOptionsPrefix(self, prefix: str | None) -> None:
        """Set the prefix used for searching for options in the database.

        Logically collective.

        See Also
        --------
        petsc_options, getOptionsPrefix, petsc.VecSetOptionsPrefix

        """
        cdef const char *cval = NULL
        prefix = str2bytes(prefix, &cval)
        CHKERR(VecSetOptionsPrefix(self.vec, cval))

    def getOptionsPrefix(self) -> str:
        """Return the prefix used for searching for options in the database.

        Not collective.

        See Also
        --------
        petsc_options, setOptionsPrefix, petsc.VecGetOptionsPrefix

        """
        cdef const char *cval = NULL
        CHKERR(VecGetOptionsPrefix(self.vec, &cval))
        return bytes2str(cval)

    def appendOptionsPrefix(self, prefix: str | None) -> None:
        """Append to the prefix used for searching for options in the database.

        Logically collective.

        See Also
        --------
        petsc_options, setOptionsPrefix, petsc.VecAppendOptionsPrefix

        """
        cdef const char *cval = NULL
        prefix = str2bytes(prefix, &cval)
        CHKERR(VecAppendOptionsPrefix(self.vec, cval))

    def setFromOptions(self) -> None:
        """Configure the vector from the options database.

        Collective.

        See Also
        --------
        petsc_options, petsc.VecSetFromOptions

        """
        CHKERR(VecSetFromOptions(self.vec))

    def setUp(self) -> Self:
        """Set up the internal data structures for using the vector.

        Collective.

        See Also
        --------
        create, destroy, petsc.VecSetUp

        """
        CHKERR(VecSetUp(self.vec))
        return self

    def setOption(self, option: Option, flag: bool) -> None:
        """Set option.

        Collective.

        See Also
        --------
        petsc.VecSetOption

        """
        CHKERR(VecSetOption(self.vec, option, flag))

    def getType(self) -> str:
        """Return the type of the vector.

        Not collective.

        See Also
        --------
        setType, petsc.VecGetType

        """
        cdef PetscVecType cval = NULL
        CHKERR(VecGetType(self.vec, &cval))
        return bytes2str(cval)

    def getSize(self) -> int:
        """Return the global size of the vector.

        Not collective.

        See Also
        --------
        setSizes, getLocalSize, petsc.VecGetSize

        """
        cdef PetscInt N = 0
        CHKERR(VecGetSize(self.vec, &N))
        return toInt(N)

    def getLocalSize(self) -> int:
        """Return the local size of the vector.

        Not collective.

        See Also
        --------
        setSizes, getSize, petsc.VecGetLocalSize

        """
        cdef PetscInt n = 0
        CHKERR(VecGetLocalSize(self.vec, &n))
        return toInt(n)

    def getSizes(self) -> LayoutSizeSpec:
        """Return the vector sizes.

        Not collective.

        See Also
        --------
        getSize, getLocalSize, petsc.VecGetLocalSize, petsc.VecGetSize

        """
        cdef PetscInt n = 0, N = 0
        CHKERR(VecGetLocalSize(self.vec, &n))
        CHKERR(VecGetSize(self.vec, &N))
        return (toInt(n), toInt(N))

    def setBlockSize(self, bsize: int) -> None:
        """Set the block size of the vector.

        Logically collective.

        See Also
        --------
        petsc.VecSetBlockSize

        """
        cdef PetscInt bs = asInt(bsize)
        CHKERR(VecSetBlockSize(self.vec, bs))

    def getBlockSize(self) -> int:
        """Return the block size of the vector.

        Not collective.

        See Also
        --------
        petsc.VecGetBlockSize

        """
        cdef PetscInt bs=0
        CHKERR(VecGetBlockSize(self.vec, &bs))
        return toInt(bs)

    def getOwnershipRange(self) -> tuple[int, int]:
        """Return the locally owned range of indices ``(start, end)``.

        Not collective.

        Returns
        -------
        start : int
            The first local element.
        end : int
            One more than the last local element.

        See Also
        --------
        getOwnershipRanges, petsc.VecGetOwnershipRange

        """
        cdef PetscInt low=0, high=0
        CHKERR(VecGetOwnershipRange(self.vec, &low, &high))
        return (toInt(low), toInt(high))

    def getOwnershipRanges(self) -> ArrayInt:
        """Return the range of indices owned by each process.

        Not collective.

        The returned array is the result of exclusive scan of the local sizes.

        See Also
        --------
        getOwnershipRange, petsc.VecGetOwnershipRanges

        """
        cdef const PetscInt *rng = NULL
        CHKERR(VecGetOwnershipRanges(self.vec, &rng))
        cdef MPI_Comm comm = MPI_COMM_NULL
        CHKERR(PetscObjectGetComm(<PetscObject>self.vec, &comm))
        cdef int size = -1
        CHKERR(<PetscErrorCode>MPI_Comm_size(comm, &size))
        return array_i(size+1, rng)

    def createLocalVector(self) -> Vec:
        """Create a local vector.

        Not collective.

        Returns
        -------
        Vec
            The local vector.

        See Also
        --------
        getLocalVector, petsc.VecCreateLocalVector

        """
        lvec = Vec()
        CHKERR(VecCreateLocalVector(self.vec, &lvec.vec))
        return lvec

    def getLocalVector(self, Vec lvec, readonly: bool = False) -> None:
        """Maps the local portion of the vector into a local vector.

        Logically collective.

        Parameters
        ----------
        lvec
            The local vector obtained from `createLocalVector`.
        readonly
            Request read-only access.

        See Also
        --------
        createLocalVector, restoreLocalVector, petsc.VecGetLocalVectorRead
        petsc.VecGetLocalVector

        """
        if readonly:
            CHKERR(VecGetLocalVectorRead(self.vec, lvec.vec))
        else:
            CHKERR(VecGetLocalVector(self.vec, lvec.vec))

    def restoreLocalVector(self, Vec lvec, readonly: bool = False) -> None:
        """Unmap a local access obtained with `getLocalVector`.

        Logically collective.

        Parameters
        ----------
        lvec
            The local vector.
        readonly
            Request read-only access.

        See Also
        --------
        createLocalVector, getLocalVector, petsc.VecRestoreLocalVectorRead
        petsc.VecRestoreLocalVector

        """
        if readonly:
            CHKERR(VecRestoreLocalVectorRead(self.vec, lvec.vec))
        else:
            CHKERR(VecRestoreLocalVector(self.vec, lvec.vec))

    # FIXME Return type should be more specific
    def getBuffer(self, readonly: bool = False) -> Any:
        """Return a buffered view of the local portion of the vector.

        Logically collective.

        Parameters
        ----------
        readonly
            Request read-only access.

        Returns
        -------
        typing.Any
            `Buffer object <python:c-api/buffer>` wrapping the local portion of
            the vector data. This can be used either as a context manager
            providing access as a numpy array or can be passed to array
            constructors accepting buffered objects such as `numpy.asarray`.

        Examples
        --------
        Accessing the data with a context manager:

        >>> vec = PETSc.Vec().createWithArray([1, 2, 3])
        >>> with vec.getBuffer() as arr:
        ...     arr
        array([1., 2., 3.])

        Converting the buffer to an `ndarray`:

        >>> buf = PETSc.Vec().createWithArray([1, 2, 3]).getBuffer()
        >>> np.asarray(buf)
        array([1., 2., 3.])

        See Also
        --------
        getArray

        """
        if readonly:
            return vec_getbuffer_r(self)
        else:
            return vec_getbuffer_w(self)

    def getArray(self, readonly: bool = False) -> ArrayScalar:
        """Return local portion of the vector as an `ndarray`.

        Logically collective.

        Parameters
        ----------
        readonly
            Request read-only access.

        See Also
        --------
        setArray, getBuffer

        """
        if readonly:
            return vec_getarray_r(self)
        else:
            return vec_getarray_w(self)

    def setArray(self, array: Sequence[Scalar]) -> None:
        """Set values for the local portion of the vector.

        Logically collective.

        See Also
        --------
        placeArray

        """
        vec_setarray(self, array)

    def placeArray(self, array: Sequence[Scalar]) -> None:
        """Set the local portion of the vector to a provided array.

        Not collective.

        See Also
        --------
        resetArray, setArray, petsc.VecPlaceArray

        """
        cdef PetscInt nv=0
        cdef PetscInt na=0
        cdef PetscScalar *a = NULL
        CHKERR(VecGetLocalSize(self.vec, &nv))
        array = oarray_s(array, &na, &a)
        if (na != nv): raise ValueError(
            "cannot place input array size %d, vector size %d" %
            (toInt(na), toInt(nv)))
        CHKERR(VecPlaceArray(self.vec, a))
        self.set_attr('__placed_array__', array)

    def resetArray(self, force: bool = False) -> ArrayScalar | None:
        """Reset the vector to use its default array.

        Not collective.

        Parameters
        ----------
        force
            Force the calling of `petsc.VecResetArray` even if no user array
            has been placed with `placeArray`.

        Returns
        -------
        ArrayScalar
            The array previously provided by the user with `placeArray`.
            Can be `None` if ``force`` is `True` and no array was placed
            before.

        See Also
        --------
        placeArray, petsc.VecResetArray

        """
        cdef object array = None
        array = self.get_attr('__placed_array__')
        if array is None and not force: return None
        CHKERR(VecResetArray(self.vec))
        self.set_attr('__placed_array__', None)
        return array

    def bindToCPU(self, flg: bool) -> None:
        """Bind vector operations execution on the CPU.

        Logically collective.

        See Also
        --------
        boundToCPU, petsc.VecBindToCPU

        """
        cdef PetscBool bindFlg = asBool(flg)
        CHKERR(VecBindToCPU(self.vec, bindFlg))

    def boundToCPU(self) -> bool:
        """Return whether the vector has been bound to the CPU.

        Not collective.

        See Also
        --------
        bindToCPU, petsc.VecBoundToCPU

        """
        cdef PetscBool flg = PETSC_TRUE
        CHKERR(VecBoundToCPU(self.vec, &flg))
        return toBool(flg)

    def getCUDAHandle(
        self,
        mode: AccessModeSpec = 'rw') -> Any:  # FIXME What is the right return type?
        """Return a pointer to the device buffer.

        Not collective.

        The returned pointer should be released using `restoreCUDAHandle`
        with the same access mode.

        Returns
        -------
        typing.Any
            CUDA device pointer.

        Notes
        -----
        This method may incur a host-to-device copy if the device data is
        out of date and ``mode`` is ``"r"`` or ``"rw"``.

        See Also
        --------
        restoreCUDAHandle, petsc.VecCUDAGetArray, petsc.VecCUDAGetArrayRead
        petsc.VecCUDAGetArrayWrite

        """
        cdef PetscScalar *hdl = NULL
        cdef const char *m = NULL
        if mode is not None: mode = str2bytes(mode, &m)
        if m == NULL or (m[0] == c'r' and m[1] == c'w'):
            CHKERR(VecCUDAGetArray(self.vec, &hdl))
        elif m[0] == c'r':
            CHKERR(VecCUDAGetArrayRead(self.vec, <const PetscScalar**>&hdl))
        elif m[0] == c'w':
            CHKERR(VecCUDAGetArrayWrite(self.vec, &hdl))
        else:
            raise ValueError("Invalid mode: expected 'rw', 'r', or 'w'")
        return <Py_uintptr_t>hdl

    def restoreCUDAHandle(
        self,
        handle: Any,  # FIXME What type hint is appropriate?
        mode: AccessModeSpec = 'rw') -> None:
        """Restore a pointer to the device buffer obtained with `getCUDAHandle`.

        Not collective.

        Parameters
        ----------
        handle
            CUDA device pointer.
        mode
            Access mode.

        See Also
        --------
        getCUDAHandle, petsc.VecCUDARestoreArray
        petsc.VecCUDARestoreArrayRead, petsc.VecCUDARestoreArrayWrite

        """
        cdef PetscScalar *hdl = <PetscScalar*>(<Py_uintptr_t>handle)
        cdef const char *m = NULL
        if mode is not None: mode = str2bytes(mode, &m)
        if m == NULL or (m[0] == c'r' and m[1] == c'w'):
            CHKERR(VecCUDARestoreArray(self.vec, &hdl))
        elif m[0] == c'r':
            CHKERR(VecCUDARestoreArrayRead(self.vec, <const PetscScalar**>&hdl))
        elif m[0] == c'w':
            CHKERR(VecCUDARestoreArrayWrite(self.vec, &hdl))
        else:
            raise ValueError("Invalid mode: expected 'rw', 'r', or 'w'")

    def getHIPHandle(
        self,
        mode: AccessModeSpec = 'rw') -> Any:  # FIXME What is the right return type?
        """Return a pointer to the device buffer.

        Not collective.

        The returned pointer should be released using `restoreHIPHandle`
        with the same access mode.

        Returns
        -------
        typing.Any
            HIP device pointer.

        Notes
        -----
        This method may incur a host-to-device copy if the device data is
        out of date and ``mode`` is ``"r"`` or ``"rw"``.

        See Also
        --------
        restoreHIPHandle, petsc.VecHIPGetArray, petsc.VecHIPGetArrayRead
        petsc.VecHIPGetArrayWrite

        """
        cdef PetscScalar *hdl = NULL
        cdef const char *m = NULL
        if mode is not None: mode = str2bytes(mode, &m)
        if m == NULL or (m[0] == c'r' and m[1] == c'w'):
            CHKERR(VecHIPGetArray(self.vec, &hdl))
        elif m[0] == c'r':
            CHKERR(VecHIPGetArrayRead(self.vec, <const PetscScalar**>&hdl))
        elif m[0] == c'w':
            CHKERR(VecHIPGetArrayWrite(self.vec, &hdl))
        else:
            raise ValueError("Invalid mode: expected 'rw', 'r', or 'w'")
        return <Py_uintptr_t>hdl

    def restoreHIPHandle(
        self,
        handle: Any,  # FIXME What type hint is appropriate?
        mode: AccessModeSpec = 'rw') -> None:
        """Restore a pointer to the device buffer obtained with `getHIPHandle`.

        Not collective.

        Parameters
        ----------
        handle
            HIP device pointer.
        mode
            Access mode.

        See Also
        --------
        getHIPHandle, petsc.VecHIPRestoreArray, petsc.VecHIPRestoreArrayRead
        petsc.VecHIPRestoreArrayWrite

        """
        cdef PetscScalar *hdl = <PetscScalar*>(<Py_uintptr_t>handle)
        cdef const char *m = NULL
        if mode is not None: mode = str2bytes(mode, &m)
        if m == NULL or (m[0] == c'r' and m[1] == c'w'):
            CHKERR(VecHIPRestoreArray(self.vec, &hdl))
        elif m[0] == c'r':
            CHKERR(VecHIPRestoreArrayRead(self.vec, <const PetscScalar**>&hdl))
        elif m[0] == c'w':
            CHKERR(VecHIPRestoreArrayWrite(self.vec, &hdl))
        else:
            raise ValueError("Invalid mode: expected 'rw', 'r', or 'w'")

    def getOffloadMask(self) -> int:
        """Return the offloading status of the vector.

        Not collective.

        Common return values include:

        - 1: ``PETSC_OFFLOAD_CPU`` - CPU has valid entries
        - 2: ``PETSC_OFFLOAD_GPU`` - GPU has valid entries
        - 3: ``PETSC_OFFLOAD_BOTH`` - CPU and GPU are in sync

        Returns
        -------
        int
            Enum value from `petsc.PetscOffloadMask` describing the offloading
            status.

        See Also
        --------
        petsc.VecGetOffloadMask, petsc.PetscOffloadMask

        """
        cdef PetscOffloadMask mask = PETSC_OFFLOAD_UNALLOCATED
        CHKERR(VecGetOffloadMask(self.vec, &mask))
        return mask

    def getCLContextHandle(self) -> int:
        """Return the OpenCL context associated with the vector.

        Not collective.

        Returns
        -------
        int
            Pointer to underlying CL context. This can be used with
            `pyopencl` through `pyopencl.Context.from_int_ptr`.

        See Also
        --------
        getCLQueueHandle, petsc.VecViennaCLGetCLContext

        """
        cdef Py_uintptr_t ctxhdl = 0
        CHKERR(VecViennaCLGetCLContext(self.vec, &ctxhdl))
        return ctxhdl

    def getCLQueueHandle(self) -> int:
        """Return the OpenCL command queue associated with the vector.

        Not collective.

        Returns
        -------
        int
            Pointer to underlying CL command queue. This can be used with
            `pyopencl` through `pyopencl.Context.from_int_ptr`.

        See Also
        --------
        getCLContextHandle, petsc.VecViennaCLGetCLQueue

        """
        cdef Py_uintptr_t queuehdl = 0
        CHKERR(VecViennaCLGetCLQueue(self.vec, &queuehdl))
        return queuehdl

    def getCLMemHandle(
        self,
        mode: AccessModeSpec = 'rw') -> int:
        """Return the OpenCL buffer associated with the vector.

        Not collective.

        Returns
        -------
        int
            Pointer to the device buffer. This can be used with
            `pyopencl` through `pyopencl.Context.from_int_ptr`.

        Notes
        -----
        This method may incur a host-to-device copy if the device data is
        out of date and ``mode`` is ``"r"`` or ``"rw"``.

        See Also
        --------
        restoreCLMemHandle, petsc.VecViennaCLGetCLMem
        petsc.VecViennaCLGetCLMemRead, petsc.VecViennaCLGetCLMemWrite

        """
        cdef Py_uintptr_t memhdl = 0
        cdef const char *m = NULL
        mode = str2bytes(mode, &m)
        if m == NULL or (m[0] == c'r' and m[1] == c'w'):
            CHKERR(VecViennaCLGetCLMem(self.vec, &memhdl))
        elif m[0] == c'r':
            CHKERR(VecViennaCLGetCLMemRead(self.vec, &memhdl))
        elif m[0] == c'w':
            CHKERR(VecViennaCLGetCLMemWrite(self.vec, &memhdl))
        else:
            raise ValueError("Invalid mode: expected 'r', 'w' or 'rw'")
        return memhdl

    def restoreCLMemHandle(self) -> None:
        """Restore a pointer to the OpenCL buffer obtained with `getCLMemHandle`.

        Not collective.

        See Also
        --------
        getCLMemHandle, petsc.VecViennaCLRestoreCLMemWrite

        """
        CHKERR(VecViennaCLRestoreCLMemWrite(self.vec))

    def duplicate(self, array: Sequence[Scalar] | None = None) -> Vec:
        """Create a new vector with the same type, optionally with data.

        Collective.

        Parameters
        ----------
        array
            Optional values to store in the new vector.

        See Also
        --------
        copy, petsc.VecDuplicate

        """
        cdef Vec vec = type(self)()
        CHKERR(VecDuplicate(self.vec, &vec.vec))
        # duplicate tensor context
        cdef object ctx0 = self.get_attr('__dltensor_ctx__')
        if ctx0 is not None:
            vec.set_attr('__dltensor_ctx__', ctx0)
        if array is not None:
            vec_setarray(vec, array)
        return vec

    def copy(self, Vec result=None) -> Vec:
        """Return a copy of the vector.

        Logically collective.

        This operation copies vector entries to the new vector.

        Parameters
        ----------
        result
            Target vector for the copy. If `None` then a new vector is
            created internally.

        See Also
        --------
        duplicate, petsc.VecCopy

        """
        if result is None:
            result = type(self)()
        if result.vec == NULL:
            CHKERR(VecDuplicate(self.vec, &result.vec))
        CHKERR(VecCopy(self.vec, result.vec))
        return result

    def chop(self, tol: float) -> None:
        """Set all vector entries less than some absolute tolerance to zero.

        Collective.

        Parameters
        ----------
        tol
            The absolute tolerance below which entries are set to zero.

        See Also
        --------
        petsc.VecFilter

        """
        cdef PetscReal rval = asReal(tol)
        CHKERR(VecFilter(self.vec, rval))

    def load(self, Viewer viewer) -> Self:
        """Load a vector.

        Collective.

        See Also
        --------
        view, petsc.VecLoad

        """
        cdef MPI_Comm comm = MPI_COMM_NULL
        cdef PetscObject obj = <PetscObject>(viewer.vwr)
        if self.vec == NULL:
            CHKERR(PetscObjectGetComm(obj, &comm))
            CHKERR(VecCreate(comm, &self.vec))
        CHKERR(VecLoad(self.vec, viewer.vwr))
        return self

    def equal(self, Vec vec) -> bool:
        """Return whether the vector is equal to another.

        Collective.

        Parameters
        ----------
        vec
            Vector to compare with.

        See Also
        --------
        petsc.VecEqual

        """
        cdef PetscBool flag = PETSC_FALSE
        CHKERR(VecEqual(self.vec, vec.vec, &flag))
        return toBool(flag)

    def dot(self, Vec vec) -> Scalar:
        """Return the dot product with ``vec``.

        Collective.

        For complex numbers this computes yᴴ·x with ``self`` as x, ``vec``
        as y and where yᴴ denotes the conjugate transpose of y.

        Use `tDot` for the indefinite form yᵀ·x where yᵀ denotes the
        transpose of y.

        Parameters
        ----------
        vec
            Vector to compute the dot product with.

        See Also
        --------
        dotBegin, dotEnd, tDot, petsc.VecDot

        """
        cdef PetscScalar sval = 0
        CHKERR(VecDot(self.vec, vec.vec, &sval))
        return toScalar(sval)

    def dotBegin(self, Vec vec) -> None:
        """Begin computing the dot product.

        Collective.

        This should be paired with a call to `dotEnd`.

        Parameters
        ----------
        vec
            Vector to compute the dot product with.

        See Also
        --------
        dotEnd, dot, petsc.VecDotBegin

        """
        cdef PetscScalar sval = 0
        CHKERR(VecDotBegin(self.vec, vec.vec, &sval))

    def dotEnd(self, Vec vec) -> Scalar:
        """Finish computing the dot product initiated with `dotBegin`.

        Collective.

        See Also
        --------
        dotBegin, dot, petsc.VecDotEnd

        """
        cdef PetscScalar sval = 0
        CHKERR(VecDotEnd(self.vec, vec.vec, &sval))
        return toScalar(sval)

    def tDot(self, Vec vec) -> Scalar:
        """Return the indefinite dot product with ``vec``.

        Collective.

        This computes yᵀ·x with ``self`` as x, ``vec``
        as y and where yᵀ denotes the transpose of y.

        Parameters
        ----------
        vec
            Vector to compute the indefinite dot product with.

        See Also
        --------
        tDotBegin, tDotEnd, dot, petsc.VecTDot

        """
        cdef PetscScalar sval = 0
        CHKERR(VecTDot(self.vec, vec.vec, &sval))
        return toScalar(sval)

    def tDotBegin(self, Vec vec) -> None:
        """Begin computing the indefinite dot product.

        Collective.

        This should be paired with a call to `tDotEnd`.

        Parameters
        ----------
        vec
            Vector to compute the indefinite dot product with.

        See Also
        --------
        tDotEnd, tDot, petsc.VecTDotBegin

        """
        cdef PetscScalar sval = 0
        CHKERR(VecTDotBegin(self.vec, vec.vec, &sval))

    def tDotEnd(self, Vec vec) -> Scalar:
        """Finish computing the indefinite dot product initiated with `tDotBegin`.

        Collective.

        See Also
        --------
        tDotBegin, tDot, petsc.VecTDotEnd

        """
        cdef PetscScalar sval = 0
        CHKERR(VecTDotEnd(self.vec, vec.vec, &sval))
        return toScalar(sval)

    def mDot(self, vecs: Sequence[Vec], out: ArrayScalar | None = None) -> ArrayScalar:
        """Compute Xᴴ·y with X an array of vectors.

        Collective.

        Parameters
        ----------
        vecs
            Array of vectors.
        out
            Optional placeholder for the result.

        See Also
        --------
        dot, tDot, mDotBegin, mDotEnd, petsc.VecMDot

        """
        cdef PetscInt nv=<PetscInt>len(vecs), no=0
        cdef PetscVec *v=NULL
        cdef PetscScalar *val=NULL
        cdef Py_ssize_t i=0
        cdef object unused = oarray_p(empty_p(nv), NULL, <void**>&v)
        for i from 0 <= i < nv:
            v[i] = (<Vec?>(vecs[i])).vec
        if out is None:
            out = empty_s(nv)
        out = oarray_s(out, &no, &val)
        if (nv != no): raise ValueError(
            ("incompatible array sizes: "
             "nv=%d, no=%d") % (toInt(nv), toInt(no)))
        CHKERR(VecMDot(self.vec, nv, v, val))
        return out

    def mDotBegin(self, vecs: Sequence[Vec], out: ArrayScalar) -> None:
        """Starts a split phase multiple dot product computation.

        Collective.

        Parameters
        ----------
        vecs
            Array of vectors.
        out
            Placeholder for the result.

        See Also
        --------
        mDot, mDotEnd, petsc.VecMDotBegin

        """
        cdef PetscInt nv=<PetscInt>len(vecs), no=0
        cdef PetscVec *v=NULL
        cdef PetscScalar *val=NULL
        cdef Py_ssize_t i=0
        cdef object unused = oarray_p(empty_p(nv), NULL, <void**>&v)
        for i from 0 <= i < nv:
            v[i] = (<Vec?>(vecs[i])).vec
        out = oarray_s(out, &no, &val)
        if (nv != no): raise ValueError(
            ("incompatible array sizes: "
             "nv=%d, no=%d") % (toInt(nv), toInt(no)))
        CHKERR(VecMDotBegin(self.vec, nv, v, val))

    def mDotEnd(self, vecs: Sequence[Vec], out: ArrayScalar) -> ArrayScalar:
        """Ends a split phase multiple dot product computation.

        Collective.

        Parameters
        ----------
        vecs
            Array of vectors.
        out
            Placeholder for the result.

        See Also
        --------
        mDot, mDotBegin, petsc.VecMDotEnd

        """
        cdef PetscInt nv=<PetscInt>len(vecs), no=0
        cdef PetscVec *v=NULL
        cdef PetscScalar *val=NULL
        cdef Py_ssize_t i=0
        cdef object unused = oarray_p(empty_p(nv), NULL, <void**>&v)
        for i from 0 <= i < nv:
            v[i] = (<Vec?>(vecs[i])).vec
        out = oarray_s(out, &no, &val)
        if (nv != no): raise ValueError(
            ("incompatible array sizes: "
             "nv=%d, no=%d") % (toInt(nv), toInt(no)))
        CHKERR(VecMDotEnd(self.vec, nv, v, val))
        return out

    def mtDot(self, vecs: Sequence[Vec], out: ArrayScalar | None = None) -> ArrayScalar:
        """Compute Xᵀ·y with X an array of vectors.

        Collective.

        Parameters
        ----------
        vecs
            Array of vectors.
        out
            Optional placeholder for the result.

        See Also
        --------
        tDot, mDot, mtDotBegin, mtDotEnd, petsc.VecMTDot

        """
        cdef PetscInt nv=<PetscInt>len(vecs), no=0
        cdef PetscVec *v=NULL
        cdef PetscScalar *val=NULL
        cdef Py_ssize_t i=0
        cdef object unused = oarray_p(empty_p(nv), NULL, <void**>&v)
        for i from 0 <= i < nv:
            v[i] = (<Vec?>(vecs[i])).vec
        if out is None:
            out = empty_s(nv)
        out = oarray_s(out, &no, &val)
        if (nv != no): raise ValueError(
            ("incompatible array sizes: "
             "nv=%d, no=%d") % (toInt(nv), toInt(no)))
        CHKERR(VecMTDot(self.vec, nv, v, val))
        return out

    def mtDotBegin(self, vecs: Sequence[Vec], out: ArrayScalar) -> None:
        """Starts a split phase transpose multiple dot product computation.

        Collective.

        Parameters
        ----------
        vecs
            Array of vectors.
        out
            Placeholder for the result.

        See Also
        --------
        mtDot, mtDotEnd, petsc.VecMTDotBegin

        """
        cdef PetscInt nv=<PetscInt>len(vecs), no=0
        cdef PetscVec *v=NULL
        cdef PetscScalar *val=NULL
        cdef Py_ssize_t i=0
        cdef object unused = oarray_p(empty_p(nv), NULL, <void**>&v)
        for i from 0 <= i < nv:
            v[i] = (<Vec?>(vecs[i])).vec
        out = oarray_s(out, &no, &val)
        if (nv != no): raise ValueError(
            ("incompatible array sizes: "
             "nv=%d, no=%d") % (toInt(nv), toInt(no)))
        CHKERR(VecMTDotBegin(self.vec, nv, v, val))

    def mtDotEnd(self, vecs: Sequence[Vec], out: ArrayScalar) -> ArrayScalar:
        """Ends a split phase transpose multiple dot product computation.

        Collective.

        Parameters
        ----------
        vecs
            Array of vectors.
        out
            Placeholder for the result.

        See Also
        --------
        mtDot, mtDotBegin, petsc.VecMTDotEnd

        """
        cdef PetscInt nv=<PetscInt>len(vecs), no=0
        cdef PetscVec *v=NULL
        cdef PetscScalar *val=NULL
        cdef Py_ssize_t i=0
        cdef object unused = oarray_p(empty_p(nv), NULL, <void**>&v)
        for i from 0 <= i < nv:
            v[i] = (<Vec?>(vecs[i])).vec
        out = oarray_s(out, &no, &val)
        if (nv != no): raise ValueError(
            ("incompatible array sizes: "
             "nv=%d, no=%d") % (toInt(nv), toInt(no)))
        CHKERR(VecMTDotEnd(self.vec, nv, v, val))
        return out

    def norm(
        self,
        norm_type: NormTypeSpec = None) -> float | tuple[float, float]:
        """Compute the vector norm.

        Collective.

        A 2-tuple is returned if `NormType.NORM_1_AND_2` is specified.

        See Also
        --------
        petsc.VecNorm, petsc.NormType

        """
        cdef PetscNormType norm_1_2 = PETSC_NORM_1_AND_2
        cdef PetscNormType ntype = PETSC_NORM_2
        if norm_type is not None: ntype = norm_type
        cdef PetscReal rval[2]
        CHKERR(VecNorm(self.vec, ntype, rval))
        if ntype != norm_1_2: return toReal(rval[0])
        else: return (toReal(rval[0]), toReal(rval[1]))

    def normBegin(
        self,
        norm_type: NormTypeSpec = None) -> None:
        """Begin computing the vector norm.

        Collective.

        This should be paired with a call to `normEnd`.

        See Also
        --------
        normEnd, norm, petsc.VecNormBegin

        """
        cdef PetscNormType ntype = PETSC_NORM_2
        if norm_type is not None: ntype = norm_type
        cdef PetscReal dummy[2]
        CHKERR(VecNormBegin(self.vec, ntype, dummy))

    def normEnd(
        self,
        norm_type: NormTypeSpec = None) -> float | tuple[float, float]:
        """Finish computations initiated with `normBegin`.

        Collective.

        See Also
        --------
        normBegin, norm, petsc.VecNormEnd

        """
        cdef PetscNormType norm_1_2 = PETSC_NORM_1_AND_2
        cdef PetscNormType ntype = PETSC_NORM_2
        if norm_type is not None: ntype = norm_type
        cdef PetscReal rval[2]
        CHKERR(VecNormEnd(self.vec, ntype, rval))
        if ntype != norm_1_2: return toReal(rval[0])
        else: return (toReal(rval[0]), toReal(rval[1]))

    def dotNorm2(self, Vec vec) -> tuple[Scalar, float]:
        """Return the dot product with ``vec`` and its squared norm.

        Collective.

        See Also
        --------
        dot, norm, petsc.VecDotNorm2

        """
        cdef PetscScalar sval = 0
        cdef PetscReal rval = 0
        CHKERR(VecDotNorm2(self.vec, vec.vec, &sval, &rval))
        return toScalar(sval), toReal(float)

    def sum(self) -> Scalar:
        """Return the sum of all the entries of the vector.

        Collective.

        See Also
        --------
        petsc.VecSum

        """
        cdef PetscScalar sval = 0
        CHKERR(VecSum(self.vec, &sval))
        return toScalar(sval)

    def mean(self) -> Scalar:
        """Return the arithmetic mean of all the entries of the vector.

        Collective.

        See Also
        --------
        petsc.VecMean

        """
        cdef PetscScalar sval = 0
        CHKERR(VecMean(self.vec, &sval))
        return toScalar(sval)

    def min(self) -> tuple[int, float]:
        """Return the vector entry with minimum real part and its location.

        Collective.

        Returns
        -------
        p : int
            Location of the minimum value. If multiple entries exist with the
            same value then the smallest index will be returned.
        val : Scalar
            Minimum value.

        See Also
        --------
        max, petsc.VecMin

        """
        cdef PetscInt  ival = 0
        cdef PetscReal rval = 0
        CHKERR(VecMin(self.vec, &ival, &rval))
        return (toInt(ival), toReal(rval))

    def max(self) -> tuple[int, float]:
        """Return the vector entry with maximum real part and its location.

        Collective.

        Returns
        -------
        p : int
            Location of the maximum value. If multiple entries exist with the
            same value then the smallest index will be returned.
        val : Scalar
            Minimum value.

        See Also
        --------
        min, petsc.VecMax

        """
        cdef PetscInt  ival = 0
        cdef PetscReal rval = 0
        CHKERR(VecMax(self.vec, &ival, &rval))
        return (toInt(ival), toReal(rval))

    def normalize(self) -> float:
        """Normalize the vector by its 2-norm.

        Collective.

        Returns
        -------
        float
            The vector norm before normalization.

        See Also
        --------
        norm, petsc.VecNormalize

        """
        cdef PetscReal rval = 0
        CHKERR(VecNormalize(self.vec, &rval))
        return toReal(rval)

    def reciprocal(self) -> None:
        """Replace each entry in the vector by its reciprocal.

        Logically collective.

        See Also
        --------
        petsc.VecReciprocal

        """
        CHKERR(VecReciprocal(self.vec))

    def exp(self) -> None:
        """Replace each entry (xₙ) in the vector by exp(xₙ).

        Logically collective.

        See Also
        --------
        log, petsc.VecExp

        """
        CHKERR(VecExp(self.vec))

    def log(self) -> None:
        """Replace each entry in the vector by its natural logarithm.

        Logically collective.

        See Also
        --------
        exp, petsc.VecLog

        """
        CHKERR(VecLog(self.vec))

    def sqrtabs(self) -> None:
        """Replace each entry (xₙ) in the vector by √|xₙ|.

        Logically collective.

        See Also
        --------
        petsc.VecSqrtAbs

        """
        CHKERR(VecSqrtAbs(self.vec))

    def abs(self) -> None:
        """Replace each entry (xₙ) in the vector by abs|xₙ|.

        Logically collective.

        See Also
        --------
        petsc.VecAbs

        """
        CHKERR(VecAbs(self.vec))

    def conjugate(self) -> None:
        """Conjugate the vector.

        Logically collective.

        See Also
        --------
        petsc.VecConjugate

        """
        CHKERR(VecConjugate(self.vec))

    def setRandom(self, Random random=None) -> None:
        """Set all components of the vector to random numbers.

        Collective.

        Parameters
        ----------
        random
            Random number generator. If `None` then one will be created
            internally.

        See Also
        --------
        petsc.VecSetRandom

        """
        cdef PetscRandom rnd = NULL
        if random is not None: rnd = random.rnd
        CHKERR(VecSetRandom(self.vec, rnd))

    def permute(self, IS order, invert: bool = False) -> None:
        """Permute the vector in-place with a provided ordering.

        Collective.

        Parameters
        ----------
        order
            Ordering for the permutation.
        invert
            Whether to invert the permutation.

        See Also
        --------
        petsc.VecPermute

        """
        cdef PetscBool cinvert = PETSC_FALSE
        if invert: cinvert = PETSC_TRUE
        CHKERR(VecPermute(self.vec, order.iset, cinvert))

    def zeroEntries(self) -> None:
        """Set all entries in the vector to zero.

        Logically collective.

        See Also
        --------
        set, petsc.VecZeroEntries

        """
        CHKERR(VecZeroEntries(self.vec))

    def set(self, alpha: Scalar) -> None:
        """Set all components of the vector to the same value.

        Collective.

        See Also
        --------
        zeroEntries, isset, petsc.VecSet

        """
        cdef PetscScalar sval = asScalar(alpha)
        CHKERR(VecSet(self.vec, sval))

    def isset(self, IS idx, alpha: Scalar) -> None:
        """Set specific elements of the vector to the same value.

        Not collective.

        Parameters
        ----------
        idx
            Index set specifying the vector entries to set.
        alpha
            Value to set the selected entries to.

        See Also
        --------
        set, zeroEntries, petsc.VecISSet

        """
        cdef PetscScalar aval = asScalar(alpha)
        CHKERR(VecISSet(self.vec, idx.iset, aval))

    def scale(self, alpha: Scalar) -> None:
        """Scale all entries of the vector.

        Collective.

        This method sets each entry (xₙ) in the vector to ɑ·xₙ.

        Parameters
        ----------
        alpha
            The scaling factor.

        See Also
        --------
        shift, petsc.VecScale

        """
        cdef PetscScalar sval = asScalar(alpha)
        CHKERR(VecScale(self.vec, sval))

    def shift(self, alpha: Scalar) -> None:
        """Shift all entries in the vector.

        Collective.

        This method sets each entry (xₙ) in the vector to xₙ + ɑ.

        Parameters
        ----------
        alpha
            The shift to apply to the vector values.

        See Also
        --------
        scale, petsc.VecShift

        """
        cdef PetscScalar sval = asScalar(alpha)
        CHKERR(VecShift(self.vec, sval))

    def swap(self, Vec vec) -> None:
        """Swap the content of two vectors.

        Logically collective.

        Parameters
        ----------
        vec
            The vector to swap data with.

        See Also
        --------
        petsc.VecSwap

        """
        CHKERR(VecSwap(self.vec, vec.vec))

    def axpy(self, alpha: Scalar, Vec x) -> None:
        """Compute and store y = ɑ·x + y.

        Logically collective.

        Parameters
        ----------
        alpha
            Scale factor.
        x
            Input vector.

        See Also
        --------
        isaxpy, petsc.VecAXPY

        """
        cdef PetscScalar sval = asScalar(alpha)
        CHKERR(VecAXPY(self.vec, sval, x.vec))

    def isaxpy(self, IS idx, alpha: Scalar, Vec x) -> None:
        """Add a scaled reduced-space vector to a subset of the vector.

        Logically collective.

        This is equivalent to ``y[idx[i]] += alpha*x[i]``.

        Parameters
        ----------
        idx
            Index set for the reduced space. Negative indices are skipped.
        alpha
            Scale factor.
        x
            Reduced-space vector.

        See Also
        --------
        axpy, aypx, axpby, petsc.VecISAXPY

        """
        cdef PetscScalar sval = asScalar(alpha)
        CHKERR(VecISAXPY(self.vec, idx.iset, sval, x.vec))

    def aypx(self, alpha: Scalar, Vec x) -> None:
        """Compute and store y = x + ɑ·y.

        Logically collective.

        Parameters
        ----------
        alpha
            Scale factor.
        x
            Input vector, must not be the current vector.

        See Also
        --------
        axpy, axpby, petsc.VecAYPX

        """
        cdef PetscScalar sval = asScalar(alpha)
        CHKERR(VecAYPX(self.vec, sval, x.vec))

    def axpby(self, alpha: Scalar, beta: Scalar, Vec x) -> None:
        """Compute and store y = ɑ·x + β·y.

        Logically collective.

        Parameters
        ----------
        alpha
            First scale factor.
        beta
            Second scale factor.
        x
            Input vector, must not be the current vector.

        See Also
        --------
        axpy, aypx, waxpy, petsc.VecAXPBY

        """
        cdef PetscScalar sval1 = asScalar(alpha)
        cdef PetscScalar sval2 = asScalar(beta)
        CHKERR(VecAXPBY(self.vec, sval1, sval2, x.vec))

    def waxpy(self, alpha: Scalar, Vec x, Vec y) -> None:
        """Compute and store w = ɑ·x + y.

        Logically collective.

        Parameters
        ----------
        alpha
            Scale factor.
        x
            First input vector.
        y
            Second input vector.

        See Also
        --------
        axpy, aypx, axpby, maxpy, petsc.VecWAXPY

        """
        cdef PetscScalar sval = asScalar(alpha)
        CHKERR(VecWAXPY(self.vec, sval, x.vec, y.vec))

    def maxpy(self, alphas: Sequence[Scalar], vecs: Sequence[Vec]) -> None:
        """Compute and store y = Σₙ(ɑₙ·Xₙ) + y with X an array of vectors.

        Logically collective.

        Equivalent to ``y[:] = alphas[i]*vecs[i, :] + y[:]``.

        Parameters
        ----------
        alphas
            Array of scale factors, one for each vector in ``vecs``.
        vecs
            Array of vectors.

        See Also
        --------
        axpy, aypx, axpby, waxpy, petsc.VecMAXPY

        """
        cdef PetscInt n = 0
        cdef PetscScalar *a = NULL
        cdef PetscVec *v = NULL
        cdef object unused1 = iarray_s(alphas, &n, &a)
        cdef object unused2 = oarray_p(empty_p(n), NULL, <void**>&v)
        assert n == len(vecs)
        cdef Py_ssize_t i=0
        for i from 0 <= i < n:
            v[i] = (<Vec?>(vecs[i])).vec
        CHKERR(VecMAXPY(self.vec, n, a, v))

    def pointwiseMult(self, Vec x, Vec y) -> None:
        """Compute and store the component-wise multiplication of two vectors.

        Logically collective.

        Equivalent to ``w[i] = x[i] * y[i]``.

        Parameters
        ----------
        x, y
            Input vectors to multiply component-wise.

        See Also
        --------
        pointwiseDivide, petsc.VecPointwiseMult

        """
        CHKERR(VecPointwiseMult(self.vec, x.vec, y.vec))

    def pointwiseDivide(self, Vec x, Vec y) -> None:
        """Compute and store the component-wise division of two vectors.

        Logically collective.

        Equivalent to ``w[i] = x[i] / y[i]``.

        Parameters
        ----------
        x
            Numerator vector.
        y
            Denominator vector.

        See Also
        --------
        pointwiseMult, petsc.VecPointwiseDivide

        """
        CHKERR(VecPointwiseDivide(self.vec, x.vec, y.vec))

    def pointwiseMin(self, Vec x, Vec y) -> None:
        """Compute and store the component-wise minimum of two vectors.

        Logically collective.

        Equivalent to ``w[i] = min(x[i], y[i])``.

        Parameters
        ----------
        x, y
            Input vectors to find the component-wise minima.

        See Also
        --------
        pointwiseMax, pointwiseMaxAbs, petsc.VecPointwiseMin

        """
        CHKERR(VecPointwiseMin(self.vec, x.vec, y.vec))

    def pointwiseMax(self, Vec x, Vec y) -> None:
        """Compute and store the component-wise maximum of two vectors.

        Logically collective.

        Equivalent to ``w[i] = max(x[i], y[i])``.

        Parameters
        ----------
        x, y
            Input vectors to find the component-wise maxima.

        See Also
        --------
        pointwiseMin, pointwiseMaxAbs, petsc.VecPointwiseMax

        """
        CHKERR(VecPointwiseMax(self.vec, x.vec, y.vec))

    def pointwiseMaxAbs(self, Vec x, Vec y) -> None:
        """Compute and store the component-wise maximum absolute values.

        Logically collective.

        Equivalent to ``w[i] = max(abs(x[i]), abs(y[i]))``.

        Parameters
        ----------
        x, y
            Input vectors to find the component-wise maxima.

        See Also
        --------
        pointwiseMin, pointwiseMax, petsc.VecPointwiseMaxAbs

        """
        CHKERR(VecPointwiseMaxAbs(self.vec, x.vec, y.vec))

    def maxPointwiseDivide(self, Vec vec) -> float:
        """Return the maximum of the component-wise absolute value division.

        Logically collective.

        Equivalent to ``result = max_i abs(x[i] / y[i])``.

        Parameters
        ----------
        x
            Numerator vector.
        y
            Denominator vector.

        See Also
        --------
        pointwiseMin, pointwiseMax, pointwiseMaxAbs
        petsc.VecMaxPointwiseDivide

        """
        cdef PetscReal rval = 0
        CHKERR(VecMaxPointwiseDivide(self.vec, vec.vec, &rval))
        return toReal(rval)

    def getValue(self, index: int) -> Scalar:
        """Return a single value from the vector.

        Not collective.

        Only values locally stored may be accessed.

        Parameters
        ----------
        index
            Location of the value to read.

        See Also
        --------
        getValues, petsc.VecGetValues

        """
        cdef PetscInt    ival = asInt(index)
        cdef PetscScalar sval = 0
        CHKERR(VecGetValues(self.vec, 1, &ival, &sval))
        return toScalar(sval)

    def getValues(
        self,
        indices: Sequence[int],
        values: Sequence[Scalar] | None = None) -> ArrayScalar:
        """Return values from certain locations in the vector.

        Not collective.

        Only values locally stored may be accessed.

        Parameters
        ----------
        indices
            Locations of the values to read.
        values
            Location to store the collected values. If not provided then a new
            array will be allocated.

        See Also
        --------
        getValue, setValues, petsc.VecGetValues

        """
        return vecgetvalues(self.vec, indices, values)

    def getValuesStagStencil(self, indices, values=None) -> None:
        """Not implemented."""
        raise NotImplementedError

    def setValue(
        self,
        index: int,
        value: Scalar,
        addv: InsertModeSpec = None) -> None:
        """Insert or add a single value in the vector.

        Not collective.

        Parameters
        ----------
        index
            Location to write to. Negative indices are ignored.
        value
            Value to insert at ``index``.
        addv
            Insertion mode.

        Notes
        -----
        The values may be cached so `assemblyBegin` and `assemblyEnd`
        must be called after all calls of this method are completed.

        Multiple calls to `setValue` cannot be made with different values
        for ``addv`` without intermediate calls to `assemblyBegin` and
        `assemblyEnd`.

        See Also
        --------
        setValues, petsc.VecSetValues

        """
        cdef PetscInt    ival = asInt(index)
        cdef PetscScalar sval = asScalar(value)
        cdef PetscInsertMode caddv = insertmode(addv)
        CHKERR(VecSetValues(self.vec, 1, &ival, &sval, caddv))

    def setValues(
        self,
        indices: Sequence[int],
        values: Sequence[Scalar],
        addv: InsertModeSpec = None) -> None:
        """Insert or add multiple values in the vector.

        Not collective.

        Parameters
        ----------
        indices
            Locations to write to. Negative indices are ignored.
        values
            Values to insert at ``indices``.
        addv
            Insertion mode.

        Notes
        -----
        The values may be cached so `assemblyBegin` and `assemblyEnd`
        must be called after all calls of this method are completed.

        Multiple calls to `setValues` cannot be made with different values
        for ``addv`` without intermediate calls to `assemblyBegin` and
        `assemblyEnd`.

        See Also
        --------
        setValue, petsc.VecSetValues

        """
        vecsetvalues(self.vec, indices, values, addv, 0, 0)

    def setValuesBlocked(
        self,
        indices: Sequence[int],
        values: Sequence[Scalar],
        addv: InsertModeSpec = None) -> None:
        """Insert or add blocks of values in the vector.

        Not collective.

        Equivalent to ``x[bs*indices[i]+j] = y[bs*i+j]`` for
        ``0 <= i < len(indices)``, ``0 <= j < bs`` and ``bs`` `block_size`.

        Parameters
        ----------
        indices
            Block indices to write to. Negative indices are ignored.
        values
            Values to insert at ``indices``. Should have length
            ``len(indices) * vec.block_size``.
        addv
            Insertion mode.

        Notes
        -----
        The values may be cached so `assemblyBegin` and `assemblyEnd`
        must be called after all calls of this method are completed.

        Multiple calls to `setValuesBlocked` cannot be made with different
        values for ``addv`` without intermediate calls to `assemblyBegin`
        and `assemblyEnd`.

        See Also
        --------
        setValues, petsc.VecSetValuesBlocked

        """
        vecsetvalues(self.vec, indices, values, addv, 1, 0)

    def setValuesStagStencil(self, indices, values, addv=None) -> None:
        """Not implemented."""
        raise NotImplementedError

    def setLGMap(self, LGMap lgmap) -> None:
        """Set the local-to-global mapping.

        Logically collective.

        This allows users to insert vector entries using a local numbering
        with `setValuesLocal`.

        See Also
        --------
        setValues, setValuesLocal, getLGMap, petsc.VecSetLocalToGlobalMapping

        """
        CHKERR(VecSetLocalToGlobalMapping(self.vec, lgmap.lgm))

    def getLGMap(self) -> LGMap:
        """Return the local-to-global mapping.

        Not collective.

        See Also
        --------
        setLGMap, petsc.VecGetLocalToGlobalMapping

        """
        cdef LGMap cmap = LGMap()
        CHKERR(VecGetLocalToGlobalMapping(self.vec, &cmap.lgm))
        CHKERR(PetscINCREF(cmap.obj))
        return cmap

    def setValueLocal(
        self,
        index: int,
        value: Scalar,
        addv: InsertModeSpec = None) -> None:
        """Insert or add a single value in the vector using a local numbering.

        Not collective.

        Parameters
        ----------
        index
            Location to write to.
        value
            Value to insert at ``index``.
        addv
            Insertion mode.

        Notes
        -----
        The values may be cached so `assemblyBegin` and `assemblyEnd`
        must be called after all calls of this method are completed.

        Multiple calls to `setValueLocal` cannot be made with different
        values for ``addv`` without intermediate calls to `assemblyBegin`
        and `assemblyEnd`.

        See Also
        --------
        setValuesLocal, petsc.VecSetValuesLocal

        """
        cdef PetscInt    ival = asInt(index)
        cdef PetscScalar sval = asScalar(value)
        cdef PetscInsertMode caddv = insertmode(addv)
        CHKERR(VecSetValuesLocal(self.vec, 1, &ival, &sval, caddv))

    def setValuesLocal(
        self,
        indices: Sequence[int],
        values: Sequence[Scalar],
        addv: InsertModeSpec = None) -> None:
        """Insert or add multiple values in the vector with a local numbering.

        Not collective.

        Parameters
        ----------
        indices
            Locations to write to.
        values
            Values to insert at ``indices``.
        addv
            Insertion mode.

        Notes
        -----
        The values may be cached so `assemblyBegin` and `assemblyEnd`
        must be called after all calls of this method are completed.

        Multiple calls to `setValuesLocal` cannot be made with different
        values for ``addv`` without intermediate calls to `assemblyBegin`
        and `assemblyEnd`.

        See Also
        --------
        setValues, petsc.VecSetValuesLocal

        """
        vecsetvalues(self.vec, indices, values, addv, 0, 1)

    def setValuesBlockedLocal(
        self,
        indices: Sequence[int],
        values: Sequence[Scalar],
        addv: InsertModeSpec = None) -> None:
        """Insert or add blocks of values in the vector with a local numbering.

        Not collective.

        Equivalent to ``x[bs*indices[i]+j] = y[bs*i+j]`` for
        ``0 <= i < len(indices)``, ``0 <= j < bs`` and ``bs`` `block_size`.

        Parameters
        ----------
        indices
            Local block indices to write to.
        values
            Values to insert at ``indices``. Should have length
            ``len(indices) * vec.block_size``.
        addv
            Insertion mode.

        Notes
        -----
        The values may be cached so `assemblyBegin` and `assemblyEnd`
        must be called after all calls of this method are completed.

        Multiple calls to `setValuesBlockedLocal` cannot be made with
        different values for ``addv`` without intermediate calls to
        `assemblyBegin` and `assemblyEnd`.

        See Also
        --------
        setValuesBlocked, setValuesLocal, petsc.VecSetValuesBlockedLocal

        """
        vecsetvalues(self.vec, indices, values, addv, 1, 1)

    def assemblyBegin(self) -> None:
        """Begin an assembling stage of the vector.

        Collective.

        See Also
        --------
        assemblyEnd, petsc.VecAssemblyBegin

        """
        CHKERR(VecAssemblyBegin(self.vec))

    def assemblyEnd(self) -> None:
        """Finish the assembling stage initiated with `assemblyBegin`.

        Collective.

        See Also
        --------
        assemblyBegin, petsc.VecAssemblyEnd

        """
        CHKERR(VecAssemblyEnd(self.vec))

    def assemble(self) -> None:
        """Assemble the vector.

        Collective.

        See Also
        --------
        assemblyBegin, assemblyEnd

        """
        CHKERR(VecAssemblyBegin(self.vec))
        CHKERR(VecAssemblyEnd(self.vec))

    # --- methods for strided vectors ---

    def strideScale(self, field: int, alpha: Scalar) -> None:
        """Scale a component of the vector.

        Logically collective.

        Parameters
        ----------
        field
            Component index. Must be between ``0`` and ``vec.block_size``.
        alpha
            Factor to multiple the component entries by.

        See Also
        --------
        strideSum, strideMin, strideMax, petsc.VecStrideScale

        """
        cdef PetscInt    ival = asInt(field)
        cdef PetscScalar sval = asScalar(alpha)
        CHKERR(VecStrideScale(self.vec, ival, sval))

    def strideSum(self, field: int) -> Scalar:
        """Sum subvector entries.

        Collective.

        Equivalent to ``sum(x[field], x[field+bs], x[field+2*bs], ...)`` where
        ``bs`` is `block_size`.

        Parameters
        ----------
        field
            Component index. Must be between ``0`` and ``vec.block_size``.

        See Also
        --------
        strideScale, strideMin, strideMax, petsc.VecStrideSum

        """
        cdef PetscInt    ival = asInt(field)
        cdef PetscScalar sval = 0
        CHKERR(VecStrideSum(self.vec, ival, &sval))
        return toScalar(sval)

    def strideMin(self, field: int) -> tuple[int, float]:
        """Return the minimum of entries in a subvector.

        Collective.

        Equivalent to ``min(x[field], x[field+bs], x[field+2*bs], ...)`` where
        ``bs`` is `block_size`.

        Parameters
        ----------
        field
            Component index. Must be between ``0`` and ``vec.block_size``.

        Returns
        -------
        int
            Location of minimum.
        float
            Minimum value.

        See Also
        --------
        strideScale, strideSum, strideMax, petsc.VecStrideMin

        """
        cdef PetscInt  ival1 = asInt(field)
        cdef PetscInt  ival2 = 0
        cdef PetscReal rval  = 0
        CHKERR(VecStrideMin(self.vec, ival1, &ival2, &rval))
        return (toInt(ival2), toReal(rval))

    def strideMax(self, field: int) -> tuple[int, float]:
        """Return the maximum of entries in a subvector.

        Collective.

        Equivalent to ``max(x[field], x[field+bs], x[field+2*bs], ...)`` where
        ``bs`` is `block_size`.

        Parameters
        ----------
        field
            Component index. Must be between ``0`` and ``vec.block_size``.

        Returns
        -------
        int
            Location of maximum.
        float
            Maximum value.

        See Also
        --------
        strideScale, strideSum, strideMin, petsc.VecStrideMax

        """
        cdef PetscInt  ival1 = asInt(field)
        cdef PetscInt  ival2 = 0
        cdef PetscReal rval  = 0
        CHKERR(VecStrideMax(self.vec, ival1, &ival2, &rval))
        return (toInt(ival2), toReal(rval))

    def strideNorm(
        self,
        field: int,
        norm_type: NormTypeSpec = None) -> float | tuple[float, float]:
        """Return the norm of entries in a subvector.

        Collective.

        Equivalent to ``norm(x[field], x[field+bs], x[field+2*bs], ...)`` where
        ``bs`` is `block_size`.

        Parameters
        ----------
        field
            Component index. Must be between ``0`` and ``vec.block_size``.
        norm_type
            The norm type.

        See Also
        --------
        norm, strideScale, strideSum, petsc.VecStrideNorm

        """
        cdef PetscInt ival = asInt(field)
        cdef PetscNormType norm_1_2 = PETSC_NORM_1_AND_2
        cdef PetscNormType ntype = PETSC_NORM_2
        if norm_type is not None: ntype = norm_type
        cdef PetscReal rval[2]
        CHKERR(VecStrideNorm(self.vec, ival, ntype, rval))
        if ntype != norm_1_2: return toReal(rval[0])
        else: return (toReal(rval[0]), toReal(rval[1]))

    def strideScatter(
        self,
        field: int,
        Vec vec,
        addv: InsertModeSpec = None) -> None:
        """Scatter entries into a component of another vector.

        Collective.

        The current vector is expected to be single-component
        (`block_size` of ``1``) and the target vector is expected to be
        multi-component.

        Parameters
        ----------
        field
            Component index. Must be between ``0`` and ``vec.block_size``.
        vec
            Multi-component vector to be scattered into.
        addv
            Insertion mode.

        See Also
        --------
        strideGather, petsc.VecStrideScatter

        """
        cdef PetscInt ival = asInt(field)
        cdef PetscInsertMode caddv = insertmode(addv)
        CHKERR(VecStrideScatter(self.vec, ival, vec.vec, caddv))

    def strideGather(
        self,
        field: int,
        Vec vec,
        addv: InsertModeSpec = None) -> None:
        """Insert component values into a single-component vector.

        Collective.

        The current vector is expected to be multi-component (`block_size`
        greater than ``1``) and the target vector is expected to be
        single-component.

        Parameters
        ----------
        field
            Component index. Must be between ``0`` and ``vec.block_size``.
        vec
            Single-component vector to be inserted into.
        addv
            Insertion mode.

        See Also
        --------
        strideScatter, petsc.VecStrideScatter

        """
        cdef PetscInt ival = asInt(field)
        cdef PetscInsertMode caddv = insertmode(addv)
        CHKERR(VecStrideGather(self.vec, ival, vec.vec, caddv))

    # --- methods for vectors with ghost values ---

    def localForm(self) -> Any:
        """Return a context manager for viewing ghost vectors in local form.

        Logically collective.

        Returns
        -------
        typing.Any
            Context manager yielding the vector in local (ghosted) form.

        Notes
        -----
        This operation does not perform a copy. To obtain up-to-date ghost
        values `ghostUpdateBegin` and `ghostUpdateEnd` must be called
        first.

        Non-ghost values can be found
        at ``values[0:nlocal]`` and ghost values at
        ``values[nlocal:nlocal+nghost]``.

        Examples
        --------
        >>> with vec.localForm() as lf:
        ...     # compute with lf

        See Also
        --------
        createGhost, ghostUpdateBegin, ghostUpdateEnd
        petsc.VecGhostGetLocalForm, petsc.VecGhostRestoreLocalForm

        """
        return _Vec_LocalForm(self)

    def ghostUpdateBegin(
        self,
        addv: InsertModeSpec = None,
        mode: ScatterModeSpec = None) -> None:
        """Begin updating ghosted vector entries.

        Neighborwise collective.

        See Also
        --------
        ghostUpdateEnd, ghostUpdate, createGhost, petsc.VecGhostUpdateBegin

        """
        cdef PetscInsertMode  caddv = insertmode(addv)
        cdef PetscScatterMode csctm = scattermode(mode)
        CHKERR(VecGhostUpdateBegin(self.vec, caddv, csctm))

    def ghostUpdateEnd(
        self,
        addv: InsertModeSpec = None,
        mode: ScatterModeSpec = None) -> None:
        """Finish updating ghosted vector entries initiated with `ghostUpdateBegin`.

        Neighborwise collective.

        See Also
        --------
        ghostUpdateBegin, ghostUpdate, createGhost, petsc.VecGhostUpdateEnd

        """
        cdef PetscInsertMode  caddv = insertmode(addv)
        cdef PetscScatterMode csctm = scattermode(mode)
        CHKERR(VecGhostUpdateEnd(self.vec, caddv, csctm))

    def ghostUpdate(
        self,
        addv: InsertModeSpec = None,
        mode: ScatterModeSpec = None) -> None:
        """Update ghosted vector entries.

        Neighborwise collective.

        Parameters
        ----------
        addv
            Insertion mode.
        mode
            Scatter mode.

        Examples
        --------
        To accumulate ghost region values onto owning processes:

        >>> vec.ghostUpdate(InsertMode.ADD_VALUES, ScatterMode.REVERSE)

        Update ghost regions:

        >>> vec.ghostUpdate(InsertMode.INSERT_VALUES, ScatterMode.FORWARD)

        See Also
        --------
        ghostUpdateBegin, ghostUpdateEnd

        """
        cdef PetscInsertMode  caddv = insertmode(addv)
        cdef PetscScatterMode csctm = scattermode(mode)
        CHKERR(VecGhostUpdateBegin(self.vec, caddv, csctm))
        CHKERR(VecGhostUpdateEnd(self.vec, caddv, csctm))

    def setMPIGhost(self, ghosts: Sequence[int]) -> None:
        """Set the ghost points for a ghosted vector.

        Collective.

        Parameters
        ----------
        ghosts
            Global indices of ghost points.

        See Also
        --------
        createGhost

        """
        cdef PetscInt ng=0, *ig=NULL
        ghosts = iarray_i(ghosts, &ng, &ig)
        CHKERR(VecMPISetGhost(self.vec, ng, ig))

    def getGhostIS(self) -> IS:
        """Return ghosting indices of a ghost vector.

        Collective.

        Returns
        -------
        IS
            Indices of ghosts.

        See Also
        --------
        petsc.VecGhostGetGhostIS

        """
        cdef PetscIS indices = NULL
        CHKERR(VecGhostGetGhostIS(self.vec, &indices))
        return ref_IS(indices)

    #

    def getSubVector(self, IS iset, Vec subvec=None) -> Vec:
        """Return a subvector from given indices.

        Collective.

        Once finished with the subvector it should be returned with
        `restoreSubVector`.

        Parameters
        ----------
        iset
            Index set describing which indices to extract into the subvector.
        subvec
            Subvector to copy entries into. If `None` then a new `Vec` will
            be created.

        See Also
        --------
        restoreSubVector, petsc.VecGetSubVector

        """
        if subvec is None: subvec = Vec()
        else: CHKERR(VecDestroy(&subvec.vec))
        CHKERR(VecGetSubVector(self.vec, iset.iset, &subvec.vec))
        return subvec

    def restoreSubVector(self, IS iset, Vec subvec) -> None:
        """Restore a subvector extracted using `getSubVector`.

        Collective.

        Parameters
        ----------
        iset
            Index set describing the indices represented by the subvector.
        subvec
            Subvector to be restored.

        See Also
        --------
        getSubVector, petsc.VecRestoreSubVector

        """
        CHKERR(VecRestoreSubVector(self.vec, iset.iset, &subvec.vec))

    def getNestSubVecs(self) -> list[Vec]:
        """Return all the vectors contained in the nested vector.

        Not collective.

        See Also
        --------
        setNestSubVecs, petsc.VecNestGetSubVecs

        """
        cdef PetscInt N=0
        cdef PetscVec* sx=NULL
        CHKERR(VecNestGetSubVecs(self.vec, &N, &sx))
        output = []
        for i in range(N):
            pyvec = Vec()
            pyvec.vec = sx[i]
            CHKERR(PetscObjectReference(<PetscObject> pyvec.vec))
            output.append(pyvec)

        return output

    def setNestSubVecs(
        self,
        sx: Sequence[Vec],
        idxm: Sequence[int] | None = None) -> None:
        """Set the component vectors at specified indices in the nested vector.

        Not collective.

        Parameters
        ----------
        sx
            Array of component vectors.
        idxm
            Indices of the component vectors, defaults to ``range(len(sx))``.

        See Also
        --------
        getNestSubVecs, petsc.VecNestSetSubVecs

        """
        if idxm is None: idxm = range(len(sx))
        else: assert len(idxm) == len(sx)
        cdef PetscInt N = 0
        cdef PetscInt* cidxm = NULL
        idxm = iarray_i(idxm, &N, &cidxm)

        cdef PetscVec* csx = NULL
        cdef object unused = oarray_p(empty_p(N), NULL, <void**>&csx)
        for i from 0 <= i < N: csx[i] = (<Vec?>sx[i]).vec

        CHKERR(VecNestSetSubVecs(self.vec, N, cidxm, csx))

    #

    def setDM(self, DM dm) -> None:
        """Associate a `DM` to the vector.

        Not collective.

        See Also
        --------
        getDM, petsc.VecSetDM

        """
        CHKERR(VecSetDM(self.vec, dm.dm))

    def getDM(self) -> DM:
        """Return the `DM` associated to the vector.

        Not collective.

        See Also
        --------
        setDM, petsc.VecGetDM

        """
        cdef PetscDM newdm = NULL
        CHKERR(VecGetDM(self.vec, &newdm))
        cdef DM dm = subtype_DM(newdm)()
        dm.dm = newdm
        CHKERR(PetscINCREF(dm.obj))
        return dm

    #

    @classmethod
    def concatenate(cls, vecs: Sequence[Vec]) -> tuple[Vec, list[IS]]:
        """Concatenate vectors into a single vector.

        Collective.

        Parameters
        ----------
        vecs
            The vectors to be concatenated.

        Returns
        -------
        vector_out : Vec
            The concatenated vector.
        indices_list : list of IS
            A list of index sets corresponding to the concatenated components.

        See Also
        --------
        petsc.VecConcatenate

        """
        vecs = list(vecs)
        cdef Py_ssize_t i, m = len(vecs)
        cdef PetscInt n = <PetscInt>m
        cdef PetscVec newvec = NULL
        cdef PetscVec *cvecs  = NULL
        cdef PetscIS  *cisets = NULL
        cdef object unused1
        cdef object vec_index_ises = []
        unused1 = oarray_p(empty_p(n), NULL, <void**>&cvecs)
        for i from 0 <= i < m:
            cvecs[i] = (<Vec?>vecs[i]).vec
        CHKERR(VecConcatenate(n, cvecs, &newvec, &cisets))
        cdef Vec self = cls()
        self.vec = newvec
        for i from 0 <= i < m:
            temp = IS()
            temp.iset = cisets[i]
            vec_index_ises.append(temp)
        CHKERR(PetscFree(cisets))
        return self, vec_index_ises

    property sizes:
        """The local and global vector sizes."""
        def __get__(self) -> LayoutSizeSpec:
            return self.getSizes()

        def __set__(self, value):
            self.setSizes(value)

    property size:
        """The global vector size."""
        def __get__(self) -> int:
            return self.getSize()

    property local_size:
        """The local vector size."""
        def __get__(self) -> int:
            return self.getLocalSize()

    property block_size:
        """The block size."""
        def __get__(self) -> int:
            return self.getBlockSize()

    property owner_range:
        """The locally owned range of indices in the form ``[low, high)``."""
        def __get__(self) -> tuple[int, int]:
            return self.getOwnershipRange()

    property owner_ranges:
        """The range of indices owned by each process."""
        def __get__(self) -> ArrayInt:
            return self.getOwnershipRanges()

    property buffer_w:
        """Writeable buffered view of the local portion of the vector."""
        def __get__(self) -> Any:
            return self.getBuffer()

    property buffer_r:
        """Read-only buffered view of the local portion of the vector."""
        def __get__(self) -> Any:
            return self.getBuffer(True)

    property array_w:
        """Writeable `ndarray` containing the local portion of the vector."""
        def __get__(self) -> ArrayScalar:
            return self.getArray()

        def __set__(self, value):
            cdef buf = self.getBuffer()
            with buf as array: array[:] = value

    property array_r:
        """Read-only `ndarray` containing the local portion of the vector."""
        def __get__(self) -> ArrayScalar:
            return self.getArray(True)

    property buffer:
        """Alias for `buffer_w`."""
        def __get__(self) -> Any:
            return self.buffer_w

    property array:
        """Alias for `array_w`."""
        def __get__(self) -> ArrayScalar:
            return self.array_w

        def __set__(self, value):
            self.array_w = value

    # --- NumPy array interface (legacy) ---

    property __array_interface__:
        def __get__(self):
            cdef buf = self.getBuffer()
            return buf.__array_interface__

# --------------------------------------------------------------------

del VecType
del VecOption

# --------------------------------------------------------------------
