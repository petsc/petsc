# --------------------------------------------------------------------

class VecType(object):
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
    IGNORE_OFF_PROC_ENTRIES = VEC_IGNORE_OFF_PROC_ENTRIES
    IGNORE_NEGATIVE_INDICES = VEC_IGNORE_NEGATIVE_INDICES

# --------------------------------------------------------------------

cdef class Vec(Object):

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
        if isinstance(self, Vec):
            return vec_add(self, other)
        else:
            return vec_radd(other, self)

    def __sub__(self, other):
        if isinstance(self, Vec):
            return vec_sub(self, other)
        else:
            return vec_rsub(other, self)

    def __mul__(self, other):
        if isinstance(self, Vec):
            return vec_mul(self, other)
        else:
            return vec_rmul(other, self)

    def __div__(self, other):
        if isinstance(self, Vec):
            return vec_div(self, other)
        else:
            return vec_rdiv(other, self)

    def __truediv__(self, other):
        if isinstance(self, Vec):
            return vec_div(self, other)
        else:
            return vec_rdiv(other, self)

    #

    #def __len__(self):
    #    cdef PetscInt size = 0
    #    CHKERR( VecGetSize(self.vec, &size) )
    #    return <Py_ssize_t>size

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

    def view(self, Viewer viewer=None):
        cdef PetscViewer vwr = NULL
        if viewer is not None: vwr = viewer.vwr
        CHKERR( VecView(self.vec, vwr) )

    def destroy(self):
        CHKERR( VecDestroy(&self.vec) )
        return self

    def create(self, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscVec newvec = NULL
        CHKERR( VecCreate(ccomm, &newvec) )
        PetscCLEAR(self.obj); self.vec = newvec
        return self

    def setType(self, vec_type):
        cdef PetscVecType cval = NULL
        vec_type = str2bytes(vec_type, &cval)
        CHKERR( VecSetType(self.vec, cval) )

    def setSizes(self, size, bsize=None):
        cdef PetscInt bs=0, n=0, N=0
        Vec_Sizes(size, bsize, &bs, &n, &N)
        CHKERR( VecSetSizes(self.vec, n, N) )
        if bs != PETSC_DECIDE:
            CHKERR( VecSetBlockSize(self.vec, bs) )

    #

    def createSeq(self, size, bsize=None, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_SELF)
        cdef PetscInt bs=0, n=0, N=0
        Vec_Sizes(size, bsize, &bs, &n, &N)
        Sys_Layout(ccomm, bs, &n, &N)
        if bs == PETSC_DECIDE: bs = 1
        cdef PetscVec newvec = NULL
        CHKERR( VecCreate(ccomm,&newvec) )
        CHKERR( VecSetSizes(newvec, n, N) )
        CHKERR( VecSetBlockSize(newvec, bs) )
        CHKERR( VecSetType(newvec, VECSEQ) )
        PetscCLEAR(self.obj); self.vec = newvec
        return self

    def createMPI(self, size, bsize=None, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscInt bs=0, n=0, N=0
        Vec_Sizes(size, bsize, &bs, &n, &N)
        Sys_Layout(ccomm, bs, &n, &N)
        if bs == PETSC_DECIDE: bs = 1
        cdef PetscVec newvec = NULL
        CHKERR( VecCreate(ccomm, &newvec) )
        CHKERR( VecSetSizes(newvec, n, N) )
        CHKERR( VecSetBlockSize(newvec, bs) )
        CHKERR( VecSetType(newvec, VECMPI) )
        PetscCLEAR(self.obj); self.vec = newvec
        return self

    def createWithArray(self, array, size=None, bsize=None, comm=None):
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
            CHKERR( VecCreateSeqWithArray(ccomm,bs,N,sa,&newvec) )
        else:
            CHKERR( VecCreateMPIWithArray(ccomm,bs,n,N,sa,&newvec) )
        PetscCLEAR(self.obj); self.vec = newvec
        self.set_attr('__array__', array)
        return self

    def createCUDAWithArrays(self, cpuarray=None, cudahandle=None, size=None, bsize=None, comm=None):
        """
        Returns an instance of :class:`Vec`, a VECCUDA with user provided
        memory spaces for CPU and GPU arrays.

        :arg cpuarray: A :class:`numpy.ndarray`. Will be lazily allocated if
            *None*.
        :arg cudahandle: Address of the array on the GPU. Will be lazily
            allocated if *None*.
        :arg size: A :class:`int` denoting the size of the Vec.
        :arg bsize: A :class:`int` denoting the block size.
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
            CHKERR( VecCreateSeqCUDAWithArrays(ccomm,bs,N,sa,gpuarray,&newvec) )
        else:
            CHKERR( VecCreateMPICUDAWithArrays(ccomm,bs,n,N,sa,gpuarray,&newvec) )
        PetscCLEAR(self.obj); self.vec = newvec

        if cpuarray is not None:
            self.set_attr('__array__', cpuarray)
        return self

    def createHIPWithArrays(self, cpuarray=None, hiphandle=None, size=None, bsize=None, comm=None):
        """
        Returns an instance of :class:`Vec`, a VECHIP with user provided
        memory spaces for CPU and GPU arrays.

        :arg cpuarray: A :class:`numpy.ndarray`. Will be lazily allocated if
            *None*.
        :arg hiphandle: Address of the array on the GPU. Will be lazily
            allocated if *None*.
        :arg size: A :class:`int` denoting the size of the Vec.
        :arg bsize: A :class:`int` denoting the block size.
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
            CHKERR( VecCreateSeqHIPWithArrays(ccomm,bs,N,sa,gpuarray,&newvec) )
        else:
            CHKERR( VecCreateMPIHIPWithArrays(ccomm,bs,n,N,sa,gpuarray,&newvec) )
        PetscCLEAR(self.obj); self.vec = newvec

        if cpuarray is not None:
            self.set_attr('__array__', cpuarray)
        return self

    def createViennaCLWithArrays(self, cpuarray=None, viennaclvechandle=None, size=None, bsize=None, comm=None):
        """
        Returns an instance :class:`Vec`, a VECVIENNACL with user provided memory
        spaces for CPU and GPU arrays.

        :arg cpuarray: A :class:`numpy.ndarray`. Will be lazily allocated if
            *None*.
        :arg viennaclvechandle: Address of the array on the GPU. Will be lazily
            allocated if *None*.
        :arg size: A :class:`int` denoting the size of the Vec.
        :arg size: A :class:`int` denoting the block size.
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
        if na < n:  raise ValueError( "array size %d and vector local size %d block size %d" % (toInt(na), toInt(n), toInt(bs)))
        cdef PetscVec newvec = NULL
        if comm_size(ccomm) == 1:
            CHKERR( VecCreateSeqViennaCLWithArrays(ccomm,bs,N,sa,vclvec,&newvec) )
        else:
            CHKERR( VecCreateMPIViennaCLWithArrays(ccomm,bs,n,N,sa,vclvec,&newvec) )
        PetscCLEAR(self.obj); self.vec = newvec

        if cpuarray is not None:
            self.set_attr('__array__', cpuarray)
        return self

    def createWithDLPack(self, object dltensor, size=None, bsize=None, comm=None):
        """
        Returns an instance :class:`Vec`, a PETSc vector from a DLPack object
        sharing the same memory.
        This operation does not modify the storage of the original tensor and
        should be used with contiguous tensors only. If the tensor is stored in
        row-major order (e.g. PyTorch tensors), the resulting vector will look
        like an unrolled tensor using row-major order.

        :arg dltensor: An object with a __dlpack__ method or a DLPack tensor object (for backward compatibility)
        :arg size: A :class:`int` denoting the size of the Vec.
        :arg bsize: A :class:`int` denoting the block size.
        """
        cdef DLManagedTensor* ptr = NULL
        cdef int bits = 0
        cdef PetscInt nz = 1
        cdef int64_t ndim = 0
        cdef int64_t* shape = NULL
        cdef int64_t* strides = NULL
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscInt bs = 0,n = 0,N = 0
        cdef DLContext* ctx = NULL

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
        if dltype in [kDLCUDA,kDLCUDAManaged]:
            if comm_size(ccomm) == 1:
                CHKERR( VecCreateSeqCUDAWithArray(ccomm,bs,N,<PetscScalar*>(ptr.dl_tensor.data),&newvec) )
            else:
                CHKERR( VecCreateMPICUDAWithArray(ccomm,bs,n,N,<PetscScalar*>(ptr.dl_tensor.data),&newvec) )
        elif dltype in [kDLCPU,kDLCUDAHost,kDLROCMHost]:
            if comm_size(ccomm) == 1:
                CHKERR( VecCreateSeqWithArray(ccomm,bs,N,<PetscScalar*>(ptr.dl_tensor.data),&newvec) )
            else:
                CHKERR( VecCreateMPIWithArray(ccomm,bs,n,N,<PetscScalar*>(ptr.dl_tensor.data),&newvec) )
        elif dltype == kDLROCM:
            if comm_size(ccomm) == 1:
                CHKERR( VecCreateSeqHIPWithArray(ccomm,bs,N,<PetscScalar*>(ptr.dl_tensor.data),&newvec) )
            else:
                CHKERR( VecCreateMPIHIPWithArray(ccomm,bs,n,N,<PetscScalar*>(ptr.dl_tensor.data),&newvec) )
        else:
            raise TypeError("Device type {} not supported".format(dltype))

        PetscCLEAR(self.obj); self.vec = newvec
        self.set_attr('__array__', dltensor)
        cdef int64_t* shape_arr = NULL
        cdef int64_t* strides_arr = NULL
        cdef object s1 = oarray_p(empty_p(ndim), NULL, <void**>&shape_arr)
        cdef object s2 = oarray_p(empty_p(ndim), NULL, <void**>&strides_arr)
        for i in range(ndim):
            shape_arr[i] = shape[i]
            strides_arr[i] = strides[i]
        self.set_attr('__dltensor_ctx__', (ptr.dl_tensor.ctx.device_type, ptr.dl_tensor.ctx.device_id, ndim, s1, s2))
        if ptr.manager_deleter != NULL:
            ptr.manager_deleter(ptr) # free the manager
        return self

    def attachDLPackInfo(self, Vec vec=None, object dltensor=None):
        """
        Attach the tensor information from an input vector (vec) or a
        DLPack tensor if it is not available in current vector. The input
        vector is typically created with createWithDlpack(). This operation
        does not copy the data from the input parameters, it simply uses
        their meta information.

        Note that the auxiliary tensor information is required when converting
        a PETSc vector to a DLPack object.

        See also :meth:`Vec.clearDLPackInfo`.

        :arg vec: A :class:'Vec' containing auxiliary tensor information
        :arg dltensor: A DLPack tensor object
        """
        cdef object ctx0 = self.get_attr('__dltensor_ctx__'), ctx = None
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

    def clearDLPackInfo(self):
        """
        Clear the tensor information
        See also :meth:`Vec.attachDLPackInfo`.
        """
        self.set_attr('__dltensor_ctx__', None)
        return self

    # TODO Stream
    def __dlpack__(self, stream=-1):
        return self.toDLPack('rw')

    def __dlpack_device__(self):
        (dltype, devId, _, _, _) = vec_get_dlpack_ctx(self)
        return (dltype, devId)

    def toDLPack(self, mode='rw'):
        """
        Return a DLPack capsule.
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
                CHKERR( VecGetArrayWrite(self.vec, <PetscScalar**>&a) )
                CHKERR( VecRestoreArrayWrite(self.vec, NULL) )
            else:
                CHKERR( VecGetArrayWriteAndMemType(self.vec, <PetscScalar**>&a, NULL) )
                CHKERR( VecRestoreArrayWriteAndMemType(self.vec, NULL) )
        elif mode == 'r':
            if hostmem:
                CHKERR( VecGetArrayRead(self.vec, <const PetscScalar**>&a) )
                CHKERR( VecRestoreArrayRead(self.vec, NULL) )
            else:
                CHKERR( VecGetArrayReadAndMemType(self.vec, <const PetscScalar**>&a, NULL) )
                CHKERR( VecRestoreArrayReadAndMemType(self.vec, NULL) )
        else:
            if hostmem:
                CHKERR( VecGetArray(self.vec, <PetscScalar**>&a) )
                CHKERR( VecRestoreArray(self.vec, NULL) )
            else:
                CHKERR( VecGetArrayAndMemType(self.vec, <PetscScalar**>&a, NULL) )
                CHKERR( VecRestoreArrayAndMemType(self.vec, NULL) )
        dl_tensor.data = <void *>a

        cdef DLContext* ctx = &dl_tensor.ctx
        ctx.device_type = device_type
        ctx.device_id = device_id
        shape_strides = <int64_t*>malloc(sizeof(int64_t)*2*ndim)
        for i in range(ndim):
            shape_strides[i] = shape[i]
        for i in range(ndim):
            shape_strides[i+ndim] = strides[i]
        dl_tensor.ndim = ndim
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
        CHKERR( PetscObjectReference(<PetscObject>self.vec) )
        dlm_tensor.manager_deleter = manager_deleter
        dlm_tensor.del_obj = <dlpack_manager_del_obj>PetscDEALLOC
        return PyCapsule_New(dlm_tensor, 'dltensor', pycapsule_deleter)

    def createGhost(self, ghosts, size, bsize=None, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscInt ng=0, *ig=NULL
        ghosts = iarray_i(ghosts, &ng, &ig)
        cdef PetscInt bs=0, n=0, N=0
        Vec_Sizes(size, bsize, &bs, &n, &N)
        Sys_Layout(ccomm, bs, &n, &N)
        cdef PetscVec newvec = NULL
        if bs == PETSC_DECIDE:
            CHKERR( VecCreateGhost(
                    ccomm, n, N, ng, ig, &newvec) )
        else:
            CHKERR( VecCreateGhostBlock(
                    ccomm, bs, n, N, ng, ig, &newvec) )
        PetscCLEAR(self.obj); self.vec = newvec
        return self

    def createGhostWithArray(self, ghosts, array,
                             size=None, bsize=None, comm=None):
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
            CHKERR( VecCreateGhostWithArray(
                    ccomm, n, N, ng, ig, sa, &newvec) )
        else:
            CHKERR( VecCreateGhostBlockWithArray(
                    ccomm, bs, n, N, ng, ig, sa, &newvec) )
        PetscCLEAR(self.obj); self.vec = newvec
        self.set_attr('__array__', array)
        return self

    def createShared(self, size, bsize=None, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscInt bs=0, n=0, N=0
        Vec_Sizes(size, bsize, &bs, &n, &N)
        Sys_Layout(ccomm, bs, &n, &N)
        cdef PetscVec newvec = NULL
        CHKERR( VecCreateShared(ccomm, n, N, &newvec) )
        PetscCLEAR(self.obj); self.vec = newvec
        if bs != PETSC_DECIDE:
            CHKERR( VecSetBlockSize(self.vec, bs) )
        return self

    def createNest(self, vecs, isets=None, comm=None):
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
        cdef object tmp1, tmp2
        tmp1 = oarray_p(empty_p(n), NULL, <void**>&cvecs)
        for i from 0 <= i < m: cvecs[i] = (<Vec?>vecs[i]).vec
        if isets is not None:
            tmp2 = oarray_p(empty_p(n), NULL, <void**>&cisets)
            for i from 0 <= i < m: cisets[i] = (<IS?>isets[i]).iset
        cdef PetscVec newvec = NULL
        CHKERR( VecCreateNest(ccomm, n, cisets, cvecs,&newvec) )
        PetscCLEAR(self.obj); self.vec = newvec
        return self

    #

    def setOptionsPrefix(self, prefix):
        cdef const char *cval = NULL
        prefix = str2bytes(prefix, &cval)
        CHKERR( VecSetOptionsPrefix(self.vec, cval) )

    def getOptionsPrefix(self):
        cdef const char *cval = NULL
        CHKERR( VecGetOptionsPrefix(self.vec, &cval) )
        return bytes2str(cval)

    def appendOptionsPrefix(self, prefix):
        cdef const char *cval = NULL
        prefix = str2bytes(prefix, &cval)
        CHKERR( VecAppendOptionsPrefix(self.vec, cval) )

    def setFromOptions(self):
        CHKERR( VecSetFromOptions(self.vec) )

    def setUp(self):
        CHKERR( VecSetUp(self.vec) )
        return self

    def setOption(self, option, flag):
        CHKERR( VecSetOption(self.vec, option, flag) )

    def getType(self):
        cdef PetscVecType cval = NULL
        CHKERR( VecGetType(self.vec, &cval) )
        return bytes2str(cval)

    def getSize(self):
        cdef PetscInt N = 0
        CHKERR( VecGetSize(self.vec, &N) )
        return toInt(N)

    def getLocalSize(self):
        cdef PetscInt n = 0
        CHKERR( VecGetLocalSize(self.vec, &n) )
        return toInt(n)

    def getSizes(self):
        cdef PetscInt n = 0, N = 0
        CHKERR( VecGetLocalSize(self.vec, &n) )
        CHKERR( VecGetSize(self.vec, &N) )
        return (toInt(n), toInt(N))

    def setBlockSize(self, bsize):
        cdef PetscInt bs = asInt(bsize)
        CHKERR( VecSetBlockSize(self.vec, bs) )

    def getBlockSize(self):
        cdef PetscInt bs=0
        CHKERR( VecGetBlockSize(self.vec, &bs) )
        return toInt(bs)

    def getOwnershipRange(self):
        cdef PetscInt low=0, high=0
        CHKERR( VecGetOwnershipRange(self.vec, &low, &high) )
        return (toInt(low), toInt(high))

    def getOwnershipRanges(self):
        cdef const PetscInt *rng = NULL
        CHKERR( VecGetOwnershipRanges(self.vec, &rng) )
        cdef MPI_Comm comm = MPI_COMM_NULL
        CHKERR( PetscObjectGetComm(<PetscObject>self.vec, &comm) )
        cdef int size = -1
        CHKERR( MPI_Comm_size(comm, &size) )
        return array_i(size+1, rng)

    def getBuffer(self, readonly=False):
        if readonly:
            return vec_getbuffer_r(self)
        else:
            return vec_getbuffer_w(self)

    def getArray(self, readonly=False):
        if readonly:
            return vec_getarray_r(self)
        else:
            return vec_getarray_w(self)

    def setArray(self, array):
        vec_setarray(self, array)

    def placeArray(self, array):
        cdef PetscInt nv=0
        cdef PetscInt na=0
        cdef PetscScalar *a = NULL
        CHKERR( VecGetLocalSize(self.vec, &nv) )
        array = oarray_s(array, &na, &a)
        if (na != nv): raise ValueError(
            "cannot place input array size %d, vector size %d" %
            (toInt(na), toInt(nv)))
        CHKERR( VecPlaceArray(self.vec, a) )
        self.set_attr('__placed_array__', array)

    def resetArray(self, force=False):
        cdef object array = None
        array = self.get_attr('__placed_array__')
        if array is None and not force: return None
        CHKERR( VecResetArray(self.vec) )
        self.set_attr('__placed_array__', None)
        return array

    def bindToCPU(self, flg):
        """
        If *flg* is *True*, all subsequent operations of *self* would be
        performed on CPU. If *flg* is *False*, all subsequent operations of
        *self* would be offloaded to the device, provided that the VecType is
        capable of offloading.

        :arg flg: An instance of :class:`bool`.
        """
        cdef PetscBool bindFlg = asBool(flg)
        CHKERR( VecBindToCPU(self.vec, bindFlg) )

    def boundToCPU(self):
        cdef PetscBool flg = PETSC_TRUE
        CHKERR( VecBoundToCPU(self.vec, &flg) )
        return toBool(flg)

    def getCUDAHandle(self, mode='rw'):
        cdef PetscScalar *hdl = NULL
        cdef const char *m = NULL
        if mode is not None: mode = str2bytes(mode, &m)
        if m == NULL or (m[0] == c'r' and m[1] == c'w'):
            CHKERR( VecCUDAGetArray(self.vec, &hdl) )
        elif m[0] == c'r':
            CHKERR( VecCUDAGetArrayRead(self.vec, <const PetscScalar**>&hdl) )
        elif m[0] == c'w':
            CHKERR( VecCUDAGetArrayWrite(self.vec, &hdl) )
        else:
            raise ValueError("Invalid mode: expected 'rw', 'r', or 'w'")
        return <Py_uintptr_t>hdl

    def restoreCUDAHandle(self, handle, mode='rw'):
        cdef PetscScalar *hdl = <PetscScalar*>(<Py_uintptr_t>handle)
        cdef const char *m = NULL
        if mode is not None: mode = str2bytes(mode, &m)
        if m == NULL or (m[0] == c'r' and m[1] == c'w'):
            CHKERR( VecCUDARestoreArray(self.vec, &hdl) )
        elif m[0] == c'r':
            CHKERR( VecCUDARestoreArrayRead(self.vec, <const PetscScalar**>&hdl) )
        elif m[0] == c'w':
            CHKERR( VecCUDARestoreArrayWrite(self.vec, &hdl) )
        else:
            raise ValueError("Invalid mode: expected 'rw', 'r', or 'w'")

    def getHIPHandle(self, mode='rw'):
        cdef PetscScalar *hdl = NULL
        cdef const char *m = NULL
        if mode is not None: mode = str2bytes(mode, &m)
        if m == NULL or (m[0] == c'r' and m[1] == c'w'):
            CHKERR( VecHIPGetArray(self.vec, &hdl) )
        elif m[0] == c'r':
            CHKERR( VecHIPGetArrayRead(self.vec, <const PetscScalar**>&hdl) )
        elif m[0] == c'w':
            CHKERR( VecHIPGetArrayWrite(self.vec, &hdl) )
        else:
            raise ValueError("Invalid mode: expected 'rw', 'r', or 'w'")
        return <Py_uintptr_t>hdl

    def restoreHIPHandle(self, handle, mode='rw'):
        cdef PetscScalar *hdl = <PetscScalar*>(<Py_uintptr_t>handle)
        cdef const char *m = NULL
        if mode is not None: mode = str2bytes(mode, &m)
        if m == NULL or (m[0] == c'r' and m[1] == c'w'):
            CHKERR( VecHIPRestoreArray(self.vec, &hdl) )
        elif m[0] == c'r':
            CHKERR( VecHIPRestoreArrayRead(self.vec, <const PetscScalar**>&hdl) )
        elif m[0] == c'w':
            CHKERR( VecHIPRestoreArrayWrite(self.vec, &hdl) )
        else:
            raise ValueError("Invalid mode: expected 'rw', 'r', or 'w'")

    def getOffloadMask(self):
        """
        Returns :class:`int` of the Vec's PetscOffloadMask enum value.
        """
        cdef PetscOffloadMask mask
        CHKERR( VecGetOffloadMask(self.vec, &mask) )
        return mask

    def getCLContextHandle(self):
        """
        Returns a Vec's CL Context as :class:`int`.

        To interface with :mod:`pyopencl` refer
        :meth:`pyopencl.Context.from_int_ptr`
        """
        cdef Py_uintptr_t ctxhdl = 0
        CHKERR( VecViennaCLGetCLContext(self.vec, &ctxhdl) )
        return ctxhdl

    def getCLQueueHandle(self):
        """
        Returns a Vec's CL Context as :class:`int`.

        To interface with :mod:`pyopencl` refer
        :meth:`pyopencl.Context.from_int_ptr`
        """
        cdef Py_uintptr_t queuehdl = 0
        CHKERR( VecViennaCLGetCLQueue(self.vec, &queuehdl) )
        return queuehdl

    def getCLMemHandle(self, mode='rw'):
        """
        Returns a Vec's CL buffer as :class:`int`.

        To interface with :mod:`pyopencl` refer
        :meth:`pyopencl.MemoryObject.from_int_ptr`.

        :arg mode: An instance of class:`str` denoting the intended access
            usage to the CL buffer. Can be one of 'r'(read-only), 'w'
            (write-only) or 'rw' (read-write). See also
            :meth:`Vec.restoreCLMemHandle`.
        """
        cdef Py_uintptr_t memhdl = 0
        cdef const char *m = NULL
        mode = str2bytes(mode, &m)
        if m == NULL or (m[0] == c'r' and m[1] == c'w'):
            CHKERR( VecViennaCLGetCLMem(self.vec, &memhdl) )
        elif m[0] == c'r':
            CHKERR( VecViennaCLGetCLMemRead(self.vec, &memhdl) )
        elif m[0] == c'w':
            CHKERR( VecViennaCLGetCLMemWrite(self.vec, &memhdl) )
        else:
            raise ValueError("Invalid mode: expected 'r', 'w' or 'rw'")
        return memhdl

    def restoreCLMemHandle(self):
        """
        To be called after accessing a Vec's cl_mem in 'w' or 'rw' modes.
        See also :meth:`Vec.getCLMemHandle`.
        """
        CHKERR( VecViennaCLRestoreCLMemWrite(self.vec) )

    def duplicate(self, array=None):
        cdef Vec vec = type(self)()
        CHKERR( VecDuplicate(self.vec, &vec.vec) )
        # duplicate tensor context
        cdef object ctx0 = self.get_attr('__dltensor_ctx__')
        if ctx0 is not None:
            vec.set_attr('__dltensor_ctx__', ctx0)
        if array is not None:
            vec_setarray(vec, array)
        return vec

    def copy(self, Vec result=None):
        if result is None:
            result = type(self)()
        if result.vec == NULL:
            CHKERR( VecDuplicate(self.vec, &result.vec) )
        CHKERR( VecCopy(self.vec, result.vec) )
        return result

    def chop(self, tol):
        cdef PetscReal rval = asReal(tol)
        CHKERR( VecChop(self.vec, rval) )

    def load(self, Viewer viewer):
        cdef MPI_Comm comm = MPI_COMM_NULL
        cdef PetscObject obj = <PetscObject>(viewer.vwr)
        if self.vec == NULL:
            CHKERR( PetscObjectGetComm(obj, &comm) )
            CHKERR( VecCreate(comm, &self.vec) )
        CHKERR( VecLoad(self.vec, viewer.vwr) )
        return self

    def equal(self, Vec vec):
        cdef PetscBool flag = PETSC_FALSE
        CHKERR( VecEqual(self.vec, vec.vec, &flag) )
        return toBool(flag)

    def dot(self, Vec vec):
        cdef PetscScalar sval = 0
        CHKERR( VecDot(self.vec, vec.vec, &sval) )
        return toScalar(sval)

    def dotBegin(self, Vec vec):
        cdef PetscScalar sval = 0
        CHKERR( VecDotBegin(self.vec, vec.vec, &sval) )

    def dotEnd(self, Vec vec):
        cdef PetscScalar sval = 0
        CHKERR( VecDotEnd(self.vec, vec.vec, &sval) )
        return toScalar(sval)

    def tDot(self, Vec vec):
        cdef PetscScalar sval = 0
        CHKERR( VecTDot(self.vec, vec.vec, &sval) )
        return toScalar(sval)

    def tDotBegin(self, Vec vec):
        cdef PetscScalar sval = 0
        CHKERR( VecTDotBegin(self.vec, vec.vec, &sval) )

    def tDotEnd(self, Vec vec):
        cdef PetscScalar sval = 0
        CHKERR( VecTDotEnd(self.vec, vec.vec, &sval) )
        return toScalar(sval)

    def mDot(self, vecs, out=None):
        <void>self; <void>vecs; <void>out; # unused
        raise NotImplementedError

    def mDotBegin(self, vecs, out=None):
        <void>self; <void>vecs; <void>out; # unused
        raise NotImplementedError

    def mDotEnd(self, vecs, out=None):
        <void>self; <void>vecs; <void>out; # unused
        raise NotImplementedError

    def mtDot(self, vecs, out=None):
        <void>self; <void>vecs; <void>out; # unused
        raise NotImplementedError

    def mtDotBegin(self, vecs, out=None):
        <void>self; <void>vecs; <void>out; # unused
        raise NotImplementedError

    def mtDotEnd(self, vecs, out=None):
        <void>self; <void>vecs; <void>out; # unused
        raise NotImplementedError

    def norm(self, norm_type=None):
        cdef PetscNormType norm_1_2 = PETSC_NORM_1_AND_2
        cdef PetscNormType ntype = PETSC_NORM_2
        if norm_type is not None: ntype = norm_type
        cdef PetscReal rval[2]
        CHKERR( VecNorm(self.vec, ntype, rval) )
        if ntype != norm_1_2: return toReal(rval[0])
        else: return (toReal(rval[0]), toReal(rval[1]))

    def normBegin(self, norm_type=None):
        cdef PetscNormType ntype = PETSC_NORM_2
        if norm_type is not None: ntype = norm_type
        cdef PetscReal dummy[2]
        CHKERR( VecNormBegin(self.vec, ntype, dummy) )

    def normEnd(self, norm_type=None):
        cdef PetscNormType norm_1_2 = PETSC_NORM_1_AND_2
        cdef PetscNormType ntype = PETSC_NORM_2
        if norm_type is not None: ntype = norm_type
        cdef PetscReal rval[2]
        CHKERR( VecNormEnd(self.vec, ntype, rval) )
        if ntype != norm_1_2: return toReal(rval[0])
        else: return (toReal(rval[0]), toReal(rval[1]))

    def sum(self):
        cdef PetscScalar sval = 0
        CHKERR( VecSum(self.vec, &sval) )
        return toScalar(sval)

    def min(self):
        cdef PetscInt  ival = 0
        cdef PetscReal rval = 0
        CHKERR( VecMin(self.vec, &ival, &rval) )
        return (toInt(ival), toReal(rval))

    def max(self):
        cdef PetscInt  ival = 0
        cdef PetscReal rval = 0
        CHKERR( VecMax(self.vec, &ival, &rval) )
        return (toInt(ival), toReal(rval))

    def normalize(self):
        cdef PetscReal rval = 0
        CHKERR( VecNormalize(self.vec, &rval) )
        return toReal(rval)

    def reciprocal(self):
        CHKERR( VecReciprocal(self.vec) )

    def exp(self):
        CHKERR( VecExp(self.vec) )

    def log(self):
        CHKERR( VecLog(self.vec) )

    def sqrtabs(self):
        CHKERR( VecSqrtAbs(self.vec) )

    def abs(self):
        CHKERR( VecAbs(self.vec) )

    def conjugate(self):
        CHKERR( VecConjugate(self.vec) )

    def setRandom(self, Random random=None):
        cdef PetscRandom rnd = NULL
        if random is not None: rnd = random.rnd
        CHKERR( VecSetRandom(self.vec, rnd) )

    def permute(self, IS order, invert=False):
        cdef PetscBool cinvert = PETSC_FALSE
        if invert: cinvert = PETSC_TRUE
        CHKERR( VecPermute(self.vec, order.iset, cinvert) )

    def zeroEntries(self):
        CHKERR( VecZeroEntries(self.vec) )

    def set(self, alpha):
        cdef PetscScalar sval = asScalar(alpha)
        CHKERR( VecSet(self.vec, sval) )

    def isset(self, IS idx, alpha):
        cdef PetscScalar aval = asScalar(alpha)
        CHKERR( VecISSet(self.vec, idx.iset, aval) )

    def scale(self, alpha):
        cdef PetscScalar sval = asScalar(alpha)
        CHKERR( VecScale(self.vec, sval) )

    def shift(self, alpha):
        cdef PetscScalar sval = asScalar(alpha)
        CHKERR( VecShift(self.vec, sval) )

    def chop(self, tol):
        cdef PetscReal rval = asReal(tol)
        CHKERR( VecChop(self.vec, rval) )

    def swap(self, Vec vec):
        CHKERR( VecSwap(self.vec, vec.vec) )

    def axpy(self, alpha, Vec x):
        cdef PetscScalar sval = asScalar(alpha)
        CHKERR( VecAXPY(self.vec, sval, x.vec) )

    def isaxpy(self, IS idx, alpha, Vec x):
        cdef PetscScalar sval = asScalar(alpha)
        CHKERR( VecISAXPY(self.vec, idx.iset, sval, x.vec) )

    def aypx(self, alpha, Vec x):
        cdef PetscScalar sval = asScalar(alpha)
        CHKERR( VecAYPX(self.vec, sval, x.vec) )

    def axpby(self, alpha, beta, Vec y):
        cdef PetscScalar sval1 = asScalar(alpha)
        cdef PetscScalar sval2 = asScalar(beta)
        CHKERR( VecAXPBY(self.vec, sval1, sval2, y.vec) )

    def waxpy(self, alpha, Vec x, Vec y):
        cdef PetscScalar sval = asScalar(alpha)
        CHKERR( VecWAXPY(self.vec, sval, x.vec, y.vec) )

    def maxpy(self, alphas, vecs):
        cdef PetscInt n = 0
        cdef PetscScalar *a = NULL
        cdef PetscVec *v = NULL
        cdef object tmp1 = iarray_s(alphas, &n, &a)
        cdef object tmp2 = oarray_p(empty_p(n),NULL, <void**>&v)
        assert n == len(vecs)
        cdef Py_ssize_t i=0
        for i from 0 <= i < n:
            v[i] = (<Vec?>(vecs[i])).vec
        CHKERR( VecMAXPY(self.vec, n, a, v) )

    def pointwiseMult(self, Vec x, Vec y):
        CHKERR( VecPointwiseMult(self.vec, x.vec, y.vec) )

    def pointwiseDivide(self, Vec x, Vec y):
        CHKERR( VecPointwiseDivide(self.vec, x.vec, y.vec) )

    def pointwiseMin(self, Vec x, Vec y):
        CHKERR( VecPointwiseMin(self.vec, x.vec, y.vec) )

    def pointwiseMax(self, Vec x, Vec y):
        CHKERR( VecPointwiseMax(self.vec, x.vec, y.vec) )

    def pointwiseMaxAbs(self, Vec x, Vec y):
        CHKERR( VecPointwiseMaxAbs(self.vec, x.vec, y.vec) )

    def maxPointwiseDivide(self, Vec vec):
        cdef PetscReal rval = 0
        CHKERR( VecMaxPointwiseDivide(self.vec, vec.vec, &rval) )
        return toReal(rval)

    def getValue(self, index):
        cdef PetscInt    ival = asInt(index)
        cdef PetscScalar sval = 0
        CHKERR( VecGetValues(self.vec, 1, &ival, &sval) )
        return toScalar(sval)

    def getValues(self, indices, values=None):
        return vecgetvalues(self.vec, indices, values)

    def getValuesStagStencil(self, indices, values=None):
        raise NotImplementedError('getValuesStagStencil not yet implemented in petsc4py')

    def setValue(self, index, value, addv=None):
        cdef PetscInt    ival = asInt(index)
        cdef PetscScalar sval = asScalar(value)
        cdef PetscInsertMode caddv = insertmode(addv)
        CHKERR( VecSetValues(self.vec, 1, &ival, &sval, caddv) )

    def setValues(self, indices, values, addv=None):
        vecsetvalues(self.vec, indices, values, addv, 0, 0)

    def setValuesBlocked(self, indices, values, addv=None):
        vecsetvalues(self.vec, indices, values, addv, 1, 0)

    def setValuesStagStencil(self, indices, values, addv=None):
        raise NotImplementedError('setValuesStagStencil not yet implemented in petsc4py')

    def setLGMap(self, LGMap lgmap):
        CHKERR( VecSetLocalToGlobalMapping(self.vec, lgmap.lgm) )

    def getLGMap(self):
        cdef LGMap cmap = LGMap()
        CHKERR( VecGetLocalToGlobalMapping(self.vec, &cmap.lgm) )
        PetscINCREF(cmap.obj)
        return cmap

    def setValueLocal(self, index, value, addv=None):
        cdef PetscInt    ival = asInt(index)
        cdef PetscScalar sval = asScalar(value)
        cdef PetscInsertMode caddv = insertmode(addv)
        CHKERR( VecSetValuesLocal(self.vec, 1, &ival, &sval, caddv) )

    def setValuesLocal(self, indices, values, addv=None):
        vecsetvalues(self.vec, indices, values, addv, 0, 1)

    def setValuesBlockedLocal(self, indices, values, addv=None):
        vecsetvalues(self.vec, indices, values, addv, 1, 1)

    def assemblyBegin(self):
        CHKERR( VecAssemblyBegin(self.vec) )

    def assemblyEnd(self):
        CHKERR( VecAssemblyEnd(self.vec) )

    def assemble(self):
        CHKERR( VecAssemblyBegin(self.vec) )
        CHKERR( VecAssemblyEnd(self.vec) )

    # --- methods for strided vectors ---

    def strideScale(self, field, alpha):
        cdef PetscInt    ival = asInt(field)
        cdef PetscScalar sval = asScalar(alpha)
        CHKERR( VecStrideScale(self.vec, ival, sval) )

    def strideSum(self, field):
        cdef PetscInt    ival = asInt(field)
        cdef PetscScalar sval = 0
        CHKERR( VecStrideSum(self.vec, ival, &sval) )
        return toScalar(sval)

    def strideMin(self, field):
        cdef PetscInt  ival1 = asInt(field)
        cdef PetscInt  ival2 = 0
        cdef PetscReal rval  = 0
        CHKERR( VecStrideMin(self.vec, ival1, &ival2, &rval) )
        return (toInt(ival2), toReal(rval))

    def strideMax(self, field):
        cdef PetscInt  ival1 = asInt(field)
        cdef PetscInt  ival2 = 0
        cdef PetscReal rval  = 0
        CHKERR( VecStrideMax(self.vec, ival1, &ival2, &rval) )
        return (toInt(ival2), toReal(rval))

    def strideNorm(self, field, norm_type=None):
        cdef PetscInt ival = asInt(field)
        cdef PetscNormType norm_1_2 = PETSC_NORM_1_AND_2
        cdef PetscNormType ntype = PETSC_NORM_2
        if norm_type is not None: ntype = norm_type
        cdef PetscReal rval[2]
        CHKERR( VecStrideNorm(self.vec, ival, ntype, rval) )
        if ntype != norm_1_2: return toReal(rval[0])
        else: return (toReal(rval[0]), toReal(rval[1]))

    def strideScatter(self, field, Vec vec, addv=None):
        cdef PetscInt ival = asInt(field)
        cdef PetscInsertMode caddv = insertmode(addv)
        CHKERR( VecStrideScatter(self.vec, ival, vec.vec, caddv) )

    def strideGather(self, field, Vec vec, addv=None):
        cdef PetscInt ival = asInt(field)
        cdef PetscInsertMode caddv = insertmode(addv)
        CHKERR( VecStrideGather(self.vec, ival, vec.vec, caddv) )

    # --- methods for vectors with ghost values ---

    def localForm(self):
        """
        Intended for use in context manager::

            with vec.localForm() as lf:
                use(lf)
        """
        return _Vec_LocalForm(self)

    def ghostUpdateBegin(self, addv=None, mode=None):
        cdef PetscInsertMode  caddv = insertmode(addv)
        cdef PetscScatterMode csctm = scattermode(mode)
        CHKERR( VecGhostUpdateBegin(self.vec, caddv, csctm) )

    def ghostUpdateEnd(self, addv=None, mode=None):
        cdef PetscInsertMode  caddv = insertmode(addv)
        cdef PetscScatterMode csctm = scattermode(mode)
        CHKERR( VecGhostUpdateEnd(self.vec, caddv, csctm) )

    def ghostUpdate(self, addv=None, mode=None):
        cdef PetscInsertMode  caddv = insertmode(addv)
        cdef PetscScatterMode csctm = scattermode(mode)
        CHKERR( VecGhostUpdateBegin(self.vec, caddv, csctm) )
        CHKERR( VecGhostUpdateEnd(self.vec, caddv, csctm) )

    def setMPIGhost(self, ghosts):
        "Alternative to createGhost()"
        cdef PetscInt ng=0, *ig=NULL
        ghosts = iarray_i(ghosts, &ng, &ig)
        CHKERR( VecMPISetGhost(self.vec, ng, ig) )

    #

    def getSubVector(self, IS iset, Vec subvec=None):
        if subvec is None: subvec = Vec()
        else: CHKERR( VecDestroy(&subvec.vec) )
        CHKERR( VecGetSubVector(self.vec, iset.iset, &subvec.vec) )
        return subvec

    def restoreSubVector(self, IS iset, Vec subvec):
        CHKERR( VecRestoreSubVector(self.vec, iset.iset, &subvec.vec) )

    def getNestSubVecs(self):
        cdef PetscInt N=0
        cdef PetscVec* sx=NULL
        CHKERR( VecNestGetSubVecs(self.vec, &N, &sx) )
        output = []
        for i in range(N):
          pyvec = Vec()
          pyvec.vec = sx[i]
          CHKERR( PetscObjectReference(<PetscObject> pyvec.vec) )
          output.append(pyvec)

        return output

    def setNestSubVecs(self, sx, idxm=None):
        if idxm is None: idxm = range(len(sx))
        else: assert len(idxm) == len(sx)
        cdef PetscInt N = 0
        cdef PetscInt* cidxm = NULL
        idxm = iarray_i(idxm, &N, &cidxm)


        cdef PetscVec* csx = NULL
        tmp = oarray_p(empty_p(N), NULL, <void**>&csx)
        for i from 0 <= i < N: csx[i] = (<Vec?>sx[i]).vec

        CHKERR( VecNestSetSubVecs(self.vec, N, cidxm, csx) )

    #

    property sizes:
        def __get__(self):
            return self.getSizes()
        def __set__(self, value):
            self.setSizes(value)

    property size:
        def __get__(self):
            return self.getSize()

    property local_size:
        def __get__(self):
            return self.getLocalSize()

    property block_size:
        def __get__(self):
            return self.getBlockSize()

    property owner_range:
        def __get__(self):
            return self.getOwnershipRange()

    property owner_ranges:
        def __get__(self):
            return self.getOwnershipRanges()

    property buffer_w:
        "Vec buffer (writable)"
        def __get__(self):
            return self.getBuffer()

    property buffer_r:
        "Vec buffer (read-only)"
        def __get__(self):
            return self.getBuffer(True)

    property array_w:
        "Vec array (writable)"
        def __get__(self):
            return self.getArray()
        def __set__(self, value):
            cdef buf = self.getBuffer()
            with buf as array: array[:] = value

    property array_r:
        "Vec array (read-only)"
        def __get__(self):
            return self.getArray(True)

    property buffer:
        def __get__(self):
            return self.buffer_w

    property array:
        def __get__(self):
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
