# --------------------------------------------------------------------

cdef class Log:

    @classmethod
    def logFlops(cls, flops):
        cdef PetscLogDouble cflops=flops
        CHKERR( PetscLogFlops(cflops) )

    @classmethod
    def addFlops(cls, flops):
        cdef PetscLogDouble cflops=flops
        CHKERR( PetscLogFlops(cflops) )

    @classmethod
    def getFlops(cls):
        cdef PetscLogDouble cflops=0
        CHKERR( PetscGetFlops(&cflops) )
        return cflops
    getFlops = classmethod(getFlops)

    @classmethod
    def getTime(cls):
        cdef PetscLogDouble wctime=0
        CHKERR( PetscGetTime(&wctime) )
        return wctime
    getTime = classmethod(getTime)

    @classmethod
    def getCPUTime(cls):
        cdef PetscLogDouble cputime=0
        CHKERR( PetscGetCPUTime(&cputime) )
        return cputime
    getCPUTime = classmethod(getCPUTime)

# --------------------------------------------------------------------
