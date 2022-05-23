
cdef class DMLabel(Object):

    def __cinit__(self):
        self.obj = <PetscObject*> &self.dmlabel
        self.dmlabel  = NULL

    def destroy(self):
        CHKERR( DMLabelDestroy(&self.dmlabel) )
        return self

    def create(self, name, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_SELF)
        cdef PetscDMLabel newdmlabel = NULL
        cdef const char *cname = NULL
        name = str2bytes(name, &cname)
        CHKERR( DMLabelCreate(ccomm, cname, &newdmlabel) )
        PetscCLEAR(self.obj); self.dmlabel = newdmlabel
        return self

    def insertIS(self, IS iset, value):
        cdef PetscInt cvalue = asInt(value)
        CHKERR( DMLabelInsertIS(self.dmlabel, iset.iset, cvalue)  )
        return self

    def setValue(self, point, value):
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt cvalue = asInt(value)
        CHKERR( DMLabelSetValue(self.dmlabel, cpoint, cvalue) )
    
