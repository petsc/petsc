
cdef class DMLabel(Object):

    def __cinit__(self):
        self.obj = <PetscObject*> &self.dmlabel
        self.dmlabel  = NULL

    def destroy(self):
        CHKERR( DMLabelDestroy(&self.dmlabel) )
        return self

    def view(self, Viewer viewer=None):
        cdef PetscViewer vwr = NULL
        if viewer is not None: vwr = viewer.vwr
        CHKERR( DMLabelView(self.dmlabel, vwr) )

    def create(self, name, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_SELF)
        cdef PetscDMLabel newdmlabel = NULL
        cdef const char *cname = NULL
        name = str2bytes(name, &cname)
        CHKERR( DMLabelCreate(ccomm, cname, &newdmlabel) )
        PetscCLEAR(self.obj); self.dmlabel = newdmlabel
        return self
   
    def duplicate(self):
        cdef DMLabel new = DMLabel()
        CHKERR( DMLabelDuplicate(self.dmlabel, &new.dmlabel) )
        return new

    def reset(self):
        CHKERR( DMLabelReset(self.dmlabel) )

    def insertIS(self, IS iset, value):
        cdef PetscInt cvalue = asInt(value)
        CHKERR( DMLabelInsertIS(self.dmlabel, iset.iset, cvalue)  )
        return self

    def setValue(self, point, value):
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt cvalue = asInt(value)
        CHKERR( DMLabelSetValue(self.dmlabel, cpoint, cvalue) )

    def getValue(self, point):
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt cvalue = 0
        CHKERR( DMLabelGetValue(self.dmlabel, cpoint, &cvalue) )
        return toInt(cvalue)

    def getDefaultValue(self):
        cdef PetscInt cvalue = 0
        CHKERR( DMLabelGetDefaultValue(self.dmlabel, &cvalue) )
        return toInt(cvalue)

    def setDefaultValue(self, value):
        cdef PetscInt cvalue = asInt(value)
        CHKERR( DMLabelSetDefaultValue(self.dmlabel, cvalue) )

    def clearValue(self, point, value):
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt cvalue = asInt(value)
        CHKERR( DMLabelClearValue(self.dmlabel, cpoint, cvalue) )

    def addStratum(self, value):
        cdef PetscInt cvalue = asInt(value)
        CHKERR( DMLabelAddStratum(self.dmlabel, cvalue) )

    def addStrata(self, strata):
        cdef PetscInt *istrata = NULL
        cdef PetscInt numStrata = 0
        fields = iarray_i(strata, &numStrata, &istrata)
        CHKERR( DMLabelAddStrata(self.dmlabel, numStrata, istrata) )

    def addStrataIS(self, IS iset):
        CHKERR( DMLabelAddStrataIS(self.dmlabel, iset.iset) )

    def getNumValues(self):
        cdef PetscInt numValues = 0
        CHKERR( DMLabelGetNumValues(self.dmlabel, &numValues) )
        return toInt(numValues)

    def getValueIS(self):
        cdef IS iset = IS()
        CHKERR( DMLabelGetValueIS(self.dmlabel, &iset.iset) )
        return iset

    def stratumHasPoint(self, value, point):
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt cvalue = asInt(value)
        cdef PetscBool ccontains = PETSC_FALSE
        CHKERR( DMLabelStratumHasPoint(self.dmlabel, cvalue, cpoint, &ccontains) )
        return toBool(ccontains)

    def hasStratum(self, value):
        cdef PetscInt cvalue = asInt(value)
        cdef PetscBool cexists = PETSC_FALSE
        CHKERR( DMLabelHasStratum(self.dmlabel, cvalue, &cexists) )
        return toBool(cexists)
    
    def getStratumSize(self, stratum):
        cdef PetscInt cstratum = asInt(stratum)
        cdef PetscInt csize = 0
        CHKERR( DMLabelGetStratumSize(self.dmlabel, stratum, &csize) )
        return toInt(csize)
    
    def getStratumIS(self, stratum):
        cdef PetscInt cstratum = asInt(stratum)
        cdef IS iset = IS()
        CHKERR( DMLabelGetStratumIS(self.dmlabel, cstratum, &iset.iset) )
        return iset
    
    def setStratumIS(self, stratum, IS iset):
        cdef PetscInt cstratum = asInt(stratum)
        CHKERR( DMLabelSetStratumIS(self.dmlabel, cstratum, iset.iset) )
    
    def clearStratum(self, stratum):
        cdef PetscInt cstratum = asInt(stratum)
        CHKERR( DMLabelClearStratum(self.dmlabel, cstratum) )
    
    def computeIndex(self):
        CHKERR( DMLabelComputeIndex(self.dmlabel) )
    
    def createIndex(self, pStart, pEnd):
        cdef PetscInt cpstart = 0, cpend = 0
        CHKERR( DMLabelCreateIndex(self.dmlabel, cpstart, cpend) )
    
    def destroyIndex(self):
        CHKERR( DMLabelDestroyIndex(self.dmlabel) )
    
    def hasValue(self, value):
        cdef PetscInt cvalue = asInt(value)
        cdef PetscBool cexists = PETSC_FALSE
        CHKERR( DMLabelHasValue(self.dmlabel, cvalue, &cexists) )
        return toBool(cexists)
    
    def hasPoint(self, point):
        cdef PetscInt cpoint = asInt(point)
        cdef PetscBool cexists = PETSC_FALSE
        CHKERR( DMLabelHasPoint(self.dmlabel, cpoint, &cexists) )
        return toBool(cexists)
    
    def getBounds(self):
        cdef PetscInt cpstart = 0, cpend = 0
        CHKERR( DMLabelGetBounds(self.dmlabel, &cpstart, &cpend) )
        return toInt(cpstart), toInt(cpend)
    
    def filter(self, start, end):
        cdef PetscInt cstart = 0, cend = 0
        CHKERR( DMLabelFilter(self.dmlabel, cstart, cend) )
    
    def permute(self, IS permutation):
        cdef DMLabel new = DMLabel()
        CHKERR( DMLabelPermute(self.dmlabel, permutation.iset, &new.dmlabel) )
        return new
    
    def distribute(self, SF sf):
        cdef DMLabel new = DMLabel()
        CHKERR( DMLabelDistribute(self.dmlabel, sf.sf, &new.dmlabel) )
        return new
    
    def gather(self, SF sf):
        cdef DMLabel new = DMLabel()
        CHKERR( DMLabelGather(self.dmlabel, sf.sf, &new.dmlabel) )
        return new
    
    def convertToSection(self):
        cdef Section section = Section()
        cdef IS iset = IS()
        CHKERR( DMLabelConvertToSection(self.dmlabel, &section.sec, &iset.iset) )
        return section, iset

    def getNonEmptyStratumValuesIS(self):
        cdef IS iset = IS()
        CHKERR( DMLabelGetNonEmptyStratumValuesIS(self.dmlabel, &iset.iset) )
        return iset
