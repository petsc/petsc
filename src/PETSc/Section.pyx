# --------------------------------------------------------------------

cdef class Section(Object):

    def __cinit__(self):
        self.obj = <PetscObject*> &self.sec
        self.sec  = NULL

    def __dealloc__(self):
        CHKERR( PetscSectionDestroy(&self.sec) )
        self.sec = NULL

    def view(self, Viewer viewer=None):
        cdef PetscViewer vwr = NULL
        if viewer is not None: vwr = viewer.vwr
        CHKERR( PetscSectionView(self.sec, vwr) )

    def destroy(self):
        CHKERR( PetscSectionDestroy(&self.sec) )
        return self

    def create(self, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscSection newsec = NULL
        CHKERR( PetscSectionCreate(ccomm, &newsec) )
        PetscCLEAR(self.obj); self.sec = newsec
        return self

    def clone(self):
        cdef Section sec = <Section>type(self)()
        CHKERR( PetscSectionClone(self.sec, &sec.sec) )
        return sec

    def setUp(self):
        CHKERR( PetscSectionSetUp(self.sec) )

    def reset(self):
        CHKERR( PetscSectionReset(self.sec) )

    def getNumFields(self):
        cdef PetscInt numFields = 0
        CHKERR( PetscSectionGetNumFields(self.sec, &numFields) )
        return toInt(numFields)

    def setNumFields(self,numFields):
        cdef PetscInt cnumFields = asInt(numFields)
        CHKERR( PetscSectionSetNumFields(self.sec, cnumFields) )

    def getFieldName(self,field):
        cdef PetscInt cfield = asInt(field)
        cdef const char *fieldName = NULL
        CHKERR( PetscSectionGetFieldName(self.sec,cfield,&fieldName) )
        return bytes2str(fieldName)

    def setFieldName(self,field,fieldName):
        cdef PetscInt cfield = asInt(field)
        cdef const char *cname = NULL
        fieldName = str2bytes(fieldName, &cname)
        CHKERR( PetscSectionSetFieldName(self.sec,cfield,cname) )

    def getFieldComponents(self,field):
        cdef PetscInt cfield = asInt(field), cnumComp = 0
        CHKERR( PetscSectionGetFieldComponents(self.sec,cfield,&cnumComp) )
        return toInt(cnumComp)

    def setFieldComponents(self,field,numComp):
        cdef PetscInt cfield = asInt(field)
        cdef PetscInt cnumComp = asInt(numComp)
        CHKERR( PetscSectionSetFieldComponents(self.sec,cfield,cnumComp) )

    def getChart(self):
        cdef PetscInt pStart = 0, pEnd = 0
        CHKERR( PetscSectionGetChart(self.sec, &pStart, &pEnd) )
        return toInt(pStart), toInt(pEnd)

    def setChart(self, pStart, pEnd):
        cdef PetscInt cStart = asInt(pStart)
        cdef PetscInt cEnd   = asInt(pEnd)
        CHKERR( PetscSectionSetChart(self.sec, cStart, cEnd) )

    def getDof(self,point):
        cdef PetscInt cpoint = asInt(point), cnumDof = 0
        CHKERR( PetscSectionGetDof(self.sec,cpoint,&cnumDof) )
        return toInt(cnumDof)

    def setDof(self,point,numDof):
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt cnumDof = asInt(numDof)
        CHKERR( PetscSectionSetDof(self.sec,cpoint,cnumDof) )

    def addDof(self,point,numDof):
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt cnumDof = asInt(numDof)
        CHKERR( PetscSectionAddDof(self.sec,cpoint,cnumDof) )

    def getFieldDof(self,point,field):
        cdef PetscInt cpoint = asInt(point), cnumDof = 0
        cdef PetscInt cfield = asInt(field)
        CHKERR( PetscSectionGetFieldDof(self.sec,cpoint,cfield,&cnumDof) )
        return toInt(cnumDof)

    def setFieldDof(self,point,field,numDof):
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt cfield = asInt(field)
        cdef PetscInt cnumDof = asInt(numDof)
        CHKERR( PetscSectionSetFieldDof(self.sec,cpoint,cfield,cnumDof) )

    def addFieldDof(self,point,field,numDof):
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt cfield = asInt(field)
        cdef PetscInt cnumDof = asInt(numDof)
        CHKERR( PetscSectionAddFieldDof(self.sec,cpoint,cfield,cnumDof) )

    def getConstraintDof(self,point):
        cdef PetscInt cpoint = asInt(point), cnumDof = 0
        CHKERR( PetscSectionGetConstraintDof(self.sec,cpoint,&cnumDof) )
        return toInt(cnumDof)

    def setConstraintDof(self,point,numDof):
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt cnumDof = asInt(numDof)
        CHKERR( PetscSectionSetConstraintDof(self.sec,cpoint,cnumDof) )

    def addConstraintDof(self,point,numDof):
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt cnumDof = asInt(numDof)
        CHKERR( PetscSectionAddConstraintDof(self.sec,cpoint,cnumDof) )

    def getFieldConstraintDof(self,point,field):
        cdef PetscInt cpoint = asInt(point), cnumDof = 0
        cdef PetscInt cfield = asInt(field)
        CHKERR( PetscSectionGetFieldConstraintDof(self.sec,cpoint,cfield,&cnumDof) )
        return toInt(cnumDof)

    def setFieldConstraintDof(self,point,field,numDof):
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt cfield = asInt(field)
        cdef PetscInt cnumDof = asInt(numDof)
        CHKERR( PetscSectionSetFieldConstraintDof(self.sec,cpoint,cfield,cnumDof) )

    def addFieldConstraintDof(self,point,field,numDof):
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt cfield = asInt(field)
        cdef PetscInt cnumDof = asInt(numDof)
        CHKERR( PetscSectionAddFieldConstraintDof(self.sec,cpoint,cfield,cnumDof) )

    def getConstraintIndices(self,point):
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt nindex = 0
        cdef const PetscInt *indices = NULL
        CHKERR( PetscSectionGetConstraintDof(self.sec, cpoint, &nindex) )
        CHKERR( PetscSectionGetConstraintIndices(self.sec, cpoint, &indices) )
        return array_i(nindex, indices)

    def setConstraintIndices(self,point,indices):
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt nindex = 0
        cdef PetscInt *cindices = NULL
        indices = iarray_i(indices, &nindex, &cindices)
        CHKERR( PetscSectionSetConstraintDof(self.sec,cpoint,nindex) )
        CHKERR( PetscSectionSetConstraintIndices(self.sec,cpoint,cindices) )

    def getFieldConstraintIndices(self,point,field):
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt cfield = asInt(field)
        cdef PetscInt nindex = 0
        cdef const PetscInt *indices = NULL
        CHKERR( PetscSectionGetFieldConstraintDof(self.sec,cpoint,cfield,&nindex) )
        CHKERR( PetscSectionGetFieldConstraintIndices(self.sec,cpoint,cfield,&indices) )
        return array_i(nindex, indices)

    def setFieldConstraintIndices(self,point,field,indices):
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt cfield = asInt(field)
        cdef PetscInt nindex = 0
        cdef PetscInt *cindices = NULL
        indices = iarray_i(indices, &nindex, &cindices)
        CHKERR( PetscSectionSetFieldConstraintDof(self.sec,cpoint,cfield,nindex) )
        CHKERR( PetscSectionSetFieldConstraintIndices(self.sec,cpoint,cfield,cindices) )

    def getMaxDof(self):
        cdef PetscInt maxDof = 0
        CHKERR( PetscSectionGetMaxDof(self.sec,&maxDof) )
        return toInt(maxDof)

    def getStorageSize(self):
        cdef PetscInt size = 0
        CHKERR( PetscSectionGetStorageSize(self.sec,&size) )
        return toInt(size)

    def getConstrainedStorageSize(self):
        cdef PetscInt size = 0
        CHKERR( PetscSectionGetConstrainedStorageSize(self.sec,&size) )
        return toInt(size)

    def getOffset(self,point):
        cdef PetscInt cpoint = asInt(point), offset = 0
        CHKERR( PetscSectionGetOffset(self.sec,cpoint,&offset) )
        return toInt(offset)

    def setOffset(self,point,offset):
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt coffset = asInt(offset)
        CHKERR( PetscSectionSetOffset(self.sec,cpoint,coffset) )

    def getFieldOffset(self,point,field):
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt cfield = asInt(field)
        cdef PetscInt offset = 0
        CHKERR( PetscSectionGetFieldOffset(self.sec,cpoint,cfield,&offset) )
        return toInt(offset)

    def setFieldOffset(self,point,field,offset):
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt cfield = asInt(field)
        cdef PetscInt coffset = asInt(offset)
        CHKERR( PetscSectionSetFieldOffset(self.sec,cpoint,cfield,coffset) )

    def getOffsetRange(self):
        cdef PetscInt oStart = 0, oEnd = 0
        CHKERR( PetscSectionGetOffsetRange(self.sec,&oStart,&oEnd) )
        return toInt(oStart),toInt(oEnd)

    def createGlobalSection(self, SF sf):
        cdef Section gsec = Section()
        CHKERR( PetscSectionCreateGlobalSection(self.sec,sf.sf,PETSC_FALSE,PETSC_FALSE,&gsec.sec) )
        return gsec
