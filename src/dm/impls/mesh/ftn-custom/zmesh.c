#include <petsc-private/fortranimpl.h>
#include <petscdmmesh.h>

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define dmmeshcreatepcice_          DMMESHCREATEPCICE
#define dmmeshcreateexodus_         DMMESHCREATEEXODUS
#define dmmeshdistribute_           DMMESHDISTRIBUTE
#define dmmeshgetvertexsectionreal_ DMMESHGETVERTEXSECTIONREAL
#define dmmeshgetcellsectionreal_   DMMESHGETCELLSECTIONREAL
#define dmmeshgetvertexsectionint_  DMMESHGETVERTEXSECTIONINT
#define dmmeshgetcellsectionint_    DMMESHGETCELLSECTIONINT
#define dmmeshcreatesectionrealis_  DMMESHCREATESECTIONREALIS
#define vertexsectionrealcreate_    VERTEXSECTIONREALCREATE
#define vertexsectionintcreate_     VERTEXSECTIONINTCREATE
#define cellsectionrealcreate_      CELLSECTIONREALCREATE
#define dmmeshgetlabelsize_         DMMESHGETLABELSIZE
#define dmmeshgetlabelidis_         DMMESHGETLABELIDIS
#define dmmeshgetstratumsize_       DMMESHGETSTRATUMSIZE
#define dmmeshgetstratumis_         DMMESHGETSTRATUMIS
#define dmmeshgetsectionreal_       DMMESHGETSECTIONREAL
#define dmmeshgetsectionint_        DMMESHGETSECTIONINT
#define dmmeshsetsectionreal_       DMMESHSETSECTIONREAL
#define dmmeshcreatesection_        DMMESHCREATESECTION
#define dmmeshsetsection_           DMMESHSETSECTION
#define dmmeshcreatematrix_         DMMESHCREATEMATRIX
#define alestagepush_               ALESTAGEPUSH
#define alestagepop_                ALESTAGEPOP
#define alestageprintmemory_        ALESTAGEPRINTMEMORY
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define dmmeshcreatepcice_          dmmeshcreatepcice
#define dmmeshcreateexodus_         dmmeshcreateexodus
#define dmmeshdistribute_           dmmeshdistribute
#define dmmeshgetvertexsectionreal_ dmmeshgetvertexsectionreal
#define dmmeshgetcellsectionreal_   dmmeshgetcellsectionreal
#define dmmeshgetvertexsectionint_  dmmeshgetvertexsectionint
#define dmmeshgetcellsectionint_    dmmeshgetcellsectionint
#define dmmeshcreatesectionrealis_  dmmeshcreatesectionrealis
#define vertexsectionrealcreate_    vertexsectionrealcreate
#define vertexsectionintcreate_     vertexsectionintcreate
#define cellsectionrealcreate_      cellsectionrealcreate
#define dmmeshgetlabelsize_         dmmeshgetlabelsize
#define dmmeshgetlabelidis_         dmmeshgetlabelidis
#define dmmeshgetstratumsize_       dmmeshgetstratumsize
#define dmmeshgetstratumis_         dmmeshgetstratumis
#define dmmeshgetsectionreal_       dmmeshgetsectionreal
#define dmmeshgetsectionint_        dmmeshgetsectionint
#define dmmeshsetsectionreal_       dmmeshsetsectionreal
#define dmmeshcreatesection_        dmmeshcreatesection
#define dmmeshsetsection_           dmmeshsetsection
#define dmmeshcreatematrix_         dmmeshcreatematrix
#define alestagepush_               alestagepush
#define alestagepop_                alestagepop
#define alestageprintmemory_        alestageprintmemory
#endif

/* Definitions of Fortran Wrapper routines */
EXTERN_C_BEGIN

void PETSC_STDCALL  dmmeshcreatepcice_(MPI_Fint * comm, int *dim, CHAR coordFilename PETSC_MIXED_LEN(lenC), CHAR adjFilename PETSC_MIXED_LEN(lenA), PetscBool  *interpolate, CHAR bcFilename PETSC_MIXED_LEN(lenB), DM *dm, PetscErrorCode *ierr PETSC_END_LEN(lenC) PETSC_END_LEN(lenA) PETSC_END_LEN(lenB))
{
  char *cF, *aF, *bF;
  FIXCHAR(coordFilename,lenC,cF);
  FIXCHAR(adjFilename,lenA,aF);
  FIXCHAR(bcFilename,lenB,bF);
  *ierr = DMMeshCreatePCICE(MPI_Comm_f2c( *(comm) ),*dim,cF,aF,*interpolate,bF,dm);
  FREECHAR(coordFilename,cF);
  FREECHAR(adjFilename,aF);
  FREECHAR(bcFilename,bF);
}
void PETSC_STDCALL  dmmeshcreateexodus_(MPI_Fint * comm, CHAR filename PETSC_MIXED_LEN(len), DM *dm, PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *cF;
  FIXCHAR(filename,len,cF);
  *ierr = DMMeshCreateExodus(MPI_Comm_f2c( *(comm) ),cF,dm);
  FREECHAR(filename,cF);
}
void PETSC_STDCALL  dmmeshdistribute_(DM *serialMesh, CHAR partitioner PETSC_MIXED_LEN(lenP), DM *parallelMesh, PetscErrorCode *ierr PETSC_END_LEN(lenP))
{
  char *pF;
  FIXCHAR(partitioner,lenP,pF);
  *ierr = DMMeshDistribute(*serialMesh,pF,parallelMesh);
  FREECHAR(partitioner,pF);
}
void PETSC_STDCALL  dmmeshgetvertexsectionreal_(DM *mesh, CHAR name PETSC_MIXED_LEN(lenN), PetscInt *fiberDim, SectionReal *section, int *ierr PETSC_END_LEN(lenN)){
  char *pN;
  FIXCHAR(name,lenN,pN);
  *ierr = DMMeshGetVertexSectionReal(*mesh, pN, *fiberDim, section);
  FREECHAR(name,pN);
}
void PETSC_STDCALL  dmmeshgetcellsectionreal_(DM *mesh, CHAR name PETSC_MIXED_LEN(lenN), PetscInt *fiberDim, SectionReal *section, int *ierr PETSC_END_LEN(lenN)){
  char *pN;
  FIXCHAR(name,lenN,pN);
  *ierr = DMMeshGetCellSectionReal(*mesh, pN, *fiberDim, section);
  FREECHAR(name,pN);
}
void PETSC_STDCALL  dmmeshcreatesectionrealis_(DM *dm, IS *is, CHAR name PETSC_MIXED_LEN(lenN), PetscInt *fiberDim, SectionReal *section, int *ierr PETSC_END_LEN(lenN)){
  char *pN;
  FIXCHAR(name,lenN,pN);
  *ierr = DMMeshCreateSectionRealIS(*dm,*is,pN, *fiberDim, section);
  FREECHAR(name,pN);
}
void PETSC_STDCALL  dmmeshgetvertexsectionint_(DM *mesh, CHAR name PETSC_MIXED_LEN(lenN), PetscInt *fiberDim, SectionInt *section, int *ierr PETSC_END_LEN(lenN)){
  char *pN;
  FIXCHAR(name,lenN,pN);
  *ierr = DMMeshGetVertexSectionInt(*mesh, pN, *fiberDim, section);
  FREECHAR(name,pN);
}
void PETSC_STDCALL  mdmeshgetcellsectionint_(DM *mesh, CHAR name PETSC_MIXED_LEN(lenN), PetscInt *fiberDim, SectionInt *section, int *ierr PETSC_END_LEN(lenN)){
  char *pN;
  FIXCHAR(name,lenN,pN);
  *ierr = DMMeshGetCellSectionInt(*mesh, pN, *fiberDim, section);
  FREECHAR(name,pN);
}
void PETSC_STDCALL  vertexsectionrealcreate_(DM *mesh, CHAR name PETSC_MIXED_LEN(lenN), PetscInt *fiberDim, int *ierr PETSC_END_LEN(lenN)){
  SectionReal section;
  char *pN;
  FIXCHAR(name,lenN,pN);
  *ierr = DMMeshGetVertexSectionReal(*mesh, pN, *fiberDim, &section);
  *ierr = DMMeshSetSectionReal(*mesh, pN, section);
  *ierr = SectionRealDestroy(&section);
  FREECHAR(name,pN);
}
void PETSC_STDCALL  vertexsectionintcreate_(DM *mesh, CHAR name PETSC_MIXED_LEN(lenN), PetscInt *fiberDim, int *ierr PETSC_END_LEN(lenN)){
  SectionInt section;
  char *pN;
  FIXCHAR(name,lenN,pN);
  *ierr = DMMeshGetVertexSectionInt(*mesh, pN, *fiberDim, &section);
  *ierr = DMMeshSetSectionInt(*mesh, section);
  *ierr = SectionIntDestroy(&section);
  FREECHAR(name,pN);
}
void PETSC_STDCALL  cellsectionrealcreate_(DM *mesh, CHAR name PETSC_MIXED_LEN(lenN), PetscInt *fiberDim, int *ierr PETSC_END_LEN(lenN)){
  SectionReal section;
  char *pN;
  FIXCHAR(name,lenN,pN);
  *ierr = DMMeshGetCellSectionReal(*mesh, pN, *fiberDim, &section);
  *ierr = DMMeshSetSectionReal(*mesh, pN, section);
  *ierr = SectionRealDestroy(&section);
  FREECHAR(name,pN);
}
void PETSC_STDCALL  dmmeshgetlabelsize_(DM *mesh, CHAR name PETSC_MIXED_LEN(lenN), PetscInt *size, int *ierr PETSC_END_LEN(lenN)){
  char *pN;
  FIXCHAR(name,lenN,pN);
  *ierr = DMMeshGetLabelSize(*mesh,pN, size);
  FREECHAR(name,pN);
}
void PETSC_STDCALL dmmeshgetlabelidis_(DM *dm, CHAR name PETSC_MIXED_LEN(lenN), IS *ids, int *ierr PETSC_END_LEN(lenN)){
  char *pN;
  FIXCHAR(name,lenN,pN);
  *ierr = DMMeshGetLabelIdIS(*dm,pN,ids);
  FREECHAR(name,pN);
}
void PETSC_STDCALL  dmmeshgetstratumsize_(DM *mesh, CHAR name PETSC_MIXED_LEN(lenN), PetscInt *value, PetscInt *size, int *ierr PETSC_END_LEN(lenN)){
  char *pN;
  FIXCHAR(name,lenN,pN);
  *ierr = DMMeshGetStratumSize(*mesh,pN, *value, size);
  FREECHAR(name,pN);
}

void PETSC_STDCALL  dmmeshgetstratumis_(DM *mesh, CHAR name PETSC_MIXED_LEN(lenN), PetscInt *value, IS *is, int *ierr PETSC_END_LEN(lenN)){
  char *pN;
  FIXCHAR(name,lenN,pN);
  *ierr = DMMeshGetStratumIS(*mesh,pN, *value, is);
  FREECHAR(name,pN);
}

void PETSC_STDCALL  dmmeshgetsectionreal_(DM *mesh, CHAR name PETSC_MIXED_LEN(lenN), SectionReal *section, int *ierr PETSC_END_LEN(lenN)){
  char *pN;
  FIXCHAR(name,lenN,pN);
  *ierr = DMMeshGetSectionReal(*mesh, pN, section);
  FREECHAR(name,pN);
}

void PETSC_STDCALL  dmmeshgetsectionint_(DM *mesh, CHAR name PETSC_MIXED_LEN(lenN), SectionInt *section, int *ierr PETSC_END_LEN(lenN)){
  char *pN;
  FIXCHAR(name,lenN,pN);
  *ierr = DMMeshGetSectionInt(*mesh, pN, section);
  FREECHAR(name,pN);
}

void PETSC_STDCALL  dmmeshsetsectionreal_(DM *mesh, CHAR name PETSC_MIXED_LEN(lenN), SectionReal *section, int *ierr PETSC_END_LEN(lenN)){
  char *pN;
  FIXCHAR(name,lenN,pN);
  *ierr = DMMeshSetSectionReal(*mesh, pN, *section);
  FREECHAR(name,pN);
}

void PETSC_STDCALL  dmmeshcreatesection_(DM *mesh, PetscInt *dim, PetscInt *numFields, PetscInt numComp[], PetscInt numDof[], PetscInt *numBC, PetscInt bcField[], IS bcPoints[], PetscSection *section, int *ierr){
  *ierr = DMMeshCreateSection(*mesh, *dim, *numFields, numComp, numDof, *numBC, bcField, bcPoints, section);
}

void PETSC_STDCALL  dmmeshsetsection_(DM *mesh, CHAR name PETSC_MIXED_LEN(lenN), PetscSection *section, int *ierr PETSC_END_LEN(lenN)){
  char *pN;
  FIXCHAR(name,lenN,pN);
  *ierr = DMMeshSetSection(*mesh, pN, *section);
  FREECHAR(name,pN);
}

void PETSC_STDCALL  dmmeshcreatematrix_(DM *mesh, SectionReal *section, CHAR mattype PETSC_MIXED_LEN(lenN), Mat *J, int *ierr PETSC_END_LEN(lenN)){
  char *pN;
  FIXCHAR(mattype,lenN,pN);
  *ierr = DMMeshCreateMatrix(*mesh, *section, pN, J);
  FREECHAR(mattype,pN);
}

void PETSC_STDCALL  alestagepush_(CHAR name PETSC_MIXED_LEN(lenN), PetscInt *debug, int *ierr PETSC_END_LEN(lenN)){
  ALE::MemoryLogger& logger = ALE::MemoryLogger::singleton();
  char *pN;

  FIXCHAR(name,lenN,pN);
  logger.setDebug(*debug);
  logger.stagePush(pN);
  FREECHAR(name,pN);
}

void PETSC_STDCALL  alestagepop_(PetscInt *debug, int *ierr){
  ALE::MemoryLogger& logger = ALE::MemoryLogger::singleton();

  logger.setDebug(*debug);
  logger.stagePop();
}

void PETSC_STDCALL  alestageprintmemory_(MPI_Fint * comm,CHAR name PETSC_MIXED_LEN(lenN), int *ierr PETSC_END_LEN(lenN)){
  ALE::MemoryLogger& logger = ALE::MemoryLogger::singleton();
  char *pN;

  FIXCHAR(name,lenN,pN);
  *ierr = PetscPrintf(MPI_Comm_f2c(*(comm)), "%s %d allocations %d bytes\n", pN, logger.getNumAllocations(pN), logger.getAllocationTotal(pN));
  *ierr = PetscPrintf(MPI_Comm_f2c(*(comm)), "%s %d deallocations %d bytes\n", pN, logger.getNumDeallocations(pN), logger.getDeallocationTotal(pN));
  FREECHAR(name,pN);
}

EXTERN_C_END
