#include "private/fortranimpl.h"
#include "petscmesh.h"
/* mesh.c */
/* Fortran interface file */

/*
* This file was generated automatically by bfort from the C source
* file.  
 */

#ifdef PETSC_USE_POINTER_CONVERSION
#if defined(__cplusplus)
extern "C" { 
#endif 
extern void *PetscToPointer(void*);
extern int PetscFromPointer(void *);
extern void PetscRmPointer(void*);
#if defined(__cplusplus)
} 
#endif 

#else

#define PetscToPointer(a) (*(long *)(a))
#define PetscFromPointer(a) (long)(a)
#define PetscRmPointer(a)
#endif

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define meshcreatepcice_        MESHCREATEPCICE
#define meshcreateexodus_       MESHCREATEEXODUS
#define meshdistribute_         MESHDISTRIBUTE
#define meshview_               MESHVIEW
#define meshgetvertexsectionreal_ MESHGETVERTEXSECTIONREAL
#define meshgetcellsectionreal_   MESHGETCELLSECTIONREAL
#define meshgetvertexsectionint_  MESHGETVERTEXSECTIONINT
#define meshgetcellsectionint_    MESHGETCELLSECTIONINT
#define vertexsectionrealcreate_ VERTEXSECTIONREALCREATE
#define vertexsectionintcreate_  VERTEXSECTIONINTCREATE
#define cellsectionrealcreate_   CELLSECTIONREALCREATE
#define bcsectionrealcreate_     BCSECTIONREALCREATE
#define restrictvector_         RESTRICTVECTOR
#define assemblevectorcomplete_ ASSEMBLEVECTORCOMPLETE
#define assemblevector_         ASSEMBLEVECTOR
#define writepcicerestart_      WRITEPCICERESTART
#define meshexodusgetinfo_      MESHEXODUSGETINFO
#define meshgetlabelsize_       MESHGETLABELSIZE
#define meshgetstratumsize_     MESHGETSTRATUMSIZE
#define meshgetsectionreal_     MESHGETSECTIONREAL
#define meshgetsectionint_      MESHGETSECTIONINT
#define meshgetmatrix_          MESHGETMATRIX
#define meshcreatematrix_       MESHCREATEMATRIX
#define alestagepush_           ALESTAGEPUSH
#define alestagepop_            ALESTAGEPOP
#define alestageprintmemory_    ALESTAGEPRINTMEMORY
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define meshcreatepcice_        meshcreatepcice
#define meshcreateexodus_       meshcreateexodus
#define meshdistribute_         meshdistribute
#define meshview_               meshview
#define meshgetvertexsectionreal_  meshgetvertexsectionreal
#define meshgetcellsectionreal_    meshgetcellsectionreal
#define meshgetvertexsectionint_   meshgetvertexsectionint
#define meshgetcellsectionint_     meshgetcellsectionint
#define vertexsectionrealcreate_ vertexsectionrealcreate
#define vertexsectionintcreate_  vertexsectionintcreate
#define cellsectionrealcreate_   cellsectionrealcreate
#define bcsectionrealcreate_     bcsectionrealcreate
#define restrictvector_         restrictvector
#define assemblevectorcomplete_ assemblevectorcomplete
#define assemblevector_         assemblevector
#define writepcicerestart_      writepcicerestart
#define meshexodusgetinfo_      meshexodusgetinfo
#define meshgetlabelsize_       meshgetlabelsize
#define meshgetstratumsize_     meshgetstratumsize
#define meshgetsectionreal_     meshgetsectionreal
#define meshgetsectionint_      meshgetsectionint
#define meshgetmatrix_          meshgetmatrix
#define meshcreatematrix_       meshcreatematrix
#define alestagepush_           alestagepush
#define alestagepop_            alestagepop
#define alestageprintmemory_    alestageprintmemory
#endif

/* Definitions of Fortran Wrapper routines */
EXTERN_C_BEGIN

void PETSC_STDCALL  meshcreatepcice_(MPI_Fint * comm, int *dim, CHAR coordFilename PETSC_MIXED_LEN(lenC), CHAR adjFilename PETSC_MIXED_LEN(lenA), PetscTruth *interpolate, CHAR bcFilename PETSC_MIXED_LEN(lenB), Mesh *mesh, PetscErrorCode *ierr PETSC_END_LEN(lenC) PETSC_END_LEN(lenA) PETSC_END_LEN(lenB))
{
  char *cF, *aF, *bF;
  FIXCHAR(coordFilename,lenC,cF);
  FIXCHAR(adjFilename,lenA,aF);
  FIXCHAR(bcFilename,lenB,bF);
  *ierr = MeshCreatePCICE(MPI_Comm_f2c( *(comm) ),*dim,cF,aF,*interpolate,bF,mesh);
  FREECHAR(coordFilename,cF);
  FREECHAR(adjFilename,aF);
  FREECHAR(bcFilename,bF);
}
void PETSC_STDCALL  meshcreateexodus_(MPI_Fint * comm, CHAR filename PETSC_MIXED_LEN(len), Mesh *mesh, PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *cF;
  FIXCHAR(filename,len,cF);
  *ierr = MeshCreateExodus(MPI_Comm_f2c( *(comm) ),cF,mesh);
  FREECHAR(filename,cF);
}
void PETSC_STDCALL  meshdistribute_(Mesh serialMesh, CHAR partitioner PETSC_MIXED_LEN(lenP), Mesh *parallelMesh, PetscErrorCode *ierr PETSC_END_LEN(lenP))
{
  char *pF;
  FIXCHAR(partitioner,lenP,pF);
  *ierr = MeshDistribute((Mesh) PetscToPointer(serialMesh),pF,parallelMesh);
  FREECHAR(partitioner,pF);
}
void PETSC_STDCALL  meshview_(Mesh mesh, PetscViewer *vin, PetscErrorCode *ierr)
{
  PetscViewer v;

  PetscPatchDefaultViewers_Fortran(vin,v);
  *ierr = MeshView((Mesh) PetscToPointer(mesh),v);
}
void PETSC_STDCALL  meshgetvertexsectionreal_(Mesh mesh, CHAR name PETSC_MIXED_LEN(lenN), PetscInt *fiberDim, SectionReal *section, int *ierr PETSC_END_LEN(lenN)){
  char *pN;
  FIXCHAR(name,lenN,pN);
  *ierr = MeshGetVertexSectionReal((Mesh) PetscToPointer(mesh), pN, *fiberDim, section);
  FREECHAR(name,pN);
}
void PETSC_STDCALL  meshgetcellsectionreal_(Mesh mesh, CHAR name PETSC_MIXED_LEN(lenN), PetscInt *fiberDim, SectionReal *section, int *ierr PETSC_END_LEN(lenN)){
  char *pN;
  FIXCHAR(name,lenN,pN);
  *ierr = MeshGetCellSectionReal((Mesh) PetscToPointer(mesh), pN, *fiberDim, section);
  FREECHAR(name,pN);
}
void PETSC_STDCALL  meshgetvertexsectionint_(Mesh mesh, CHAR name PETSC_MIXED_LEN(lenN), PetscInt *fiberDim, SectionInt *section, int *ierr PETSC_END_LEN(lenN)){
  char *pN;
  FIXCHAR(name,lenN,pN);
  *ierr = MeshGetVertexSectionInt((Mesh) PetscToPointer(mesh), pN, *fiberDim, section);
  FREECHAR(name,pN);
}
void PETSC_STDCALL  meshgetcellsectionint_(Mesh mesh, CHAR name PETSC_MIXED_LEN(lenN), PetscInt *fiberDim, SectionInt *section, int *ierr PETSC_END_LEN(lenN)){
  char *pN;
  FIXCHAR(name,lenN,pN);
  *ierr = MeshGetCellSectionInt((Mesh) PetscToPointer(mesh), pN, *fiberDim, section);
  FREECHAR(name,pN);
}
void PETSC_STDCALL  vertexsectionrealcreate_(Mesh mesh, CHAR name PETSC_MIXED_LEN(lenN), PetscInt *fiberDim, int *ierr PETSC_END_LEN(lenN)){
  SectionReal section;
  char *pN;
  FIXCHAR(name,lenN,pN);
  *ierr = MeshGetVertexSectionReal((Mesh) PetscToPointer(mesh), pN, *fiberDim, &section);
  *ierr = MeshSetSectionReal((Mesh) PetscToPointer(mesh), section);
  *ierr = SectionRealDestroy(section);
  FREECHAR(name,pN);
}
void PETSC_STDCALL  vertexsectionintcreate_(Mesh mesh, CHAR name PETSC_MIXED_LEN(lenN), PetscInt *fiberDim, int *ierr PETSC_END_LEN(lenN)){
  SectionInt section;
  char *pN;
  FIXCHAR(name,lenN,pN);
  *ierr = MeshGetVertexSectionInt((Mesh) PetscToPointer(mesh), pN, *fiberDim, &section);
  *ierr = MeshSetSectionInt((Mesh) PetscToPointer(mesh), section);
  *ierr = SectionIntDestroy(section);
  FREECHAR(name,pN);
}
void PETSC_STDCALL  cellsectionrealcreate_(Mesh mesh, CHAR name PETSC_MIXED_LEN(lenN), PetscInt *fiberDim, int *ierr PETSC_END_LEN(lenN)){
  SectionReal section;
  char *pN;
  FIXCHAR(name,lenN,pN);
  *ierr = MeshGetCellSectionReal((Mesh) PetscToPointer(mesh), pN, *fiberDim, &section);
  *ierr = MeshSetSectionReal((Mesh) PetscToPointer(mesh), section);
  *ierr = SectionRealDestroy(section);
  FREECHAR(name,pN);
}
void PETSC_STDCALL  bcsectionrealcreate_(Mesh mesh, CHAR name PETSC_MIXED_LEN(lenN), PetscInt *fiberDim, int *ierr PETSC_END_LEN(lenN)){
  char *pN;
  FIXCHAR(name,lenN,pN);
  *ierr = BCSectionRealCreate((Mesh) PetscToPointer(mesh),pN, *fiberDim);
  FREECHAR(name,pN);
}
void PETSC_STDCALL  restrictvector_(Vec g,Vec l,InsertMode *mode, int *__ierr ){
*__ierr = restrictVector(
	(Vec)PetscToPointer((g) ),
	(Vec)PetscToPointer((l) ),*mode);
}
void PETSC_STDCALL  assemblevectorcomplete_(Vec g,Vec l,InsertMode *mode, int *__ierr ){
*__ierr = assembleVectorComplete(
	(Vec)PetscToPointer((g) ),
	(Vec)PetscToPointer((l) ),*mode);
}
void PETSC_STDCALL  assemblevector_(Vec b,PetscInt *e,PetscScalar v[],InsertMode *mode, int *__ierr ){
*__ierr = assembleVector(
	(Vec)PetscToPointer((b) ),*e,v,*mode);
}
void PETSC_STDCALL  writepcicerestart_(Mesh mesh, PetscViewer viewer, int *ierr){
  *ierr = WritePCICERestart((Mesh) PetscToPointer(mesh), (PetscViewer) PetscToPointer(viewer));
}
void PETSC_STDCALL  meshexodusgetinfo_(Mesh mesh, PetscInt *dim, PetscInt *numVertices, PetscInt *numCells, PetscInt *numCellBlocks, PetscInt *numVertexSets, int *ierr){
  *ierr = MeshExodusGetInfo((Mesh) PetscToPointer(mesh), dim, numVertices, numCells, numCellBlocks, numVertexSets);
}
void PETSC_STDCALL  meshgetlabelsize_(Mesh mesh, CHAR name PETSC_MIXED_LEN(lenN), PetscInt *size, int *ierr PETSC_END_LEN(lenN)){
  char *pN;
  FIXCHAR(name,lenN,pN);
  *ierr = MeshGetLabelSize((Mesh) PetscToPointer(mesh),pN, size);
  FREECHAR(name,pN);
}
void PETSC_STDCALL  meshgetstratumsize_(Mesh mesh, CHAR name PETSC_MIXED_LEN(lenN), PetscInt *value, PetscInt *size, int *ierr PETSC_END_LEN(lenN)){
  char *pN;
  FIXCHAR(name,lenN,pN);
  *ierr = MeshGetStratumSize((Mesh) PetscToPointer(mesh),pN, *value, size);
  FREECHAR(name,pN);
}

void PETSC_STDCALL  meshgetsectionreal_(Mesh mesh, CHAR name PETSC_MIXED_LEN(lenN), SectionReal *section, int *ierr PETSC_END_LEN(lenN)){
  char *pN;
  FIXCHAR(name,lenN,pN);
  *ierr = MeshGetSectionReal((Mesh) PetscToPointer(mesh), pN, section);
  FREECHAR(name,pN);
}

void PETSC_STDCALL  meshgetsectionint_(Mesh mesh, CHAR name PETSC_MIXED_LEN(lenN), SectionInt *section, int *ierr PETSC_END_LEN(lenN)){
  char *pN;
  FIXCHAR(name,lenN,pN);
  *ierr = MeshGetSectionInt((Mesh) PetscToPointer(mesh), pN, section);
  FREECHAR(name,pN);
}

void PETSC_STDCALL  meshgetmatrix_(Mesh mesh, CHAR mattype PETSC_MIXED_LEN(lenN), Mat *J, int *ierr PETSC_END_LEN(lenN)){
  char *pN;
  FIXCHAR(mattype,lenN,pN);
  *ierr = MeshGetMatrix((Mesh) PetscToPointer(mesh), pN, J);
  FREECHAR(mattype,pN);
}

void PETSC_STDCALL  meshcreatematrix_(Mesh mesh, SectionReal section, CHAR mattype PETSC_MIXED_LEN(lenN), Mat *J, int *ierr PETSC_END_LEN(lenN)){
  char *pN;
  FIXCHAR(mattype,lenN,pN);
  *ierr = MeshCreateMatrix((Mesh) PetscToPointer(mesh), (SectionReal) PetscToPointer(section), pN, J);
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

void PETSC_STDCALL  alestageprintmemory_(CHAR name PETSC_MIXED_LEN(lenN), int *ierr PETSC_END_LEN(lenN)){
  ALE::MemoryLogger& logger = ALE::MemoryLogger::singleton();
  char *pN;

  FIXCHAR(name,lenN,pN);
  *ierr = PetscPrintf(PETSC_COMM_WORLD, "%s %d allocations %d bytes\n", pN, logger.getNumAllocations(pN), logger.getAllocationTotal(pN));
  *ierr = PetscPrintf(PETSC_COMM_WORLD, "%s %d deallocations %d bytes\n", pN, logger.getNumDeallocations(pN), logger.getDeallocationTotal(pN));
  FREECHAR(name,pN);
}

EXTERN_C_END
