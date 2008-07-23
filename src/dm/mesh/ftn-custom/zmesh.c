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
#define meshcreatepflotran_     MESHCREATEPFLOTRAN
#define meshcreatepcice_        MESHCREATEPCICE
#define meshcreateexodus_       MESHCREATEEXODUS
#define meshdistribute_         MESHDISTRIBUTE
#define meshview_               MESHVIEW
#define meshgetvertexsectionreal_   MESHGETVERTEXSECTIONREAL
#define vertexsectionrealcreate_ VERTEXSECTIONREALCREATE
#define vertexsectionintcreate_  VERTEXSECTIONINTCREATE
#define cellsectionrealcreate_   CELLSECTIONREALCREATE
#define bcsectionrealcreate_     BCSECTIONREALCREATE
#define restrictvector_         RESTRICTVECTOR
#define assemblevectorcomplete_ ASSEMBLEVECTORCOMPLETE
#define assemblevector_         ASSEMBLEVECTOR
#define assemblematrix_         ASSEMBLEMATRIX
#define writepcicerestart_      WRITEPCICERESTART
#define meshgetstratumsize_     MESHGETSTRATUMSIZE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define meshcreatepflotran_     meshcreatepflotran_
#define meshcreatepcice_        meshcreatepcice
#define meshcreateexodus_       meshcreateexodus
#define meshdistribute_         meshdistribute
#define meshview_               meshview
#define meshgetvertexsectionreal_   meshgetvertexsectionreal
#define vertexsectionrealcreate_ vertexsectionrealcreate
#define vertexsectionintcreate_  vertexsectionintcreate
#define cellsectionrealcreate_   cellsectionrealcreate
#define bcsectionrealcreate_     bcsectionrealcreate
#define restrictvector_         restrictvector
#define assemblevectorcomplete_ assemblevectorcomplete
#define assemblevector_         assemblevector
#define assemblematrix_         assemblematrix
#define writepcicerestart_      writepcicerestart
#define meshgetstratumsize_     meshgetstratumsize
#endif

/* Definitions of Fortran Wrapper routines */
EXTERN_C_BEGIN

void PETSC_STDCALL  meshcreatepflotran(MPI_Fint * comm, int *dim, CHAR hdf5Filename PETSC_MIXED_LEN(lenG), PetscTruth *interpolate, Mesh *mesh, PetscErrorCode *ierr PETSC_END_LEN(lenG))
{
  char *gF;
  FIXCHAR(hdf5Filename,lenG,gF);
  *ierr = MeshCreatePFLOTRAN(MPI_Comm_f2c( *(comm) ),*dim,gF,*interpolate,mesh);
  FREECHAR(hdf5Filename,gF);
}
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
void PETSC_STDCALL  meshview_(Mesh mesh, PetscViewer viewer, PetscErrorCode *ierr)
{
  *ierr = MeshView((Mesh) PetscToPointer(mesh),(PetscViewer) PetscToPointer(viewer));
}
void PETSC_STDCALL  meshgetvertexsectionreal_(Mesh mesh, PetscInt *fiberDim, SectionReal *section, int *ierr){
  *ierr = MeshGetVertexSectionReal((Mesh) PetscToPointer(mesh), *fiberDim, section);
}
void PETSC_STDCALL  vertexsectionrealcreate_(Mesh mesh, CHAR name PETSC_MIXED_LEN(lenN), PetscInt *fiberDim, int *ierr PETSC_END_LEN(lenN)){
  SectionReal section;
  char *pN;
  FIXCHAR(name,lenN,pN);
  *ierr = MeshGetVertexSectionReal((Mesh) PetscToPointer(mesh), *fiberDim, &section);
  *ierr = PetscObjectSetName((PetscObject) section, pN);
  *ierr = MeshSetSectionReal((Mesh) PetscToPointer(mesh), section);
  *ierr = SectionRealDestroy(section);
  FREECHAR(name,pN);
}
void PETSC_STDCALL  vertexsectionintcreate_(Mesh mesh, CHAR name PETSC_MIXED_LEN(lenN), PetscInt *fiberDim, int *ierr PETSC_END_LEN(lenN)){
  SectionInt section;
  char *pN;
  FIXCHAR(name,lenN,pN);
  *ierr = MeshGetVertexSectionInt((Mesh) PetscToPointer(mesh), *fiberDim, &section);
  *ierr = PetscObjectSetName((PetscObject) section, pN);
  *ierr = MeshSetSectionInt((Mesh) PetscToPointer(mesh), section);
  *ierr = SectionIntDestroy(section);
  FREECHAR(name,pN);
}
void PETSC_STDCALL  cellsectionrealcreate_(Mesh mesh, CHAR name PETSC_MIXED_LEN(lenN), PetscInt *fiberDim, int *ierr PETSC_END_LEN(lenN)){
  SectionReal section;
  char *pN;
  FIXCHAR(name,lenN,pN);
  *ierr = MeshGetCellSectionReal((Mesh) PetscToPointer(mesh), *fiberDim, &section);
  *ierr = PetscObjectSetName((PetscObject) section, pN);
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
void PETSC_STDCALL  assemblematrix_(Mat A,PetscInt *e,PetscScalar v[],InsertMode *mode, int *__ierr ){
*__ierr = assembleMatrix(
	(Mat)PetscToPointer((A) ),*e,v,*mode);
}
void PETSC_STDCALL  writepcicerestart_(Mesh mesh, PetscViewer viewer, int *ierr){
  *ierr = WritePCICERestart((Mesh) PetscToPointer(mesh), (PetscViewer) PetscToPointer(viewer));
}
void PETSC_STDCALL  meshgetstratumsize_(Mesh mesh, CHAR name PETSC_MIXED_LEN(lenN), PetscInt *value, PetscInt *size, int *ierr PETSC_END_LEN(lenN)){
  char *pN;
  FIXCHAR(name,lenN,pN);
  *ierr = MeshGetStratumSize((Mesh) PetscToPointer(mesh),pN, *value, size);
  FREECHAR(name,pN);
}

EXTERN_C_END
