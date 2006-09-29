#include "zpetsc.h"
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
#define meshdistribute_         MESHDISTRIBUTE
#define meshview_               MESHVIEW
#define fieldview_              FIELDVIEW
#define vertexsectioncreate_    VERTEXSECTIONCREATE
#define cellsectioncreate_      CELLSECTIONCREATE
#define restrictvector_         RESTRICTVECTOR
#define assemblevectorcomplete_ ASSEMBLEVECTORCOMPLETE
#define assemblevector_         ASSEMBLEVECTOR
#define assemblematrix_         ASSEMBLEMATRIX
#define writepcicerestart_      WRITEPCICERESTART
#define sectioncomplete_        SECTIONCOMPLETE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define meshcreatepcice_        meshcreatepcice
#define meshdistribute_         meshdistribute
#define meshview_               meshview
#define fieldview_              fieldview
#define vertexsectioncreate_    vertexsectioncreate
#define cellsectioncreate_      cellsectioncreate
#define restrictvector_         restrictvector
#define assemblevectorcomplete_ assemblevectorcomplete
#define assemblevector_         assemblevector
#define assemblematrix_         assemblematrix
#define writepcicerestart_      writepcicerestart
#define sectioncomplete_        sectioncomplete
#endif

/* Definitions of Fortran Wrapper routines */
EXTERN_C_BEGIN

void PETSC_STDCALL  meshcreatepcice_(MPI_Fint * comm, int *dim, CHAR coordFilename PETSC_MIXED_LEN(lenC), CHAR adjFilename PETSC_MIXED_LEN(lenA), CHAR bcFilename PETSC_MIXED_LEN(lenB), int *numBdFaces, int *numBdVertices, Mesh *mesh, PetscErrorCode *ierr PETSC_END_LEN(lenC) PETSC_END_LEN(lenA) PETSC_END_LEN(lenB))
{
  char *cF, *aF, *bF;
  FIXCHAR(coordFilename,lenC,cF);
  FIXCHAR(adjFilename,lenA,aF);
  FIXCHAR(bcFilename,lenB,bF);
  *ierr = MeshCreatePCICE(MPI_Comm_f2c( *(comm) ),*dim,cF,aF,bF,*numBdFaces,*numBdVertices,mesh);
  FREECHAR(coordFilename,cF);
  FREECHAR(adjFilename,aF);
  FREECHAR(bcFilename,bF);
}
void PETSC_STDCALL  meshdistribute_(Mesh serialMesh, Mesh *parallelMesh, PetscErrorCode *ierr)
{
  *ierr = MeshDistribute((Mesh) PetscToPointer(serialMesh),parallelMesh);
}
void PETSC_STDCALL  meshview_(Mesh mesh, PetscViewer viewer, PetscErrorCode *ierr)
{
  *ierr = MeshView((Mesh) PetscToPointer(mesh),(PetscViewer) PetscToPointer(viewer));
}
void PETSC_STDCALL  fieldview_(Mesh mesh, CHAR name PETSC_MIXED_LEN(len), PetscViewer viewer, PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *nF;
  FIXCHAR(name,len,nF);
  *ierr = FieldView((Mesh) PetscToPointer(mesh), nF,(PetscViewer) PetscToPointer(viewer));
  FREECHAR(name,nF);
}
void PETSC_STDCALL  vertexsectioncreate_(Mesh mesh, CHAR name PETSC_MIXED_LEN(len), PetscInt *fiberDim, int *ierr PETSC_END_LEN(len)){
  char *nF;
  FIXCHAR(name,len,nF);
  *ierr = VertexSectionCreate((Mesh) PetscToPointer(mesh), nF, *fiberDim);
  FREECHAR(name,nF);
}
void PETSC_STDCALL  cellsectioncreate_(Mesh mesh, CHAR name PETSC_MIXED_LEN(len), PetscInt *fiberDim, int *ierr PETSC_END_LEN(len)){
  char *nF;
  FIXCHAR(name,len,nF);
  *ierr = CellSectionCreate((Mesh) PetscToPointer(mesh), nF, *fiberDim);
  FREECHAR(name,nF);
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
void PETSC_STDCALL  sectioncomplete_(Mesh mesh, CHAR name PETSC_MIXED_LEN(len), int *ierr PETSC_END_LEN(len)){
  char *nF;
  FIXCHAR(name,len,nF);
  *ierr = SectionComplete((Mesh) PetscToPointer(mesh), nF);
  FREECHAR(name,nF);
}

EXTERN_C_END
