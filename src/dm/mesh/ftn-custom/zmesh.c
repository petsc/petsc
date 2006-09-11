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
#define meshcreatepcice_ MESHCREATEPCICE
#define vertexsectioncreate_ VERTEXSECTIONCREATE
#define cellsectioncreate_ CELLSECTIONCREATE
#define restrictvector_ RESTRICTVECTOR
#define assemblevectorcomplete_ ASSEMBLEVECTORCOMPLETE
#define assemblevector_ ASSEMBLEVECTOR
#define assemblematrix_ ASSEMBLEMATRIX
#define writepcicerestart_ WRITEPCICERESTART
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define meshcreatepcice_ meshcreatepcice
#define vertexsectioncreate_ vertexsectioncreate
#define cellsectioncreate_ cellsectioncreate
#define restrictvector_ restrictvector
#define assemblevectorcomplete_ assemblevectorcomplete
#define assemblevector_ assemblevector
#define assemblematrix_ assemblematrix
#define writepcicerestart_ writepcicerestart
#endif

/* Definitions of Fortran Wrapper routines */
EXTERN_C_BEGIN

void PETSC_STDCALL  meshcreatepcice_(MPI_Fint * comm, int *dim, CHAR coordFilename PETSC_MIXED_LEN(lenC), CHAR adjFilename PETSC_MIXED_LEN(lenA), Mesh *mesh, PetscErrorCode *ierr PETSC_END_LEN(lenC) PETSC_END_LEN(lenA))
{
  char *cF, *aF;
  FIXCHAR(coordFilename,lenC,cF);
  FIXCHAR(adjFilename,lenA,aF);
  *ierr = MeshCreatePCICE(MPI_Comm_f2c( *(comm) ),*dim,cF,aF,mesh);
  FREECHAR(coordFilename,cF);
  FREECHAR(adjFilename,aF);
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

EXTERN_C_END
