#define PETSCDM_DLL

#if !defined(__mesh_h)
#define __mesh_h

#include "petscmat.h"    /*I      "petscmat.h"    I*/
#include "petscmesh.h"   /*I      "petscmesh.h"   I*/
#include "private/dmimpl.h"

typedef struct _MeshOps *MeshOps;
struct _MeshOps {
  PetscErrorCode (*view)(Mesh,PetscViewer); 
  PetscErrorCode (*setfromoptions)(Mesh); 
  PetscErrorCode (*setup)(Mesh); 
  PetscErrorCode (*createglobalvector)(Mesh,Vec*);
  PetscErrorCode (*createlocalvector)(Mesh,Vec*);

  PetscErrorCode (*getcoloring)(Mesh,ISColoringType,const MatType,ISColoring*);	
  PetscErrorCode (*getmatrix)(Mesh, const MatType,Mat*);
  PetscErrorCode (*getinterpolation)(Mesh,Mesh,Mat*,Vec*);
  PetscErrorCode (*getaggregates)(Mesh,Mesh,Mat*);
  PetscErrorCode (*getinjection)(Mesh,Mesh,VecScatter*);

  PetscErrorCode (*refine)(Mesh,MPI_Comm,Mesh*);
  PetscErrorCode (*coarsen)(Mesh,MPI_Comm,Mesh*);
  PetscErrorCode (*refinehierarchy)(Mesh,PetscInt,Mesh*);
  PetscErrorCode (*coarsenhierarchy)(Mesh,PetscInt,Mesh*);

  PetscErrorCode (*forminitialguess)(Mesh,PetscErrorCode (*)(void),Vec,void*);
  PetscErrorCode (*formfunction)(Mesh,PetscErrorCode (*)(void),Vec,Vec);

  PetscErrorCode (*globaltolocalbegin)(Mesh,Vec,InsertMode,Vec);		
  PetscErrorCode (*globaltolocalend)(Mesh,Vec,InsertMode,Vec); 
  PetscErrorCode (*localtoglobalbegin)(Mesh,Vec,InsertMode,Vec); 
  PetscErrorCode (*localtoglobalend)(Mesh,Vec,InsertMode,Vec); 

  PetscErrorCode (*getelements)(Mesh,PetscInt*,const PetscInt*[]);   
  PetscErrorCode (*restoreelements)(Mesh,PetscInt*,const PetscInt*[]); 

  PetscErrorCode (*initialguess)(Mesh,Vec); 
  PetscErrorCode (*function)(Mesh,Vec,Vec);			
  PetscErrorCode (*functionj)(Mesh,Vec,Vec);			
  PetscErrorCode (*jacobian)(Mesh,Vec,Mat,Mat,MatStructure*);	

  PetscErrorCode (*destroy)(Mesh);
};

struct _p_Mesh {
  PETSCHEADER(struct _MeshOps);
  ALE::Obj<PETSC_MESH_TYPE> m;
  VecScatter          globalScatter;
  PetscErrorCode    (*lf)(Mesh, SectionReal, SectionReal, void *);
  PetscErrorCode    (*lj)(Mesh, SectionReal, Mat, void *);

  void *data; // Implementation data
};

extern PetscClassId MESH_CLASSID;
extern PetscLogEvent Mesh_View, Mesh_GetGlobalScatter, Mesh_restrictVector, Mesh_assembleVector,
                  Mesh_assembleVectorComplete, Mesh_assembleMatrix, Mesh_updateOperator;

typedef struct _SectionRealOps *SectionRealOps;
struct _SectionRealOps {
  PetscErrorCode (*view)(SectionReal,PetscViewer);
  PetscErrorCode (*restrictClosure)(SectionReal,int,PetscScalar**);
  PetscErrorCode (*update)(SectionReal,int,const PetscScalar*,InsertMode);
};

struct _p_SectionReal {
  PETSCHEADER(struct _SectionRealOps);
  ALE::Obj<PETSC_MESH_TYPE::real_section_type> s;
  ALE::Obj<PETSC_MESH_TYPE> b;
};

extern PetscClassId SECTIONREAL_CLASSID;
extern PetscLogEvent SectionReal_View;

typedef struct _SectionIntOps *SectionIntOps;
struct _SectionIntOps {
  PetscErrorCode (*view)(SectionInt,PetscViewer);
  PetscErrorCode (*restrictClosure)(SectionInt,int,PetscInt**);
  PetscErrorCode (*update)(SectionInt,int,const PetscInt*,InsertMode);
};

struct _p_SectionInt {
  PETSCHEADER(struct _SectionIntOps);
  ALE::Obj<PETSC_MESH_TYPE::int_section_type> s;
  ALE::Obj<PETSC_MESH_TYPE> b;
};

extern PetscClassId SECTIONINT_CLASSID;
extern PetscLogEvent SectionInt_View;

#endif
