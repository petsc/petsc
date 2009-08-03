#define PETSCDM_DLL

#if !defined(__mesh_h)
#define __mesh_h

#include "petscmat.h"    /*I      "petscmat.h"    I*/
#include "petscmesh.h"   /*I      "petscmesh.h"   I*/
#include "private/dmimpl.h"

typedef struct _MeshOps *MeshOps;
struct _MeshOps {
  DMOPS(Mesh)
};

struct _p_Mesh {
  PETSCHEADER(struct _MeshOps);
  ALE::Obj<PETSC_MESH_TYPE> m;
  VecScatter          globalScatter;
  PetscErrorCode    (*lf)(Mesh, SectionReal, SectionReal, void *);
  PetscErrorCode    (*lj)(Mesh, SectionReal, Mat, void *);

  void *data; // Implementation data
};

extern PetscCookie MESH_COOKIE;
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

extern PetscCookie SECTIONREAL_COOKIE;
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

extern PetscCookie SECTIONINT_COOKIE;
extern PetscLogEvent SectionInt_View;

#endif
