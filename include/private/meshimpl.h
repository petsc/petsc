#if !defined(_MESHIMPL_H)
#define _MESHIMPL_H

#include "petscmat.h"    /*I      "petscmat.h"    I*/
#include "private/dmimpl.h"

typedef struct {
  ALE::Obj<PETSC_MESH_TYPE> m;

  VecScatter           globalScatter;
  DMMeshLocalFunction1 lf;
  DMMeshLocalJacobian1 lj;
} DM_Mesh;

typedef struct {
  ALE::Obj<ALE::CartesianMesh> m;
} DM_Cartesian;

extern PetscLogEvent Mesh_View, Mesh_GetGlobalScatter, Mesh_restrictVector, Mesh_assembleVector, Mesh_assembleVectorComplete, Mesh_assembleMatrix, Mesh_updateOperator;

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

#endif /* _MESHIMPL_H */
