#define PETSCDM_DLL

#if !defined(__mesh_h)
#define __mesh_h

#include "petscmat.h"    /*I      "petscmat.h"    I*/
#include "petscmesh.h"   /*I      "petscmesh.h"   I*/
#include "src/dm/dmimpl.h"

typedef struct _MeshOps *MeshOps;
struct _MeshOps {
  DMOPS(Mesh)
};

struct _p_Mesh {
  PETSCHEADER(struct _MeshOps);
  ALE::Obj<ALE::Mesh> m;
  VecScatter          globalScatter;
  PetscErrorCode    (*lf)(Mesh, SectionReal, SectionReal, void *);
  PetscErrorCode    (*lj)(Mesh, SectionReal, Mat, void *);
  ALE::Obj<ALECompat::Mesh> mcompat;

  void *data; // Implementation data
};

extern PetscCookie MESH_COOKIE;
extern PetscEvent Mesh_View, Mesh_GetGlobalScatter, Mesh_restrictVector, Mesh_assembleVector,
                  Mesh_assembleVectorComplete, Mesh_assembleMatrix, Mesh_updateOperator;

typedef struct _SectionRealOps *SectionRealOps;
struct _SectionRealOps {
  PetscErrorCode (*view)(SectionReal,PetscViewer);
  PetscErrorCode (*restrict)(SectionReal,int,PetscScalar**);
  PetscErrorCode (*update)(SectionReal,int,const PetscScalar*);
};

struct _p_SectionReal {
  PETSCHEADER(struct _SectionRealOps);
  ALE::Obj<ALE::Mesh::real_section_type> s;
  ALE::Obj<ALE::Mesh> b;
};

extern PetscCookie SECTIONREAL_COOKIE;
extern PetscEvent SectionReal_View;

typedef struct _SectionIntOps *SectionIntOps;
struct _SectionIntOps {
  PetscErrorCode (*view)(SectionInt,PetscViewer);
  PetscErrorCode (*restrict)(SectionInt,int,PetscInt**);
  PetscErrorCode (*update)(SectionInt,int,const PetscInt*);
};

struct _p_SectionInt {
  PETSCHEADER(struct _SectionIntOps);
  ALE::Obj<ALE::Mesh::int_section_type> s;
  ALE::Obj<ALE::Mesh> b;
};

extern PetscCookie SECTIONINT_COOKIE;
extern PetscEvent SectionInt_View;

#endif
