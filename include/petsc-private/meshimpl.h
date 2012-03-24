#if !defined(_MESHIMPL_H)
#define _MESHIMPL_H

#include <petscmat.h>    /*I      "petscmat.h"    I*/
#include <petscdmmesh.h> /*I      "petscdmmesh.h"    I*/
#include "petsc-private/dmimpl.h"

typedef struct Sieve_Label *SieveLabel;
struct Sieve_Label {
  char      *name;           /* Label name */
  PetscInt   numStrata;      /* Number of integer values */
  PetscInt  *stratumValues;  /* Value of each stratum */
  PetscInt  *stratumOffsets; /* Offset of each stratum */
  PetscInt  *stratumSizes;   /* Size of each stratum */
  PetscInt  *points;         /* Points for each stratum, sorted after setup */
  SieveLabel next;           /* Linked list */
};

typedef struct {
  ALE::Obj<PETSC_MESH_TYPE> m;

  PetscSection         defaultSection;
  VecScatter           globalScatter;
  DMMeshLocalFunction1 lf;
  DMMeshLocalJacobian1 lj;

  /*-------- NEW_MESH_IMPL -------------*/
  PetscBool            useNewImpl;
  PetscInt             dim; /* Topological mesh dimension */
  PetscSF              sf;  /* SF for parallel point overlap */

  /*   Sieve */
  PetscSection         coneSection;    /* Layout of cones (inedges for DAG) */
  PetscInt             maxConeSize;    /* Cached for fast lookup */
  PetscInt            *cones;          /* Cone for each point */
  PetscInt            *coneOrientations; /* TODO */
  PetscSection         supportSection; /* Layout of cones (inedges for DAG) */
  PetscInt             maxSupportSize; /* Cached for fast lookup */
  PetscInt            *supports;       /* Cone for each point */
  PetscSection         coordSection;   /* Layout for coordinates */
  Vec                  coordinates;    /* Coordinate values */

  PetscInt            *meetTmpA,    *meetTmpB;    /* Work space for meet operation */
  PetscInt            *joinTmpA,    *joinTmpB;    /* Work space for join operation */
  PetscInt            *closureTmpA, *closureTmpB; /* Work space for closure operation */

  /* Labels */
  SieveLabel           labels;         /* Linked list of labels */
} DM_Mesh;

typedef struct {
  ALE::Obj<ALE::CartesianMesh> m;
} DM_Cartesian;

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
