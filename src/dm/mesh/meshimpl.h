#define PETSCDM_DLL

#if !defined(__mesh_h)
#define __mesh_h
 
#include "petscmesh.h"   /*I      "petscmesh.h"   I*/
#include "petscmat.h"    /*I      "petscmat.h"    I*/


typedef struct _MeshOps *MeshOps;
struct _MeshOps {
  PetscErrorCode (*view)(Mesh,PetscViewer);
  PetscErrorCode (*createglobalvector)(Mesh,Vec*);
  PetscErrorCode (*getcoloring)(Mesh,ISColoringType,ISColoring*);
  PetscErrorCode (*getmatrix)(Mesh,MatType,Mat*);
  PetscErrorCode (*getinterpolation)(Mesh,Mesh,Mat*,Vec*);
  PetscErrorCode (*refine)(Mesh,MPI_Comm,Mesh*);
};

struct _p_Mesh {
  PETSCHEADER(struct _MeshOps);
  ALE::Obj<ALE::Mesh> m;
  Vec                      globalvector;
  PetscInt                 bs,n,N,Nghosts,*ghosts;
  PetscInt                 d_nz,o_nz,*d_nnz,*o_nnz;
};


extern PetscEvent Mesh_View, Mesh_GetGlobalScatter, Mesh_restrictVector, Mesh_assembleVector,
                  Mesh_assembleVectorComplete, Mesh_assembleMatrix;

#endif
