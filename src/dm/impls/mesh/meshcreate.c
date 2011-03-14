#define PETSCDM_DLL
#include "private/meshimpl.h"    /*I   "petscdmmesh.h"   I*/

#undef __FUNCT__
#define __FUNCT__ "DMSetFromOptions_Mesh"
PetscErrorCode  DMSetFromOptions_Mesh(DM dm)
{
  //DM_Mesh       *mesh = (DM_Mesh *) dm->data;
  char           typeName[256];
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscOptionsBegin(((PetscObject) dm)->comm, ((PetscObject) dm)->prefix, "DMMesh Options", "DMMesh");CHKERRQ(ierr);
    /* Handle DMMesh refinement */
    /* Handle associated vectors */
    if (!VecRegisterAllCalled) {ierr = VecRegisterAll(PETSC_NULL);CHKERRQ(ierr);}
    ierr = PetscOptionsList("-dm_vec_type", "Vector type used for created vectors", "DMSetVecType", VecList, dm->vectype, typeName, 256, &flg);CHKERRQ(ierr);
    if (flg) {
      ierr = DMSetVecType(dm, typeName);CHKERRQ(ierr);
    }
    /* process any options handlers added with PetscObjectAddOptionsHandler() */
    ierr = PetscObjectProcessOptionsHandlers((PetscObject) dm);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* External function declarations here */
extern PetscErrorCode DMCreateGlobalVector_Mesh(DM dm, Vec *gvec);
extern PetscErrorCode DMCreateLocalVector_Mesh(DM dm, Vec *lvec);
extern PetscErrorCode DMGetInterpolation_Mesh(DM dmCoarse, DM dmFine, Mat *interpolation, Vec *scaling);
extern PetscErrorCode DMGetMatrix_Mesh(DM dm, const MatType mtype, Mat *J);
extern PetscErrorCode DMRefine_Mesh(DM dm, MPI_Comm comm, DM *dmRefined);
extern PetscErrorCode DMCoarsenHierarchy_Mesh(DM dm, int numLevels, DM *coarseHierarchy);
extern PetscErrorCode DMDestroy_Mesh(DM dm);
extern PetscErrorCode DMView_Mesh(DM dm, PetscViewer viewer);

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "DMCreate_Mesh"
PetscErrorCode  DMCreate_Mesh(DM dm)
{
  DM_Mesh       *mesh;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscNewLog(dm, DM_Mesh, &mesh);CHKERRQ(ierr);
  dm->data = mesh;

  new(&mesh->m) ALE::Obj<PETSC_MESH_TYPE>(PETSC_NULL);

  mesh->globalScatter = PETSC_NULL;
  mesh->lf            = PETSC_NULL;
  mesh->lj            = PETSC_NULL;

  ierr = PetscStrallocpy(VECSTANDARD, &dm->vectype);CHKERRQ(ierr);
  dm->ops->globaltolocalbegin = 0;
  dm->ops->globaltolocalend   = 0;
  dm->ops->localtoglobalbegin = 0;
  dm->ops->localtoglobalend   = 0;
  dm->ops->createglobalvector = DMCreateGlobalVector_Mesh;
  dm->ops->createlocalvector  = DMCreateLocalVector_Mesh;
  dm->ops->getinterpolation   = DMGetInterpolation_Mesh;
  dm->ops->getcoloring        = 0;
  dm->ops->getelements        = 0;
  dm->ops->getmatrix          = DMGetMatrix_Mesh;
  dm->ops->refine             = DMRefine_Mesh;
  dm->ops->coarsen            = 0;
  dm->ops->refinehierarchy    = 0;
  dm->ops->coarsenhierarchy   = DMCoarsenHierarchy_Mesh;
  dm->ops->getinjection       = 0;
  dm->ops->getaggregates      = 0;
  dm->ops->destroy            = DMDestroy_Mesh;
  dm->ops->view               = DMView_Mesh;
  dm->ops->setfromoptions     = DMSetFromOptions_Mesh;
  dm->ops->setup              = 0;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "DMMeshCreate"
/*@
  DMMeshCreate - Creates a DMMesh object.

  Collective on MPI_Comm

  Input Parameter:
. comm - The communicator for the DMMesh object

  Output Parameter:
. mesh  - The DMMesh object

  Level: beginner

.keywords: DMMesh, create
@*/
PetscErrorCode  DMMeshCreate(MPI_Comm comm, DM *mesh)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(mesh,2);
  ierr = DMCreate(comm, mesh);CHKERRQ(ierr);
  ierr = DMSetType(*mesh, DMMESH);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
