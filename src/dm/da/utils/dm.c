#define PETSCDM_DLL
 
#include "private/dmimpl.h"     /*I      "petscda.h"     I*/

/*
   Provides an interface for functionality needed by the DAMG routines.
   Currently this interface is supported by the DA and DMComposite objects
  
   Note: this is actually no such thing as a DM object, rather it is 
   the common set of functions shared by DA and DMComposite.

*/

#undef __FUNCT__  
#define __FUNCT__ "DMDestroy"
/*@
    DMDestroy - Destroys a vector packer or DA.

    Collective on DM

    Input Parameter:
.   dm - the DM object to destroy

    Level: developer

.seealso DMView(), DMCreateGlobalVector(), DMGetInterpolation(), DMGetColoring(), DMGetMatrix()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMDestroy(DM dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = (*((PetscObject)dm)->bops->destroy)((PetscObject)dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMView"
/*@
    DMView - Views a vector packer or DA.

    Collective on DM

    Input Parameter:
+   dm - the DM object to view
-   v - the viewer

    Level: developer

.seealso DMDestroy(), DMCreateGlobalVector(), DMGetInterpolation(), DMGetColoring(), DMGetMatrix()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMView(DM dm,PetscViewer v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (((PetscObject)dm)->bops->view) {
    ierr = (*((PetscObject)dm)->bops->view)((PetscObject)dm,v);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCreateGlobalVector"
/*@
    DMCreateGlobalVector - Creates a global vector from a DA or DMComposite object

    Collective on DM

    Input Parameter:
.   dm - the DM object

    Output Parameter:
.   vec - the global vector

    Level: developer

.seealso DMDestroy(), DMView(), DMGetInterpolation(), DMGetColoring(), DMGetMatrix()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMCreateGlobalVector(DM dm,Vec *vec)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = (*dm->ops->createglobalvector)(dm,vec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCreateLocalVector"
/*@
    DMCreateLocalVector - Creates a local vector from a DA or DMComposite object

    Collective on DM

    Input Parameter:
.   dm - the DM object

    Output Parameter:
.   vec - the local vector

    Level: developer

.seealso DMDestroy(), DMView(), DMGetInterpolation(), DMGetColoring(), DMGetMatrix()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMCreateLocalVector(DM dm,Vec *vec)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = (*dm->ops->createlocalvector)(dm,vec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMGetInterpolation"
/*@
    DMGetInterpolation - Gets interpolation matrix between two DA or DMComposite objects

    Collective on DM

    Input Parameter:
+   dm1 - the DM object
-   dm2 - the second, finer DM object

    Output Parameter:
+  mat - the interpolation
-  vec - the scaling (optional)

    Level: developer

.seealso DMDestroy(), DMView(), DMCreateGlobalVector(), DMGetColoring(), DMGetMatrix()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMGetInterpolation(DM dm1,DM dm2,Mat *mat,Vec *vec)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = (*dm1->ops->getinterpolation)(dm1,dm2,mat,vec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMGetInjection"
/*@
    DMGetInjection - Gets injection matrix between two DA or DMComposite objects

    Collective on DM

    Input Parameter:
+   dm1 - the DM object
-   dm2 - the second, finer DM object

    Output Parameter:
.   ctx - the injection

    Level: developer

.seealso DMDestroy(), DMView(), DMCreateGlobalVector(), DMGetColoring(), DMGetMatrix(), DMGetInterpolation()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMGetInjection(DM dm1,DM dm2,VecScatter *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = (*dm1->ops->getinjection)(dm1,dm2,ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMGetColoring"
/*@
    DMGetColoring - Gets coloring for a DA or DMComposite

    Collective on DM

    Input Parameter:
+   dm - the DM object
.   ctype - IS_COLORING_GHOSTED or IS_COLORING_GLOBAL
-   matype - either MATAIJ or MATBAIJ

    Output Parameter:
.   coloring - the coloring

    Level: developer

.seealso DMDestroy(), DMView(), DMCreateGlobalVector(), DMGetInterpolation(), DMGetMatrix()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMGetColoring(DM dm,ISColoringType ctype,const MatType mtype,ISColoring *coloring)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!dm->ops->getcoloring) SETERRQ(PETSC_ERR_SUP,"No coloring for this type of DM yet");
  ierr = (*dm->ops->getcoloring)(dm,ctype,mtype,coloring);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMGetMatrix"
/*@C
    DMGetMatrix - Gets empty Jacobian for a DA or DMComposite

    Collective on DM

    Input Parameter:
+   dm - the DM object
-   mtype - Supported types are MATSEQAIJ, MATMPIAIJ, MATSEQBAIJ, MATMPIBAIJ, or
            any type which inherits from one of these (such as MATAIJ, MATLUSOL, etc.).

    Output Parameter:
.   mat - the empty Jacobian 

    Level: developer

.seealso DMDestroy(), DMView(), DMCreateGlobalVector(), DMGetInterpolation(), DMGetMatrix()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMGetMatrix(DM dm, const MatType mtype,Mat *mat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = (*dm->ops->getmatrix)(dm,mtype,mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMRefine"
/*@
    DMRefine - Refines a DM object

    Collective on DM

    Input Parameter:
+   dm - the DM object
-   comm - the communicator to contain the new DM object (or PETSC_NULL)

    Output Parameter:
.   dmf - the refined DM

    Level: developer

.seealso DMCoarsen(), DMDestroy(), DMView(), DMCreateGlobalVector(), DMGetInterpolation()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMRefine(DM dm,MPI_Comm comm,DM *dmf)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = (*dm->ops->refine)(dm,comm,dmf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMGlobalToLocalBegin"
/*@
    DMGlobalToLocalBegin - Begins updating local vectors from local vectors

    Collective on DM

    Input Parameters:
+   dm - the DM object
.   g - the global vector
.   mode - INSERT_VALUES or ADD_VALUES
-   l - the local vector


    Level: beginner

.seealso DMCoarsen(), DMDestroy(), DMView(), DMCreateGlobalVector(), DMGetInterpolation(), DMGlobalToLocalEnd(), DMLocalToGlobal()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMGlobalToLocalBegin(DM dm,Vec g,InsertMode mode,Vec l)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = (*dm->ops->globaltolocalbegin)(dm,g,mode,l);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMGlobalToLocalEnd"
/*@
    DMGlobalToLocalEnd - Ends updating local vectors from local vectors

    Collective on DM

    Input Parameters:
+   dm - the DM object
.   g - the global vector
.   mode - INSERT_VALUES or ADD_VALUES
-   l - the local vector


    Level: beginner

.seealso DMCoarsen(), DMDestroy(), DMView(), DMCreateGlobalVector(), DMGetInterpolation(), DMGlobalToLocalEnd(), DMLocalToGlobal()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMGlobalToLocalEnd(DM dm,Vec g,InsertMode mode,Vec l)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = (*dm->ops->globaltolocalend)(dm,g,mode,l);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMLocalToGlobal"
/*@
    DMLocalToGlobal - updates global vectors from local vectors

    Collective on DM

    Input Parameters:
+   dm - the DM object
.   g - the global vector
.   mode - INSERT_VALUES or ADD_VALUES
-   l - the local vector


    Level: beginner

.seealso DMCoarsen(), DMDestroy(), DMView(), DMCreateGlobalVector(), DMGetInterpolation(), DMGlobalToLocalEnd(), DMGlobalToLocalBegin()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMLocalToGlobal(DM dm,Vec g,InsertMode mode,Vec l)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = (*dm->ops->localtoglobal)(dm,g,mode,l);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCoarsen"
/*@
    DMCoarsen - Coarsens a DM object

    Collective on DM

    Input Parameter:
+   dm - the DM object
-   comm - the communicator to contain the new DM object (or PETSC_NULL)

    Output Parameter:
.   dmc - the coarsened DM

    Level: developer

.seealso DMRefine(), DMDestroy(), DMView(), DMCreateGlobalVector(), DMGetInterpolation()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMCoarsen(DM dm, MPI_Comm comm, DM *dmc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = (*dm->ops->coarsen)(dm, comm, dmc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMRefineHierarchy"
/*@C
    DMRefineHierarchy - Refines a DM object, all levels at once

    Collective on DM

    Input Parameter:
+   dm - the DM object
-   nlevels - the number of levels of refinement

    Output Parameter:
.   dmf - the refined DM hierarchy

    Level: developer

.seealso DMCoarsenHierarchy(), DMDestroy(), DMView(), DMCreateGlobalVector(), DMGetInterpolation()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMRefineHierarchy(DM dm,PetscInt nlevels,DM dmf[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (nlevels < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"nlevels cannot be negative");
  if (nlevels == 0) PetscFunctionReturn(0);
  if (dm->ops->refinehierarchy) {
    ierr = (*dm->ops->refinehierarchy)(dm,nlevels,dmf);CHKERRQ(ierr);
  } else if (dm->ops->refine) {
    PetscInt i;

    ierr = DMRefine(dm,((PetscObject)dm)->comm,&dmf[0]);CHKERRQ(ierr);
    for (i=1; i<nlevels; i++) {
      ierr = DMRefine(dmf[i-1],((PetscObject)dm)->comm,&dmf[i]);CHKERRQ(ierr);
    }
  } else {
    SETERRQ(PETSC_ERR_SUP,"No RefineHierarchy for this DM yet");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCoarsenHierarchy"
/*@C
    DMCoarsenHierarchy - Coarsens a DM object, all levels at once

    Collective on DM

    Input Parameter:
+   dm - the DM object
-   nlevels - the number of levels of coarsening

    Output Parameter:
.   dmc - the coarsened DM hierarchy

    Level: developer

.seealso DMRefineHierarchy(), DMDestroy(), DMView(), DMCreateGlobalVector(), DMGetInterpolation()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMCoarsenHierarchy(DM dm, PetscInt nlevels, DM dmc[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (nlevels < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"nlevels cannot be negative");
  if (nlevels == 0) PetscFunctionReturn(0);
  PetscValidPointer(dmc,3);
  if (dm->ops->coarsenhierarchy) {
    ierr = (*dm->ops->coarsenhierarchy)(dm, nlevels, dmc);CHKERRQ(ierr);
  } else if (dm->ops->coarsen) {
    PetscInt i;

    ierr = DMCoarsen(dm,((PetscObject)dm)->comm,&dmc[0]);CHKERRQ(ierr);
    for (i=1; i<nlevels; i++) {
      ierr = DMCoarsen(dmc[i-1],((PetscObject)dm)->comm,&dmc[i]);CHKERRQ(ierr);
    }
  } else {
    SETERRQ(PETSC_ERR_SUP,"No CoarsenHierarchy for this DM yet");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMGetAggregates"
/*@
   DMGetAggregates - Gets the aggregates that map between 
   grids associated with two DMs.

   Collective on DM

   Input Parameters:
+  dmc - the coarse grid DM
-  dmf - the fine grid DM

   Output Parameters:
.  rest - the restriction matrix (transpose of the projection matrix)

   Level: intermediate

.keywords: interpolation, restriction, multigrid 

.seealso: DMRefine(), DMGetInjection(), DMGetInterpolation()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DMGetAggregates(DM dmc, DM dmf, Mat *rest) 
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = (*dmc->ops->getaggregates)(dmc, dmf, rest);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
