#define PETSCDM_DLL
 
#include "private/dmimpl.h"     /*I      "petscdm.h"     I*/

#undef __FUNCT__  
#define __FUNCT__ "DMCreate"
/*@
  DMCreate - Creates an empty vector object. The type can then be set with DMetType().

   If you never  call DMSetType()  it will generate an 
   error when you try to use the vector.

  Collective on MPI_Comm

  Input Parameter:
. comm - The communicator for the DM object

  Output Parameter:
. dm - The DM object

  Level: beginner

.seealso: DMSetType(), DMDA, DMSLICED, DMCOMPOSITE
@*/
PetscErrorCode PETSCVEC_DLLEXPORT DMCreate(MPI_Comm comm,DM *dm)
{
  DM             v;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(dm,2);
  *dm = PETSC_NULL;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = DMInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif

  ierr = PetscHeaderCreate(v, _p_DM, struct _DMOps, DM_CLASSID, -1, "DM", comm, DMDestroy, DMView);CHKERRQ(ierr);
  ierr = PetscMemzero(v->ops, sizeof(struct _DMOps));CHKERRQ(ierr);

  v->ltogmap      = PETSC_NULL;
  v->ltogmapb     = PETSC_NULL;
  v->bs           = 1;

  *dm = v;
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "DMSetVecType"
/*@C
       DMSetVecType - Sets the type of vector created with DMCreateLocalVector() and DMCreateGlobalVector()

   Logically Collective on DMDA

   Input Parameter:
+  da - initial distributed array
.  ctype - the vector type, currently either VECSTANDARD or VECCUDA

   Options Database:
.   -da_vec_type ctype

   Level: intermediate

.seealso: DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), DMDestroy(), DMDA, DMDAInterpolationType, VecType
@*/
PetscErrorCode PETSCDM_DLLEXPORT DMSetVecType(DM da,const VecType ctype)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  ierr = PetscFree(da->vectype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(ctype,&da->vectype);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMSetOptionsPrefix"
/*@C
   DMSetOptionsPrefix - Sets the prefix used for searching for all 
   DMDA options in the database.

   Logically Collective on DMDA

   Input Parameter:
+  da - the DMDA context
-  prefix - the prefix to prepend to all option names

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

   Level: advanced

.keywords: DMDA, set, options, prefix, database

.seealso: DMSetFromOptions()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DMSetOptionsPrefix(DM dm,const char prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)dm,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMDestroy"
/*@
    DMDestroy - Destroys a vector packer or DMDA.

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
  ierr = (*dm->ops->destroy)(dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMSetUp"
/*@
    DMSetUp - sets up the data structures inside a DM object

    Collective on DM

    Input Parameter:
.   dm - the DM object to setup

    Level: developer

.seealso DMView(), DMCreateGlobalVector(), DMGetInterpolation(), DMGetColoring(), DMGetMatrix()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMSetUp(DM dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (dm->ops->setup) {
    ierr = (*dm->ops->setup)(dm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMSetFromOptions"
/*@
    DMSetFromOptions - sets parameters in a DM from the options database

    Collective on DM

    Input Parameter:
.   dm - the DM object to set options for

    Level: developer

.seealso DMView(), DMCreateGlobalVector(), DMGetInterpolation(), DMGetColoring(), DMGetMatrix()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMSetFromOptions(DM dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (dm->ops->setfromoptions) {
    ierr = (*dm->ops->setfromoptions)(dm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMView"
/*@
    DMView - Views a vector packer or DMDA.

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
 if (!v) {
    ierr = PetscViewerASCIIGetStdout(((PetscObject)dm)->comm,&v);CHKERRQ(ierr);
  }
  if (dm->ops->view) {
    ierr = (*dm->ops->view)(dm,v);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCreateGlobalVector"
/*@
    DMCreateGlobalVector - Creates a global vector from a DMDA or DMComposite object

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
    DMCreateLocalVector - Creates a local vector from a DMDA or DMComposite object

    Not Collective

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
#define __FUNCT__ "DMGetLocalToGlobalMapping"
/*@
   DMGetLocalToGlobalMapping - Accesses the local-to-global mapping in a DM.

   Collective on DM

   Input Parameter:
.  dm - the DM that provides the mapping

   Output Parameter:
.  ltog - the mapping

   Level: intermediate

   Notes:
   This mapping can then be used by VecSetLocalToGlobalMapping() or
   MatSetLocalToGlobalMapping().

.seealso: DMCreateLocalVector(), DMGetLocalToGlobalMappingBlock()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DMGetLocalToGlobalMapping(DM dm,ISLocalToGlobalMapping *ltog)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(ltog,2);
  if (!dm->ltogmap) {
    if (!dm->ops->createlocaltoglobalmapping) SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_SUP,"DM can not create LocalToGlobalMapping");
    ierr = (*dm->ops->createlocaltoglobalmapping)(dm);CHKERRQ(ierr);
  }
  *ltog = dm->ltogmap;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMGetLocalToGlobalMappingBlock"
/*@
   DMGetLocalToGlobalMappingBlock - Accesses the blocked local-to-global mapping in a DM.

   Collective on DM

   Input Parameter:
.  da - the distributed array that provides the mapping

   Output Parameter:
.  ltog - the block mapping

   Level: intermediate

   Notes:
   This mapping can then be used by VecSetLocalToGlobalMappingBlock() or
   MatSetLocalToGlobalMappingBlock().

.seealso: DMCreateLocalVector(), DMGetLocalToGlobalMapping(), DMGetBlockSize(), VecSetBlockSize(), MatSetBlockSize()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DMGetLocalToGlobalMappingBlock(DM dm,ISLocalToGlobalMapping *ltog)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(ltog,2);
  if (!dm->ltogmapb) {
    PetscInt bs;
    ierr = DMGetBlockSize(dm,&bs);CHKERRQ(ierr);
    if (bs > 1) {
      if (!dm->ops->createlocaltoglobalmappingblock) SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_SUP,"DM can not create LocalToGlobalMappingBlock");
      ierr = (*dm->ops->createlocaltoglobalmappingblock)(dm);CHKERRQ(ierr);
    } else {
      ierr = DMGetLocalToGlobalMapping(dm,&dm->ltogmapb);CHKERRQ(ierr);
      ierr = PetscObjectReference((PetscObject)dm->ltogmapb);CHKERRQ(ierr);
    }
  }
  *ltog = dm->ltogmapb;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMGetBlockSize"
/*@
   DMGetBlockSize - Gets the inherent block size associated with a DM

   Not Collective

   Input Parameter:
.  dm - the DM with block structure

   Output Parameter:
.  bs - the block size, 1 implies no exploitable block structure

   Level: intermediate

.seealso: ISCreateBlock(), VecSetBlockSize(), MatSetBlockSize(), DMGetLocalToGlobalMappingBlock()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DMGetBlockSize(DM dm,PetscInt *bs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(bs,2);
  if (dm->bs < 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"DM does not have enough information to provide a block size yet");
  *bs = dm->bs;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMGetInterpolation"
/*@
    DMGetInterpolation - Gets interpolation matrix between two DMDA or DMComposite objects

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
    DMGetInjection - Gets injection matrix between two DMDA or DMComposite objects

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
    DMGetColoring - Gets coloring for a DMDA or DMComposite

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
  if (!dm->ops->getcoloring) SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_SUP,"No coloring for this type of DM yet");
  ierr = (*dm->ops->getcoloring)(dm,ctype,mtype,coloring);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMGetMatrix"
/*@C
    DMGetMatrix - Gets empty Jacobian for a DMDA or DMComposite

    Collective on DM

    Input Parameter:
+   dm - the DM object
-   mtype - Supported types are MATSEQAIJ, MATMPIAIJ, MATSEQBAIJ, MATMPIBAIJ, or
            any type which inherits from one of these (such as MATAIJ)

    Output Parameter:
.   mat - the empty Jacobian 

    Level: developer

    Notes: This properly preallocates the number of nonzeros in the sparse matrix so you 
       do not need to do it yourself. 

       By default it also sets the nonzero structure and puts in the zero entries. To prevent setting 
       the nonzero pattern call DMDASetMatPreallocateOnly()

       For structured grid problems, when you call MatView() on this matrix it is displayed using the global natural ordering, NOT in the ordering used
       internally by PETSc.

       For structured grid problems, in general it is easiest to use MatSetValuesStencil() or MatSetValuesLocal() to put values into the matrix because MatSetValues() requires 
       the indices for the global numbering for DMDAs which is complicated.

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
    DMGlobalToLocalBegin - Begins updating local vectors from global vector

    Neighbor-wise Collective on DM

    Input Parameters:
+   dm - the DM object
.   g - the global vector
.   mode - INSERT_VALUES or ADD_VALUES
-   l - the local vector


    Level: beginner

.seealso DMCoarsen(), DMDestroy(), DMView(), DMCreateGlobalVector(), DMGetInterpolation(), DMGlobalToLocalEnd(), DMLocalToGlobalBegin()

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
    DMGlobalToLocalEnd - Ends updating local vectors from global vector

    Neighbor-wise Collective on DM

    Input Parameters:
+   dm - the DM object
.   g - the global vector
.   mode - INSERT_VALUES or ADD_VALUES
-   l - the local vector


    Level: beginner

.seealso DMCoarsen(), DMDestroy(), DMView(), DMCreateGlobalVector(), DMGetInterpolation(), DMGlobalToLocalEnd(), DMLocalToGlobalBegin()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMGlobalToLocalEnd(DM dm,Vec g,InsertMode mode,Vec l)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = (*dm->ops->globaltolocalend)(dm,g,mode,l);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMLocalToGlobalBegin"
/*@
    DMLocalToGlobalBegin - updates global vectors from local vectors

    Neighbor-wise Collective on DM

    Input Parameters:
+   dm - the DM object
.   l - the local vector
.   mode - if INSERT_VALUES then no parallel communication is used, if ADD_VALUES then all ghost points from the same base point accumulate into that
           base point. 
- - the global vector

    Notes: In the ADD_VALUES case you normally would zero the receiving vector before beginning this operation. If you would like to simply add the non-ghosted values in the local
           array into the global array you need to either (1) zero the ghosted locations and use ADD_VALUES or (2) use INSERT_VALUES into a work global array and then add the work 
           global array to the final global array with VecAXPY().

    Level: beginner

.seealso DMCoarsen(), DMDestroy(), DMView(), DMCreateGlobalVector(), DMGetInterpolation(), DMGlobalToLocalEnd(), DMGlobalToLocalBegin()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMLocalToGlobalBegin(DM dm,Vec l,InsertMode mode,Vec g)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = (*dm->ops->localtoglobalbegin)(dm,l,mode,g);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMLocalToGlobalEnd"
/*@
    DMLocalToGlobalEnd - updates global vectors from local vectors

    Neighbor-wise Collective on DM

    Input Parameters:
+   dm - the DM object
.   l - the local vector
.   mode - INSERT_VALUES or ADD_VALUES
-   g - the global vector


    Level: beginner

.seealso DMCoarsen(), DMDestroy(), DMView(), DMCreateGlobalVector(), DMGetInterpolation(), DMGlobalToLocalEnd(), DMGlobalToLocalEnd()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMLocalToGlobalEnd(DM dm,Vec l,InsertMode mode,Vec g)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = (*dm->ops->localtoglobalend)(dm,l,mode,g);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMComputeJacobianDefault"
/*@
    DMComputeJacobianDefault - computes the Jacobian using the DMComputeFunction() if Jacobian computer is not provided

    Collective on DM

    Input Parameter:
+   dm - the DM object 
.   x - location to compute Jacobian at; may be ignored for linear problems
.   A - matrix that defines the operator for the linear solve
-   B - the matrix used to construct the preconditioner

    Level: developer

.seealso DMView(), DMCreateGlobalVector(), DMGetInterpolation(), DMGetColoring(), DMGetMatrix(), DMGetContext(), DMSetInitialGuess(), 
         DMSetFunction()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMComputeJacobianDefault(DM dm,Vec x,Mat A,Mat B,MatStructure *stflag)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  *stflag = SAME_NONZERO_PATTERN;
  ierr  = MatFDColoringApply(B,dm->fd,x,stflag,dm);CHKERRQ(ierr);
  if (A != B) {
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
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
  (*dmc)->ops->initialguess = dm->ops->initialguess;
  (*dmc)->ops->function     = dm->ops->function;
  (*dmc)->ops->functionj    = dm->ops->functionj;
  if (dm->ops->jacobian != DMComputeJacobianDefault) {
    (*dmc)->ops->jacobian     = dm->ops->jacobian;
  }
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
  if (nlevels < 0) SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_ARG_OUTOFRANGE,"nlevels cannot be negative");
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
    SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_SUP,"No RefineHierarchy for this DM yet");
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
  if (nlevels < 0) SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_ARG_OUTOFRANGE,"nlevels cannot be negative");
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
    SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_SUP,"No CoarsenHierarchy for this DM yet");
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

#undef __FUNCT__  
#define __FUNCT__ "DMSetContext"
/*@
    DMSetContext - Set a user context into a DM object

    Not Collective

    Input Parameters:
+   dm - the DM object 
-   ctx - the user context

    Level: intermediate

.seealso DMView(), DMCreateGlobalVector(), DMGetInterpolation(), DMGetColoring(), DMGetMatrix(), DMGetContext()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMSetContext(DM dm,void *ctx)
{
  PetscFunctionBegin;
  dm->ctx = ctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMGetContext"
/*@
    DMGetContext - Gets a user context from a DM object

    Not Collective

    Input Parameter:
.   dm - the DM object 

    Output Parameter:
.   ctx - the user context

    Level: intermediate

.seealso DMView(), DMCreateGlobalVector(), DMGetInterpolation(), DMGetColoring(), DMGetMatrix(), DMGetContext()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMGetContext(DM dm,void **ctx)
{
  PetscFunctionBegin;
  *ctx = dm->ctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMSetInitialGuess"
/*@
    DMSetInitialGuess - sets a function to compute an initial guess vector entries for the solvers

    Logically Collective on DM

    Input Parameter:
+   dm - the DM object to destroy
-   f - the function to compute the initial guess

    Level: intermediate

.seealso DMView(), DMCreateGlobalVector(), DMGetInterpolation(), DMGetColoring(), DMGetMatrix(), DMGetContext(), DMSetFunction(), DMSetJacobian()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMSetInitialGuess(DM dm,PetscErrorCode (*f)(DM,Vec))
{
  PetscFunctionBegin;
  dm->ops->initialguess = f;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMSetFunction"
/*@
    DMSetFunction - sets a function to compute the right hand side vector entries for the KSP solver or nonlinear function for SNES

    Logically Collective on DM

    Input Parameter:
+   dm - the DM object 
-   f - the function to compute (use PETSC_NULL to cancel a previous function that was set)

    Level: intermediate

    Notes: This sets both the function for function evaluations and the function used to compute Jacobians via finite differences if no Jacobian 
           computer is provided with DMSetJacobian(). Canceling cancels the function, but not the function used to compute the Jacobian.

.seealso DMView(), DMCreateGlobalVector(), DMGetInterpolation(), DMGetColoring(), DMGetMatrix(), DMGetContext(), DMSetInitialGuess(),
         DMSetJacobian()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMSetFunction(DM dm,PetscErrorCode (*f)(DM,Vec,Vec))
{
  PetscFunctionBegin;
  dm->ops->function = f;
  if (f) {
    dm->ops->functionj = f;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMSetJacobian"
/*@
    DMSetJacobian - sets a function to compute the matrix entries for the KSP solver or Jacobian for SNES

    Logically Collective on DM

    Input Parameter:
+   dm - the DM object to destroy
-   f - the function to compute the matrix entries

    Level: intermediate

.seealso DMView(), DMCreateGlobalVector(), DMGetInterpolation(), DMGetColoring(), DMGetMatrix(), DMGetContext(), DMSetInitialGuess(), 
         DMSetFunction()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMSetJacobian(DM dm,PetscErrorCode (*f)(DM,Vec,Mat,Mat,MatStructure*))
{
  PetscFunctionBegin;
  dm->ops->jacobian = f;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMComputeInitialGuess"
/*@
    DMComputeInitialGuess - computes an initial guess vector entries for the KSP solvers

    Collective on DM

    Input Parameter:
+   dm - the DM object to destroy
-   x - the vector to hold the initial guess values

    Level: developer

.seealso DMView(), DMCreateGlobalVector(), DMGetInterpolation(), DMGetColoring(), DMGetMatrix(), DMGetContext(), DMSetRhs(), DMSetMat()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMComputeInitialGuess(DM dm,Vec x)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (!dm->ops->initialguess) SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_ARG_WRONGSTATE,"Need to provide function with DMSetInitialGuess()");
  ierr = (*dm->ops->initialguess)(dm,x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMHasInitialGuess"
/*@
    DMHasInitialGuess - does the DM object have an initial guess function

    Not Collective

    Input Parameter:
.   dm - the DM object to destroy

    Output Parameter:
.   flg - PETSC_TRUE if function exists

    Level: developer

.seealso DMView(), DMCreateGlobalVector(), DMGetInterpolation(), DMGetColoring(), DMGetMatrix(), DMGetContext(), DMSetFunction(), DMSetJacobian()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMHasInitialGuess(DM dm,PetscBool  *flg)
{
  PetscFunctionBegin;
  *flg =  (dm->ops->initialguess) ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMHasFunction"
/*@
    DMHasFunction - does the DM object have a function

    Not Collective

    Input Parameter:
.   dm - the DM object to destroy

    Output Parameter:
.   flg - PETSC_TRUE if function exists

    Level: developer

.seealso DMView(), DMCreateGlobalVector(), DMGetInterpolation(), DMGetColoring(), DMGetMatrix(), DMGetContext(), DMSetFunction(), DMSetJacobian()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMHasFunction(DM dm,PetscBool  *flg)
{
  PetscFunctionBegin;
  *flg =  (dm->ops->function) ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMHasJacobian"
/*@
    DMHasJacobian - does the DM object have a matrix function

    Not Collective

    Input Parameter:
.   dm - the DM object to destroy

    Output Parameter:
.   flg - PETSC_TRUE if function exists

    Level: developer

.seealso DMView(), DMCreateGlobalVector(), DMGetInterpolation(), DMGetColoring(), DMGetMatrix(), DMGetContext(), DMSetFunction(), DMSetJacobian()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMHasJacobian(DM dm,PetscBool  *flg)
{
  PetscFunctionBegin;
  *flg =  (dm->ops->jacobian) ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMComputeFunction"
/*@
    DMComputeFunction - computes the right hand side vector entries for the KSP solver or nonlinear function for SNES

    Collective on DM

    Input Parameter:
+   dm - the DM object to destroy
.   x - the location where the function is evaluationed, may be ignored for linear problems
-   b - the vector to hold the right hand side entries

    Level: developer

.seealso DMView(), DMCreateGlobalVector(), DMGetInterpolation(), DMGetColoring(), DMGetMatrix(), DMGetContext(), DMSetInitialGuess(),
         DMSetJacobian()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMComputeFunction(DM dm,Vec x,Vec b)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (!dm->ops->function) SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_ARG_WRONGSTATE,"Need to provide function with DMSetFunction()");
  if (!x) x = dm->x;
  ierr = (*dm->ops->function)(dm,x,b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "DMComputeJacobian"
/*@
    DMComputeJacobian - compute the matrix entries for the solver

    Collective on DM

    Input Parameter:
+   dm - the DM object 
.   x - location to compute Jacobian at; may be ignored for linear problems
.   A - matrix that defines the operator for the linear solve
-   B - the matrix used to construct the preconditioner

    Level: developer

.seealso DMView(), DMCreateGlobalVector(), DMGetInterpolation(), DMGetColoring(), DMGetMatrix(), DMGetContext(), DMSetInitialGuess(), 
         DMSetFunction()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMComputeJacobian(DM dm,Vec x,Mat A,Mat B,MatStructure *stflag)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!dm->ops->jacobian) {
    ISColoring    coloring;
    MatFDColoring fd;

    ierr = DMGetColoring(dm,IS_COLORING_GHOSTED,MATAIJ,&coloring);CHKERRQ(ierr);
    ierr = MatFDColoringCreate(B,coloring,&fd);CHKERRQ(ierr);
    ierr = ISColoringDestroy(coloring);CHKERRQ(ierr);
    ierr = MatFDColoringSetFunction(fd,(PetscErrorCode (*)(void))dm->ops->functionj,dm);CHKERRQ(ierr);
    dm->fd = fd;
    dm->ops->jacobian = DMComputeJacobianDefault;

    if (!dm->x) {
      ierr = MatGetVecs(B,&dm->x,PETSC_NULL);CHKERRQ(ierr);
    }
  }
  if (!x) x = dm->x;
  ierr = (*dm->ops->jacobian)(dm,x,A,B,stflag);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscFList DMList                       = PETSC_NULL;
PetscBool  DMRegisterAllCalled          = PETSC_FALSE;

#undef __FUNCT__  
#define __FUNCT__ "DMSetType"
/*@C
  DMSetType - Builds a DM, for a particular DM implementation.

  Collective on DM

  Input Parameters:
+ dm     - The DM object
- method - The name of the DM type

  Options Database Key:
. -dm_type <type> - Sets the DM type; use -help for a list of available types

  Notes:
  See "petsc/include/petscdm.h" for available DM types (for instance, DM1D, DM2D, or DM3D).

  Level: intermediate

.keywords: DM, set, type
.seealso: DMGetType(), DMCreate()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DMSetType(DM dm, const DMType method)
{
  PetscErrorCode (*r)(DM);
  PetscBool      match;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID,1);
  ierr = PetscTypeCompare((PetscObject) dm, method, &match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  if (!DMRegisterAllCalled) {ierr = DMRegisterAll(PETSC_NULL);CHKERRQ(ierr);}
  ierr = PetscFListFind(DMList, ((PetscObject)dm)->comm, method,(void (**)(void)) &r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown DM type: %s", method);

  if (dm->ops->destroy) {
    ierr = (*dm->ops->destroy)(dm);CHKERRQ(ierr);
  } 
  ierr = (*r)(dm);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)dm,method);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMGetType"
/*@C
  DMGetType - Gets the DM type name (as a string) from the DM.

  Not Collective

  Input Parameter:
. dm  - The DM

  Output Parameter:
. type - The DM type name

  Level: intermediate

.keywords: DM, get, type, name
.seealso: DMSetType(), DMCreate()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DMGetType(DM dm, const DMType *type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID,1);
  PetscValidCharPointer(type,2);
  if (!DMRegisterAllCalled) {
    ierr = DMRegisterAll(PETSC_NULL);CHKERRQ(ierr);
  }
  *type = ((PetscObject)dm)->type_name;
  PetscFunctionReturn(0);
}


/*--------------------------------------------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "DMRegister"
/*@C
  DMRegister - See DMRegisterDynamic()

  Level: advanced
@*/
PetscErrorCode PETSCDM_DLLEXPORT DMRegister(const char sname[], const char path[], const char name[], PetscErrorCode (*function)(DM))
{
  char fullname[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStrcpy(fullname, path);CHKERRQ(ierr);
  ierr = PetscStrcat(fullname, ":");CHKERRQ(ierr);
  ierr = PetscStrcat(fullname, name);CHKERRQ(ierr);
  ierr = PetscFListAdd(&DMList, sname, fullname, (void (*)(void)) function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*--------------------------------------------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "DMRegisterDestroy"
/*@C
   DMRegisterDestroy - Frees the list of DM methods that were registered by DMRegister()/DMRegisterDynamic().

   Not Collective

   Level: advanced

.keywords: DM, register, destroy
.seealso: DMRegister(), DMRegisterAll(), DMRegisterDynamic()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DMRegisterDestroy(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFListDestroy(&DMList);CHKERRQ(ierr);
  DMRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_MATLAB_ENGINE)
#include "mex.h"

typedef struct {char *funcname; char *jacname; mxArray *ctx;} DMMatlabContext;

#undef __FUNCT__  
#define __FUNCT__ "DMComputeFunction_Matlab"
/*
   DMComputeFunction_Matlab - Calls the function that has been set with
                         DMSetFunctionMatlab().  

   For linear problems x is null
   
.seealso: DMSetFunction(), DMGetFunction()
*/
PetscErrorCode PETSCDM_DLLEXPORT DMComputeFunction_Matlab(DM dm,Vec x,Vec y)
{
  PetscErrorCode    ierr;
  DMMatlabContext   *sctx;
  int               nlhs = 1,nrhs = 4;
  mxArray	    *plhs[1],*prhs[4];
  long long int     lx = 0,ly = 0,ls = 0;
      
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);
  PetscCheckSameComm(dm,1,y,3);

  /* call Matlab function in ctx with arguments x and y */
  ierr = DMGetContext(dm,(void**)&sctx);CHKERRQ(ierr);
  ierr = PetscMemcpy(&ls,&dm,sizeof(dm));CHKERRQ(ierr); 
  ierr = PetscMemcpy(&lx,&x,sizeof(x));CHKERRQ(ierr); 
  ierr = PetscMemcpy(&ly,&y,sizeof(y));CHKERRQ(ierr); 
  prhs[0] =  mxCreateDoubleScalar((double)ls);
  prhs[1] =  mxCreateDoubleScalar((double)lx);
  prhs[2] =  mxCreateDoubleScalar((double)ly);
  prhs[3] =  mxCreateString(sctx->funcname);
  ierr    =  mexCallMATLAB(nlhs,plhs,nrhs,prhs,"PetscDMComputeFunctionInternal");CHKERRQ(ierr);
  ierr    =  mxGetScalar(plhs[0]);CHKERRQ(ierr);
  mxDestroyArray(prhs[0]);
  mxDestroyArray(prhs[1]);
  mxDestroyArray(prhs[2]);
  mxDestroyArray(prhs[3]);
  mxDestroyArray(plhs[0]);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "DMSetFunctionMatlab"
/*
   DMSetFunctionMatlab - Sets the function evaluation routine 

*/
PetscErrorCode PETSCDM_DLLEXPORT DMSetFunctionMatlab(DM dm,const char *func)
{
  PetscErrorCode    ierr;
  DMMatlabContext   *sctx;

  PetscFunctionBegin;
  /* currently sctx is memory bleed */
  ierr = DMGetContext(dm,(void**)&sctx);CHKERRQ(ierr);
  if (!sctx) {
    ierr = PetscMalloc(sizeof(DMMatlabContext),&sctx);CHKERRQ(ierr);
  }
  ierr = PetscStrallocpy(func,&sctx->funcname);CHKERRQ(ierr);
  ierr = DMSetContext(dm,sctx);CHKERRQ(ierr);
  ierr = DMSetFunction(dm,DMComputeFunction_Matlab);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMComputeJacobian_Matlab"
/*
   DMComputeJacobian_Matlab - Calls the function that has been set with
                         DMSetJacobianMatlab().  

   For linear problems x is null
   
.seealso: DMSetFunction(), DMGetFunction()
*/
PetscErrorCode PETSCDM_DLLEXPORT DMComputeJacobian_Matlab(DM dm,Vec x,Mat A,Mat B,MatStructure *str)
{
  PetscErrorCode    ierr;
  DMMatlabContext   *sctx;
  int               nlhs = 2,nrhs = 5;
  mxArray	    *plhs[2],*prhs[5];
  long long int     lx = 0,lA = 0,lB = 0,ls = 0;
      
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(A,MAT_CLASSID,3);

  /* call Matlab function in ctx with arguments x, A, and B */
  ierr = DMGetContext(dm,(void**)&sctx);CHKERRQ(ierr);
  ierr = PetscMemcpy(&ls,&dm,sizeof(dm));CHKERRQ(ierr); 
  ierr = PetscMemcpy(&lx,&x,sizeof(x));CHKERRQ(ierr); 
  ierr = PetscMemcpy(&lA,&A,sizeof(A));CHKERRQ(ierr); 
  ierr = PetscMemcpy(&lB,&B,sizeof(B));CHKERRQ(ierr); 
  prhs[0] =  mxCreateDoubleScalar((double)ls);
  prhs[1] =  mxCreateDoubleScalar((double)lx);
  prhs[2] =  mxCreateDoubleScalar((double)lA);
  prhs[3] =  mxCreateDoubleScalar((double)lB);
  prhs[4] =  mxCreateString(sctx->jacname);
  ierr    =  mexCallMATLAB(nlhs,plhs,nrhs,prhs,"PetscDMComputeJacobianInternal");CHKERRQ(ierr);
  *str    =  mxGetScalar(plhs[0]);CHKERRQ(ierr);
  ierr    =  mxGetScalar(plhs[1]);CHKERRQ(ierr);
  mxDestroyArray(prhs[0]);
  mxDestroyArray(prhs[1]);
  mxDestroyArray(prhs[2]);
  mxDestroyArray(prhs[3]);
  mxDestroyArray(prhs[4]);
  mxDestroyArray(plhs[0]);
  mxDestroyArray(plhs[1]);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "DMSetJacobianMatlab"
/*
   DMSetJacobianMatlab - Sets the Jacobian function evaluation routine 

*/
PetscErrorCode PETSCDM_DLLEXPORT DMSetJacobianMatlab(DM dm,const char *func)
{
  PetscErrorCode    ierr;
  DMMatlabContext   *sctx;

  PetscFunctionBegin;
  /* currently sctx is memory bleed */
  ierr = DMGetContext(dm,(void**)&sctx);CHKERRQ(ierr);
  if (!sctx) {
    ierr = PetscMalloc(sizeof(DMMatlabContext),&sctx);CHKERRQ(ierr);
  }
  ierr = PetscStrallocpy(func,&sctx->jacname);CHKERRQ(ierr);
  ierr = DMSetContext(dm,sctx);CHKERRQ(ierr);
  ierr = DMSetJacobian(dm,DMComputeJacobian_Matlab);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif
