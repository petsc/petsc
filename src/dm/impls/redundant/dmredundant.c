#include <petsc-private/dmimpl.h>     /*I      "petscdm.h"          I*/
#include <petscdmredundant.h>   /*I      "petscdmredundant.h" I*/
#include <petscmat.h>           /*I      "petscmat.h"         I*/

typedef struct  {
  PetscInt rank;                /* owner */
  PetscInt N;                   /* total number of dofs */
  PetscInt n;                   /* owned number of dofs, n=N on owner, n=0 on non-owners */
} DM_Redundant;

#undef __FUNCT__
#define __FUNCT__ "DMCreateMatrix_Redundant"
static PetscErrorCode DMCreateMatrix_Redundant(DM dm,const MatType mtype,Mat *J)
{
  DM_Redundant           *red = (DM_Redundant*)dm->data;
  PetscErrorCode         ierr;
  ISLocalToGlobalMapping ltog,ltogb;
  PetscInt               i,rstart,rend,*cols;
  PetscScalar            *vals;

  PetscFunctionBegin;
  ierr = MatCreate(((PetscObject)dm)->comm,J);CHKERRQ(ierr);
  ierr = MatSetSizes(*J,red->n,red->n,red->N,red->N);CHKERRQ(ierr);
  ierr = MatSetType(*J,mtype);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(*J,red->n,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatSeqBAIJSetPreallocation(*J,1,red->n,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(*J,red->n,PETSC_NULL,red->N-red->n,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatMPIBAIJSetPreallocation(*J,1,red->n,PETSC_NULL,red->N-red->n,PETSC_NULL);CHKERRQ(ierr);

  ierr = DMGetLocalToGlobalMapping(dm,&ltog);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMappingBlock(dm,&ltogb);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(*J,ltog,ltog);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMappingBlock(*J,ltogb,ltogb);CHKERRQ(ierr);

  ierr = PetscMalloc2(red->N,PetscInt,&cols,red->N,PetscScalar,&vals);CHKERRQ(ierr);
  for (i=0; i<red->N; i++) {
    cols[i] = i;
    vals[i] = 0.0;
  }
  ierr = MatGetOwnershipRange(*J,&rstart,&rend);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) {
    ierr = MatSetValues(*J,1,&i,red->N,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = PetscFree2(cols,vals);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDestroy_Redundant"
static PetscErrorCode DMDestroy_Redundant(DM dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)dm,"DMRedundantSetSize_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)dm,"DMRedundantGetSize_C","",PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCreateGlobalVector_Redundant"
static PetscErrorCode DMCreateGlobalVector_Redundant(DM dm,Vec *gvec)
{
  PetscErrorCode         ierr;
  DM_Redundant           *red = (DM_Redundant*)dm->data;
  ISLocalToGlobalMapping ltog,ltogb;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(gvec,2);
  *gvec = 0;
  ierr = VecCreate(((PetscObject)dm)->comm,gvec);CHKERRQ(ierr);
  ierr = VecSetSizes(*gvec,red->n,red->N);CHKERRQ(ierr);
  ierr = VecSetType(*gvec,dm->vectype);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMapping(dm,&ltog);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMappingBlock(dm,&ltogb);CHKERRQ(ierr);
  ierr = VecSetLocalToGlobalMapping(*gvec,ltog);CHKERRQ(ierr);
  ierr = VecSetLocalToGlobalMappingBlock(*gvec,ltog);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)*gvec,"DM",(PetscObject)dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCreateLocalVector_Redundant"
static PetscErrorCode DMCreateLocalVector_Redundant(DM dm,Vec *lvec)
{
  PetscErrorCode         ierr;
  DM_Redundant           *red = (DM_Redundant*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(lvec,2);
  *lvec = 0;
  ierr = VecCreate(PETSC_COMM_SELF,lvec);CHKERRQ(ierr);
  ierr = VecSetSizes(*lvec,red->N,red->N);CHKERRQ(ierr);
  ierr = VecSetType(*lvec,dm->vectype);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)*lvec,"DM",(PetscObject)dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMLocalToGlobalBegin_Redundant"
static PetscErrorCode DMLocalToGlobalBegin_Redundant(DM dm,Vec l,InsertMode imode,Vec g)
{
  PetscErrorCode    ierr;
  DM_Redundant      *red = (DM_Redundant*)dm->data;
  const PetscScalar *lv;
  PetscScalar       *gv;
  PetscMPIInt       rank;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(((PetscObject)dm)->comm,&rank);CHKERRQ(ierr);
  ierr = VecGetArrayRead(l,&lv);CHKERRQ(ierr);
  ierr = VecGetArray(g,&gv);CHKERRQ(ierr);
  switch (imode) {
  case ADD_VALUES:
  case MAX_VALUES:
  {
    void *source;
    PetscScalar *buffer;
    PetscInt i;
    if (rank == red->rank) {
#if defined (PETSC_HAVE_MPI_IN_PLACE)
      buffer = gv;
      source = MPI_IN_PLACE;
#else
      ierr = PetscMalloc(red->N*sizeof(PetscScalar),&buffer);CHKERRQ(ierr);
      source = buffer;
#endif
      if (imode == ADD_VALUES) for (i=0; i<red->N; i++) buffer[i] = gv[i] + lv[i];
#if !defined(PETSC_USE_COMPLEX)
      if (imode == MAX_VALUES) for (i=0; i<red->N; i++) buffer[i] = PetscMax(gv[i],lv[i]);
#endif
    } else {
      source = (void*)lv;
    }
    ierr = MPI_Reduce(source,gv,red->N,MPIU_SCALAR,(imode == ADD_VALUES)?MPI_SUM:MPI_MAX,red->rank,((PetscObject)dm)->comm);CHKERRQ(ierr);
#if !defined(PETSC_HAVE_MPI_IN_PLACE)
    if (rank == red->rank) {ierr = PetscFree(buffer);CHKERRQ(ierr);}
#endif
  } break;
  case INSERT_VALUES:
  ierr = PetscMemcpy(gv,lv,red->n*sizeof(PetscScalar));CHKERRQ(ierr);
  break;
  default: SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_SUP,"InsertMode not supported");
  }
  ierr = VecRestoreArrayRead(l,&lv);CHKERRQ(ierr);
  ierr = VecRestoreArray(g,&gv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMLocalToGlobalEnd_Redundant"
static PetscErrorCode DMLocalToGlobalEnd_Redundant(DM dm,Vec l,InsertMode imode,Vec g)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMGlobalToLocalBegin_Redundant"
static PetscErrorCode DMGlobalToLocalBegin_Redundant(DM dm,Vec g,InsertMode imode,Vec l)
{
  PetscErrorCode    ierr;
  DM_Redundant      *red = (DM_Redundant*)dm->data;
  const PetscScalar *gv;
  PetscScalar       *lv;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(g,&gv);CHKERRQ(ierr);
  ierr = VecGetArray(l,&lv);CHKERRQ(ierr);
  switch (imode) {
  case INSERT_VALUES:
    if (red->n) {ierr = PetscMemcpy(lv,gv,red->n*sizeof(PetscScalar));CHKERRQ(ierr);}
    ierr = MPI_Bcast(lv,red->N,MPIU_SCALAR,red->rank,((PetscObject)dm)->comm);CHKERRQ(ierr);
    break;
  default: SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_SUP,"InsertMode not supported");
  }
  ierr = VecRestoreArrayRead(g,&gv);CHKERRQ(ierr);
  ierr = VecRestoreArray(l,&lv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMGlobalToLocalEnd_Redundant"
static PetscErrorCode DMGlobalToLocalEnd_Redundant(DM dm,Vec g,InsertMode imode,Vec l)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSetUp_Redundant"
static PetscErrorCode DMSetUp_Redundant(DM dm)
{
  PetscErrorCode ierr;
  DM_Redundant   *red = (DM_Redundant*)dm->data;
  PetscInt       i,*globals;

  PetscFunctionBegin;
  ierr = PetscMalloc(red->N*sizeof(PetscInt),&globals);CHKERRQ(ierr);
  for (i=0; i<red->N; i++) globals[i] = i;
  ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_SELF,red->N,globals,PETSC_OWN_POINTER,&dm->ltogmap);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)dm->ltogmap);CHKERRQ(ierr);
  dm->ltogmapb = dm->ltogmap;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMView_Redundant"
static PetscErrorCode DMView_Redundant(DM dm,PetscViewer viewer)
{
  PetscErrorCode ierr;
  DM_Redundant   *red = (DM_Redundant*)dm->data;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"redundant: rank=%D N=%D\n",red->rank,red->N);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCreateColoring_Redundant"
static PetscErrorCode DMCreateColoring_Redundant(DM dm,ISColoringType ctype,const MatType mtype,ISColoring *coloring)
{
  DM_Redundant   *red = (DM_Redundant*)dm->data;
  PetscErrorCode ierr;
  PetscInt i,nloc;
  ISColoringValue *colors;

  PetscFunctionBegin;
  switch (ctype) {
  case IS_COLORING_GLOBAL:
    nloc = red->n;
    break;
  case IS_COLORING_GHOSTED:
    nloc = red->N;
    break;
  default: SETERRQ1(((PetscObject)dm)->comm,PETSC_ERR_ARG_WRONG,"Unknown ISColoringType %d",(int)ctype);
  }
  ierr = PetscMalloc(nloc*sizeof(ISColoringValue),&colors);CHKERRQ(ierr);
  for (i=0; i<nloc; i++) colors[i] = i;
  ierr = ISColoringCreate(((PetscObject)dm)->comm,red->N,nloc,colors,coloring);CHKERRQ(ierr);
  ierr = ISColoringSetType(*coloring,ctype);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMRefine_Redundant"
static PetscErrorCode DMRefine_Redundant(DM dmc,MPI_Comm comm,DM *dmf)
{
  PetscErrorCode ierr;
  PetscMPIInt flag;
  DM_Redundant *redc = (DM_Redundant*)dmc->data;

  PetscFunctionBegin;
  if (comm == MPI_COMM_NULL) comm = ((PetscObject)dmc)->comm;
  ierr = MPI_Comm_compare(((PetscObject)dmc)->comm,comm,&flag); CHKERRQ(ierr);
  if (flag != MPI_CONGRUENT && flag != MPI_IDENT) SETERRQ(((PetscObject)dmc)->comm,PETSC_ERR_SUP,"cannot change communicators");
  ierr = DMRedundantCreate(comm,redc->rank,redc->N,dmf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCoarsen_Redundant"
static PetscErrorCode DMCoarsen_Redundant(DM dmf,MPI_Comm comm,DM *dmc)
{
  PetscErrorCode ierr;
  PetscMPIInt flag;
  DM_Redundant *redf = (DM_Redundant*)dmf->data;

  PetscFunctionBegin;
  if (comm == MPI_COMM_NULL) comm = ((PetscObject)dmf)->comm;
  ierr = MPI_Comm_compare(((PetscObject)dmf)->comm,comm,&flag); CHKERRQ(ierr);
  if (flag != MPI_CONGRUENT && flag != MPI_IDENT) SETERRQ(((PetscObject)dmf)->comm,PETSC_ERR_SUP,"cannot change communicators");
  ierr = DMRedundantCreate(comm,redf->rank,redf->N,dmc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCreateInterpolation_Redundant"
static PetscErrorCode DMCreateInterpolation_Redundant(DM dmc,DM dmf,Mat *P,Vec *scale)
{
  PetscErrorCode ierr;
  DM_Redundant   *redc = (DM_Redundant*)dmc->data;
  DM_Redundant   *redf = (DM_Redundant*)dmf->data;
  PetscMPIInt    flag;
  PetscInt       i,rstart,rend;

  PetscFunctionBegin;
  ierr = MPI_Comm_compare(((PetscObject)dmc)->comm,((PetscObject)dmf)->comm,&flag); CHKERRQ(ierr);
  if (flag != MPI_CONGRUENT && flag != MPI_IDENT) SETERRQ(((PetscObject)dmf)->comm,PETSC_ERR_SUP,"cannot change communicators");
  if (redc->rank != redf->rank) SETERRQ(((PetscObject)dmf)->comm,PETSC_ERR_ARG_INCOMP,"Owning rank does not match");
  if (redc->N != redf->N) SETERRQ(((PetscObject)dmf)->comm,PETSC_ERR_ARG_INCOMP,"Global size does not match");
  ierr = MatCreate(((PetscObject)dmc)->comm,P);CHKERRQ(ierr);
  ierr = MatSetSizes(*P,redc->n,redc->n,redc->N,redc->N);CHKERRQ(ierr);
  ierr = MatSetType(*P,MATAIJ);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(*P,1,0);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(*P,1,0,0,0);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(*P,&rstart,&rend);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) {ierr = MatSetValue(*P,i,i,1.0,INSERT_VALUES);CHKERRQ(ierr);}
  ierr = MatAssemblyBegin(*P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (scale) {ierr = DMCreateInterpolationScale(dmc,dmf,*P,scale);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMRedundantSetSize"
/*@
    DMRedundantSetSize - Sets the size of a densely coupled redundant object

    Collective on DM

    Input Parameter:
+   dm - redundant DM
.   rank - rank of process to own redundant degrees of freedom
-   N - total number of redundant degrees of freedom

    Level: advanced

.seealso DMDestroy(), DMCreateGlobalVector(), DMRedundantCreate(), DMRedundantGetSize()
@*/
PetscErrorCode DMRedundantSetSize(DM dm,PetscInt rank,PetscInt N)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidType(dm,1);
  PetscValidLogicalCollectiveInt(dm,rank,2);
  PetscValidLogicalCollectiveInt(dm,N,3);
  ierr = PetscTryMethod(dm,"DMRedundantSetSize_C",(DM,PetscInt,PetscInt),(dm,rank,N));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMRedundantGetSize"
/*@
    DMRedundantGetSize - Gets the size of a densely coupled redundant object

    Not Collective

    Input Parameter:
+   dm - redundant DM

    Output Parameters:
+   rank - rank of process to own redundant degrees of freedom (or PETSC_NULL)
-   N - total number of redundant degrees of freedom (or PETSC_NULL)

    Level: advanced

.seealso DMDestroy(), DMCreateGlobalVector(), DMRedundantCreate(), DMRedundantSetSize()
@*/
PetscErrorCode DMRedundantGetSize(DM dm,PetscInt *rank,PetscInt *N)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidType(dm,1);
  ierr = PetscUseMethod(dm,"DMRedundantGetSize_C",(DM,PetscInt*,PetscInt*),(dm,rank,N));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "DMRedundantSetSize_Redundant"
PetscErrorCode DMRedundantSetSize_Redundant(DM dm,PetscInt rank,PetscInt N)
{
  DM_Redundant   *red = (DM_Redundant*)dm->data;
  PetscErrorCode ierr;
  PetscMPIInt    myrank;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(((PetscObject)dm)->comm,&myrank);CHKERRQ(ierr);
  red->rank = rank;
  red->N = N;
  red->n = (myrank == rank) ? N : 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMRedundantGetSize_Redundant"
PetscErrorCode DMRedundantGetSize_Redundant(DM dm,PetscInt *rank,PetscInt *N)
{
  DM_Redundant   *red = (DM_Redundant*)dm->data;

  PetscFunctionBegin;
  if (rank) *rank = red->rank;
  if (N)    *N = red->N;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "DMCreate_Redundant"
PetscErrorCode DMCreate_Redundant(DM dm)
{
  PetscErrorCode ierr;
  DM_Redundant   *red;

  PetscFunctionBegin;
  ierr = PetscNewLog(dm,DM_Redundant,&red);CHKERRQ(ierr);
  dm->data = red;

  ierr = PetscObjectChangeTypeName((PetscObject)dm,DMREDUNDANT);CHKERRQ(ierr);
  dm->ops->setup              = DMSetUp_Redundant;
  dm->ops->view               = DMView_Redundant;
  dm->ops->createglobalvector = DMCreateGlobalVector_Redundant;
  dm->ops->createlocalvector  = DMCreateLocalVector_Redundant;
  dm->ops->creatematrix          = DMCreateMatrix_Redundant;
  dm->ops->destroy            = DMDestroy_Redundant;
  dm->ops->globaltolocalbegin = DMGlobalToLocalBegin_Redundant;
  dm->ops->globaltolocalend   = DMGlobalToLocalEnd_Redundant;
  dm->ops->localtoglobalbegin = DMLocalToGlobalBegin_Redundant;
  dm->ops->localtoglobalend   = DMLocalToGlobalEnd_Redundant;
  dm->ops->refine             = DMRefine_Redundant;
  dm->ops->coarsen            = DMCoarsen_Redundant;
  dm->ops->createinterpolation   = DMCreateInterpolation_Redundant;
  dm->ops->getcoloring        = DMCreateColoring_Redundant;
  ierr = PetscStrallocpy(VECSTANDARD,&dm->vectype);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)dm,"DMRedundantSetSize_C","DMRedundantSetSize_Redundant",DMRedundantSetSize_Redundant);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)dm,"DMRedundantGetSize_C","DMRedundantGetSize_Redundant",DMRedundantGetSize_Redundant);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "DMRedundantCreate"
/*@C
    DMRedundantCreate - Creates a DM object, used to manage data for dense globally coupled variables

    Collective on MPI_Comm

    Input Parameter:
+   comm - the processors that will share the global vector
.   rank - rank to own the redundant values
-   N - total number of degrees of freedom

    Output Parameters:
.   red - the redundant DM

    Level: advanced

.seealso DMDestroy(), DMCreateGlobalVector(), DMCreateMatrix(), DMCompositeAddDM()

@*/
PetscErrorCode DMRedundantCreate(MPI_Comm comm,PetscInt rank,PetscInt N,DM *dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(dm,2);
  ierr = DMCreate(comm,dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm,DMREDUNDANT);CHKERRQ(ierr);
  ierr = DMRedundantSetSize(*dm,rank,N);CHKERRQ(ierr);
  ierr = DMSetUp(*dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
