#include <petsc/private/dmimpl.h>
#include <petscdmredundant.h>   /*I      "petscdmredundant.h" I*/

typedef struct  {
  PetscMPIInt rank;                /* owner */
  PetscInt    N;                   /* total number of dofs */
  PetscInt    n;                   /* owned number of dofs, n=N on owner, n=0 on non-owners */
} DM_Redundant;

static PetscErrorCode DMCreateMatrix_Redundant(DM dm,Mat *J)
{
  DM_Redundant           *red = (DM_Redundant*)dm->data;
  ISLocalToGlobalMapping ltog;
  PetscInt               i,rstart,rend,*cols;
  PetscScalar            *vals;

  PetscFunctionBegin;
  CHKERRQ(MatCreate(PetscObjectComm((PetscObject)dm),J));
  CHKERRQ(MatSetSizes(*J,red->n,red->n,red->N,red->N));
  CHKERRQ(MatSetType(*J,dm->mattype));
  CHKERRQ(MatSeqAIJSetPreallocation(*J,red->n,NULL));
  CHKERRQ(MatSeqBAIJSetPreallocation(*J,1,red->n,NULL));
  CHKERRQ(MatMPIAIJSetPreallocation(*J,red->n,NULL,red->N-red->n,NULL));
  CHKERRQ(MatMPIBAIJSetPreallocation(*J,1,red->n,NULL,red->N-red->n,NULL));

  CHKERRQ(DMGetLocalToGlobalMapping(dm,&ltog));
  CHKERRQ(MatSetLocalToGlobalMapping(*J,ltog,ltog));
  CHKERRQ(MatSetDM(*J,dm));

  CHKERRQ(PetscMalloc2(red->N,&cols,red->N,&vals));
  for (i=0; i<red->N; i++) {
    cols[i] = i;
    vals[i] = 0.0;
  }
  CHKERRQ(MatGetOwnershipRange(*J,&rstart,&rend));
  for (i=rstart; i<rend; i++) {
    CHKERRQ(MatSetValues(*J,1,&i,red->N,cols,vals,INSERT_VALUES));
  }
  CHKERRQ(PetscFree2(cols,vals));
  CHKERRQ(MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMDestroy_Redundant(DM dm)
{
  PetscFunctionBegin;
  CHKERRQ(PetscObjectComposeFunction((PetscObject)dm,"DMRedundantSetSize_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)dm,"DMRedundantGetSize_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)dm,"DMSetUpGLVisViewer_C",NULL));
  /* This was originally freed in DMDestroy(), but that prevents reference counting of backend objects */
  CHKERRQ(PetscFree(dm->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCreateGlobalVector_Redundant(DM dm,Vec *gvec)
{
  DM_Redundant           *red = (DM_Redundant*)dm->data;
  ISLocalToGlobalMapping ltog;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(gvec,2);
  *gvec = NULL;
  CHKERRQ(VecCreate(PetscObjectComm((PetscObject)dm),gvec));
  CHKERRQ(VecSetSizes(*gvec,red->n,red->N));
  CHKERRQ(VecSetType(*gvec,dm->vectype));
  CHKERRQ(DMGetLocalToGlobalMapping(dm,&ltog));
  CHKERRQ(VecSetLocalToGlobalMapping(*gvec,ltog));
  CHKERRQ(VecSetDM(*gvec,dm));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCreateLocalVector_Redundant(DM dm,Vec *lvec)
{
  DM_Redundant   *red = (DM_Redundant*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(lvec,2);
  *lvec = NULL;
  CHKERRQ(VecCreate(PETSC_COMM_SELF,lvec));
  CHKERRQ(VecSetSizes(*lvec,red->N,red->N));
  CHKERRQ(VecSetType(*lvec,dm->vectype));
  CHKERRQ(VecSetDM(*lvec,dm));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMLocalToGlobalBegin_Redundant(DM dm,Vec l,InsertMode imode,Vec g)
{
  DM_Redundant      *red = (DM_Redundant*)dm->data;
  const PetscScalar *lv;
  PetscScalar       *gv;
  PetscMPIInt       rank;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank));
  CHKERRQ(VecGetArrayRead(l,&lv));
  CHKERRQ(VecGetArray(g,&gv));
  switch (imode) {
  case ADD_VALUES:
  case MAX_VALUES:
  {
    void        *source;
    PetscScalar *buffer;
    PetscInt    i;
    if (rank == red->rank) {
      buffer = gv;
      source = MPI_IN_PLACE;
      if (imode == ADD_VALUES) for (i=0; i<red->N; i++) buffer[i] = gv[i] + lv[i];
#if !defined(PETSC_USE_COMPLEX)
      if (imode == MAX_VALUES) for (i=0; i<red->N; i++) buffer[i] = PetscMax(gv[i],lv[i]);
#endif
    } else source = (void*)lv;
    CHKERRMPI(MPI_Reduce(source,gv,red->N,MPIU_SCALAR,(imode == ADD_VALUES) ? MPIU_SUM : MPIU_MAX,red->rank,PetscObjectComm((PetscObject)dm)));
  } break;
  case INSERT_VALUES:
    CHKERRQ(PetscArraycpy(gv,lv,red->n));
    break;
  default: SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"InsertMode not supported");
  }
  CHKERRQ(VecRestoreArrayRead(l,&lv));
  CHKERRQ(VecRestoreArray(g,&gv));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMLocalToGlobalEnd_Redundant(DM dm,Vec l,InsertMode imode,Vec g)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMGlobalToLocalBegin_Redundant(DM dm,Vec g,InsertMode imode,Vec l)
{
  DM_Redundant      *red = (DM_Redundant*)dm->data;
  const PetscScalar *gv;
  PetscScalar       *lv;

  PetscFunctionBegin;
  CHKERRQ(VecGetArrayRead(g,&gv));
  CHKERRQ(VecGetArray(l,&lv));
  switch (imode) {
  case INSERT_VALUES:
    if (red->n) CHKERRQ(PetscArraycpy(lv,gv,red->n));
    CHKERRMPI(MPI_Bcast(lv,red->N,MPIU_SCALAR,red->rank,PetscObjectComm((PetscObject)dm)));
    break;
  default: SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"InsertMode not supported");
  }
  CHKERRQ(VecRestoreArrayRead(g,&gv));
  CHKERRQ(VecRestoreArray(l,&lv));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMGlobalToLocalEnd_Redundant(DM dm,Vec g,InsertMode imode,Vec l)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMSetUp_Redundant(DM dm)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMView_Redundant(DM dm,PetscViewer viewer)
{
  DM_Redundant   *red = (DM_Redundant*)dm->data;
  PetscBool      iascii;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"redundant: rank=%D N=%D\n",red->rank,red->N));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCreateColoring_Redundant(DM dm,ISColoringType ctype,ISColoring *coloring)
{
  DM_Redundant    *red = (DM_Redundant*)dm->data;
  PetscInt        i,nloc;
  ISColoringValue *colors;

  PetscFunctionBegin;
  switch (ctype) {
  case IS_COLORING_GLOBAL:
    nloc = red->n;
    break;
  case IS_COLORING_LOCAL:
    nloc = red->N;
    break;
  default: SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONG,"Unknown ISColoringType %d",(int)ctype);
  }
  CHKERRQ(PetscMalloc1(nloc,&colors));
  for (i=0; i<nloc; i++) colors[i] = i;
  CHKERRQ(ISColoringCreate(PetscObjectComm((PetscObject)dm),red->N,nloc,colors,PETSC_OWN_POINTER,coloring));
  CHKERRQ(ISColoringSetType(*coloring,ctype));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMRefine_Redundant(DM dmc,MPI_Comm comm,DM *dmf)
{
  PetscMPIInt    flag;
  DM_Redundant   *redc = (DM_Redundant*)dmc->data;

  PetscFunctionBegin;
  if (comm == MPI_COMM_NULL) {
    CHKERRQ(PetscObjectGetComm((PetscObject)dmc,&comm));
  }
  CHKERRMPI(MPI_Comm_compare(PetscObjectComm((PetscObject)dmc),comm,&flag));
  PetscCheckFalse(flag != MPI_CONGRUENT && flag != MPI_IDENT,PetscObjectComm((PetscObject)dmc),PETSC_ERR_SUP,"cannot change communicators");
  CHKERRQ(DMRedundantCreate(comm,redc->rank,redc->N,dmf));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCoarsen_Redundant(DM dmf,MPI_Comm comm,DM *dmc)
{
  PetscMPIInt    flag;
  DM_Redundant   *redf = (DM_Redundant*)dmf->data;

  PetscFunctionBegin;
  if (comm == MPI_COMM_NULL) {
    CHKERRQ(PetscObjectGetComm((PetscObject)dmf,&comm));
  }
  CHKERRMPI(MPI_Comm_compare(PetscObjectComm((PetscObject)dmf),comm,&flag));
  PetscCheckFalse(flag != MPI_CONGRUENT && flag != MPI_IDENT,PetscObjectComm((PetscObject)dmf),PETSC_ERR_SUP,"cannot change communicators");
  CHKERRQ(DMRedundantCreate(comm,redf->rank,redf->N,dmc));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCreateInterpolation_Redundant(DM dmc,DM dmf,Mat *P,Vec *scale)
{
  DM_Redundant   *redc = (DM_Redundant*)dmc->data;
  DM_Redundant   *redf = (DM_Redundant*)dmf->data;
  PetscMPIInt    flag;
  PetscInt       i,rstart,rend;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_compare(PetscObjectComm((PetscObject)dmc),PetscObjectComm((PetscObject)dmf),&flag));
  PetscCheckFalse(flag != MPI_CONGRUENT && flag != MPI_IDENT,PetscObjectComm((PetscObject)dmf),PETSC_ERR_SUP,"cannot change communicators");
  PetscCheckFalse(redc->rank != redf->rank,PetscObjectComm((PetscObject)dmf),PETSC_ERR_ARG_INCOMP,"Owning rank does not match");
  PetscCheckFalse(redc->N != redf->N,PetscObjectComm((PetscObject)dmf),PETSC_ERR_ARG_INCOMP,"Global size does not match");
  CHKERRQ(MatCreate(PetscObjectComm((PetscObject)dmc),P));
  CHKERRQ(MatSetSizes(*P,redc->n,redc->n,redc->N,redc->N));
  CHKERRQ(MatSetType(*P,MATAIJ));
  CHKERRQ(MatSeqAIJSetPreallocation(*P,1,NULL));
  CHKERRQ(MatMPIAIJSetPreallocation(*P,1,NULL,0,NULL));
  CHKERRQ(MatGetOwnershipRange(*P,&rstart,&rend));
  for (i=rstart; i<rend; i++) CHKERRQ(MatSetValue(*P,i,i,1.0,INSERT_VALUES));
  CHKERRQ(MatAssemblyBegin(*P,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(*P,MAT_FINAL_ASSEMBLY));
  if (scale) CHKERRQ(DMCreateInterpolationScale(dmc,dmf,*P,scale));
  PetscFunctionReturn(0);
}

/*@
    DMRedundantSetSize - Sets the size of a densely coupled redundant object

    Collective on dm

    Input Parameters:
+   dm - redundant DM
.   rank - rank of process to own redundant degrees of freedom
-   N - total number of redundant degrees of freedom

    Level: advanced

.seealso DMDestroy(), DMCreateGlobalVector(), DMRedundantCreate(), DMRedundantGetSize()
@*/
PetscErrorCode DMRedundantSetSize(DM dm,PetscMPIInt rank,PetscInt N)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidType(dm,1);
  PetscValidLogicalCollectiveMPIInt(dm,rank,2);
  PetscValidLogicalCollectiveInt(dm,N,3);
  CHKERRQ(PetscTryMethod(dm,"DMRedundantSetSize_C",(DM,PetscMPIInt,PetscInt),(dm,rank,N)));
  PetscFunctionReturn(0);
}

/*@
    DMRedundantGetSize - Gets the size of a densely coupled redundant object

    Not Collective

    Input Parameter:
.   dm - redundant DM

    Output Parameters:
+   rank - rank of process to own redundant degrees of freedom (or NULL)
-   N - total number of redundant degrees of freedom (or NULL)

    Level: advanced

.seealso DMDestroy(), DMCreateGlobalVector(), DMRedundantCreate(), DMRedundantSetSize()
@*/
PetscErrorCode DMRedundantGetSize(DM dm,PetscMPIInt *rank,PetscInt *N)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidType(dm,1);
  CHKERRQ(PetscUseMethod(dm,"DMRedundantGetSize_C",(DM,PetscMPIInt*,PetscInt*),(dm,rank,N)));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMRedundantSetSize_Redundant(DM dm,PetscMPIInt rank,PetscInt N)
{
  DM_Redundant   *red = (DM_Redundant*)dm->data;
  PetscMPIInt    myrank;
  PetscInt       i,*globals;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&myrank));
  red->rank = rank;
  red->N    = N;
  red->n    = (myrank == rank) ? N : 0;

  /* mapping is setup here */
  CHKERRQ(PetscMalloc1(red->N,&globals));
  for (i=0; i<red->N; i++) globals[i] = i;
  CHKERRQ(ISLocalToGlobalMappingDestroy(&dm->ltogmap));
  CHKERRQ(ISLocalToGlobalMappingCreate(PetscObjectComm((PetscObject)dm),1,red->N,globals,PETSC_OWN_POINTER,&dm->ltogmap));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMRedundantGetSize_Redundant(DM dm,PetscInt *rank,PetscInt *N)
{
  DM_Redundant *red = (DM_Redundant*)dm->data;

  PetscFunctionBegin;
  if (rank) *rank = red->rank;
  if (N)    *N = red->N;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMSetUpGLVisViewer_Redundant(PetscObject odm, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/*MC
   DMREDUNDANT = "redundant" - A DM object that is used to manage data for a small set of dense globally coupled variables.
         In the global representation of the vector the variables are all stored on a single MPI process (all the other MPI processes have
         no variables) in the local representation all the variables are stored on ALL the MPI processes (because they are all needed for each
         processes local computations).

         This DM is generally used inside a DMCOMPOSITE object. For example, it may be used to store continuation parameters for a bifurcation problem.

  Level: intermediate

.seealso: DMType, DMCOMPOSITE,  DMCreate(), DMRedundantSetSize(), DMRedundantGetSize()
M*/

PETSC_EXTERN PetscErrorCode DMCreate_Redundant(DM dm)
{
  DM_Redundant   *red;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(dm,&red));
  dm->data = red;

  dm->ops->setup               = DMSetUp_Redundant;
  dm->ops->view                = DMView_Redundant;
  dm->ops->createglobalvector  = DMCreateGlobalVector_Redundant;
  dm->ops->createlocalvector   = DMCreateLocalVector_Redundant;
  dm->ops->creatematrix        = DMCreateMatrix_Redundant;
  dm->ops->destroy             = DMDestroy_Redundant;
  dm->ops->globaltolocalbegin  = DMGlobalToLocalBegin_Redundant;
  dm->ops->globaltolocalend    = DMGlobalToLocalEnd_Redundant;
  dm->ops->localtoglobalbegin  = DMLocalToGlobalBegin_Redundant;
  dm->ops->localtoglobalend    = DMLocalToGlobalEnd_Redundant;
  dm->ops->refine              = DMRefine_Redundant;
  dm->ops->coarsen             = DMCoarsen_Redundant;
  dm->ops->createinterpolation = DMCreateInterpolation_Redundant;
  dm->ops->getcoloring         = DMCreateColoring_Redundant;

  CHKERRQ(PetscObjectComposeFunction((PetscObject)dm,"DMRedundantSetSize_C",DMRedundantSetSize_Redundant));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)dm,"DMRedundantGetSize_C",DMRedundantGetSize_Redundant));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)dm,"DMSetUpGLVisViewer_C",DMSetUpGLVisViewer_Redundant));
  PetscFunctionReturn(0);
}

/*@C
    DMRedundantCreate - Creates a DM object, used to manage data for dense globally coupled variables

    Collective

    Input Parameters:
+   comm - the processors that will share the global vector
.   rank - rank to own the redundant values
-   N - total number of degrees of freedom

    Output Parameters:
.   dm - the redundant DM

    Level: advanced

.seealso DMDestroy(), DMCreateGlobalVector(), DMCreateMatrix(), DMCompositeAddDM(), DMREDUNDANT, DMSetType(), DMRedundantSetSize(), DMRedundantGetSize()

@*/
PetscErrorCode DMRedundantCreate(MPI_Comm comm,PetscMPIInt rank,PetscInt N,DM *dm)
{
  PetscFunctionBegin;
  PetscValidPointer(dm,4);
  CHKERRQ(DMCreate(comm,dm));
  CHKERRQ(DMSetType(*dm,DMREDUNDANT));
  CHKERRQ(DMRedundantSetSize(*dm,rank,N));
  CHKERRQ(DMSetUp(*dm));
  PetscFunctionReturn(0);
}
