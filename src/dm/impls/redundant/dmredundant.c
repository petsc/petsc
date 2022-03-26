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
  PetscCall(MatCreate(PetscObjectComm((PetscObject)dm),J));
  PetscCall(MatSetSizes(*J,red->n,red->n,red->N,red->N));
  PetscCall(MatSetType(*J,dm->mattype));
  PetscCall(MatSeqAIJSetPreallocation(*J,red->n,NULL));
  PetscCall(MatSeqBAIJSetPreallocation(*J,1,red->n,NULL));
  PetscCall(MatMPIAIJSetPreallocation(*J,red->n,NULL,red->N-red->n,NULL));
  PetscCall(MatMPIBAIJSetPreallocation(*J,1,red->n,NULL,red->N-red->n,NULL));

  PetscCall(DMGetLocalToGlobalMapping(dm,&ltog));
  PetscCall(MatSetLocalToGlobalMapping(*J,ltog,ltog));
  PetscCall(MatSetDM(*J,dm));

  PetscCall(PetscMalloc2(red->N,&cols,red->N,&vals));
  for (i=0; i<red->N; i++) {
    cols[i] = i;
    vals[i] = 0.0;
  }
  PetscCall(MatGetOwnershipRange(*J,&rstart,&rend));
  for (i=rstart; i<rend; i++) {
    PetscCall(MatSetValues(*J,1,&i,red->N,cols,vals,INSERT_VALUES));
  }
  PetscCall(PetscFree2(cols,vals));
  PetscCall(MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMDestroy_Redundant(DM dm)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectComposeFunction((PetscObject)dm,"DMRedundantSetSize_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)dm,"DMRedundantGetSize_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)dm,"DMSetUpGLVisViewer_C",NULL));
  /* This was originally freed in DMDestroy(), but that prevents reference counting of backend objects */
  PetscCall(PetscFree(dm->data));
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
  PetscCall(VecCreate(PetscObjectComm((PetscObject)dm),gvec));
  PetscCall(VecSetSizes(*gvec,red->n,red->N));
  PetscCall(VecSetType(*gvec,dm->vectype));
  PetscCall(DMGetLocalToGlobalMapping(dm,&ltog));
  PetscCall(VecSetLocalToGlobalMapping(*gvec,ltog));
  PetscCall(VecSetDM(*gvec,dm));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCreateLocalVector_Redundant(DM dm,Vec *lvec)
{
  DM_Redundant   *red = (DM_Redundant*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(lvec,2);
  *lvec = NULL;
  PetscCall(VecCreate(PETSC_COMM_SELF,lvec));
  PetscCall(VecSetSizes(*lvec,red->N,red->N));
  PetscCall(VecSetType(*lvec,dm->vectype));
  PetscCall(VecSetDM(*lvec,dm));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMLocalToGlobalBegin_Redundant(DM dm,Vec l,InsertMode imode,Vec g)
{
  DM_Redundant      *red = (DM_Redundant*)dm->data;
  const PetscScalar *lv;
  PetscScalar       *gv;
  PetscMPIInt       rank;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank));
  PetscCall(VecGetArrayRead(l,&lv));
  PetscCall(VecGetArray(g,&gv));
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
    PetscCallMPI(MPI_Reduce(source,gv,red->N,MPIU_SCALAR,(imode == ADD_VALUES) ? MPIU_SUM : MPIU_MAX,red->rank,PetscObjectComm((PetscObject)dm)));
  } break;
  case INSERT_VALUES:
    PetscCall(PetscArraycpy(gv,lv,red->n));
    break;
  default: SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"InsertMode not supported");
  }
  PetscCall(VecRestoreArrayRead(l,&lv));
  PetscCall(VecRestoreArray(g,&gv));
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
  PetscCall(VecGetArrayRead(g,&gv));
  PetscCall(VecGetArray(l,&lv));
  switch (imode) {
  case INSERT_VALUES:
    if (red->n) PetscCall(PetscArraycpy(lv,gv,red->n));
    PetscCallMPI(MPI_Bcast(lv,red->N,MPIU_SCALAR,red->rank,PetscObjectComm((PetscObject)dm)));
    break;
  default: SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"InsertMode not supported");
  }
  PetscCall(VecRestoreArrayRead(g,&gv));
  PetscCall(VecRestoreArray(l,&lv));
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
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer,"redundant: rank=%D N=%D\n",red->rank,red->N));
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
  PetscCall(PetscMalloc1(nloc,&colors));
  for (i=0; i<nloc; i++) colors[i] = i;
  PetscCall(ISColoringCreate(PetscObjectComm((PetscObject)dm),red->N,nloc,colors,PETSC_OWN_POINTER,coloring));
  PetscCall(ISColoringSetType(*coloring,ctype));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMRefine_Redundant(DM dmc,MPI_Comm comm,DM *dmf)
{
  PetscMPIInt    flag;
  DM_Redundant   *redc = (DM_Redundant*)dmc->data;

  PetscFunctionBegin;
  if (comm == MPI_COMM_NULL) {
    PetscCall(PetscObjectGetComm((PetscObject)dmc,&comm));
  }
  PetscCallMPI(MPI_Comm_compare(PetscObjectComm((PetscObject)dmc),comm,&flag));
  PetscCheckFalse(flag != MPI_CONGRUENT && flag != MPI_IDENT,PetscObjectComm((PetscObject)dmc),PETSC_ERR_SUP,"cannot change communicators");
  PetscCall(DMRedundantCreate(comm,redc->rank,redc->N,dmf));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCoarsen_Redundant(DM dmf,MPI_Comm comm,DM *dmc)
{
  PetscMPIInt    flag;
  DM_Redundant   *redf = (DM_Redundant*)dmf->data;

  PetscFunctionBegin;
  if (comm == MPI_COMM_NULL) {
    PetscCall(PetscObjectGetComm((PetscObject)dmf,&comm));
  }
  PetscCallMPI(MPI_Comm_compare(PetscObjectComm((PetscObject)dmf),comm,&flag));
  PetscCheckFalse(flag != MPI_CONGRUENT && flag != MPI_IDENT,PetscObjectComm((PetscObject)dmf),PETSC_ERR_SUP,"cannot change communicators");
  PetscCall(DMRedundantCreate(comm,redf->rank,redf->N,dmc));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCreateInterpolation_Redundant(DM dmc,DM dmf,Mat *P,Vec *scale)
{
  DM_Redundant   *redc = (DM_Redundant*)dmc->data;
  DM_Redundant   *redf = (DM_Redundant*)dmf->data;
  PetscMPIInt    flag;
  PetscInt       i,rstart,rend;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_compare(PetscObjectComm((PetscObject)dmc),PetscObjectComm((PetscObject)dmf),&flag));
  PetscCheckFalse(flag != MPI_CONGRUENT && flag != MPI_IDENT,PetscObjectComm((PetscObject)dmf),PETSC_ERR_SUP,"cannot change communicators");
  PetscCheckFalse(redc->rank != redf->rank,PetscObjectComm((PetscObject)dmf),PETSC_ERR_ARG_INCOMP,"Owning rank does not match");
  PetscCheckFalse(redc->N != redf->N,PetscObjectComm((PetscObject)dmf),PETSC_ERR_ARG_INCOMP,"Global size does not match");
  PetscCall(MatCreate(PetscObjectComm((PetscObject)dmc),P));
  PetscCall(MatSetSizes(*P,redc->n,redc->n,redc->N,redc->N));
  PetscCall(MatSetType(*P,MATAIJ));
  PetscCall(MatSeqAIJSetPreallocation(*P,1,NULL));
  PetscCall(MatMPIAIJSetPreallocation(*P,1,NULL,0,NULL));
  PetscCall(MatGetOwnershipRange(*P,&rstart,&rend));
  for (i=rstart; i<rend; i++) PetscCall(MatSetValue(*P,i,i,1.0,INSERT_VALUES));
  PetscCall(MatAssemblyBegin(*P,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*P,MAT_FINAL_ASSEMBLY));
  if (scale) PetscCall(DMCreateInterpolationScale(dmc,dmf,*P,scale));
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
  PetscCall(PetscTryMethod(dm,"DMRedundantSetSize_C",(DM,PetscMPIInt,PetscInt),(dm,rank,N)));
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
  PetscCall(PetscUseMethod(dm,"DMRedundantGetSize_C",(DM,PetscMPIInt*,PetscInt*),(dm,rank,N)));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMRedundantSetSize_Redundant(DM dm,PetscMPIInt rank,PetscInt N)
{
  DM_Redundant   *red = (DM_Redundant*)dm->data;
  PetscMPIInt    myrank;
  PetscInt       i,*globals;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&myrank));
  red->rank = rank;
  red->N    = N;
  red->n    = (myrank == rank) ? N : 0;

  /* mapping is setup here */
  PetscCall(PetscMalloc1(red->N,&globals));
  for (i=0; i<red->N; i++) globals[i] = i;
  PetscCall(ISLocalToGlobalMappingDestroy(&dm->ltogmap));
  PetscCall(ISLocalToGlobalMappingCreate(PetscObjectComm((PetscObject)dm),1,red->N,globals,PETSC_OWN_POINTER,&dm->ltogmap));
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
  PetscCall(PetscNewLog(dm,&red));
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

  PetscCall(PetscObjectComposeFunction((PetscObject)dm,"DMRedundantSetSize_C",DMRedundantSetSize_Redundant));
  PetscCall(PetscObjectComposeFunction((PetscObject)dm,"DMRedundantGetSize_C",DMRedundantGetSize_Redundant));
  PetscCall(PetscObjectComposeFunction((PetscObject)dm,"DMSetUpGLVisViewer_C",DMSetUpGLVisViewer_Redundant));
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
  PetscCall(DMCreate(comm,dm));
  PetscCall(DMSetType(*dm,DMREDUNDANT));
  PetscCall(DMRedundantSetSize(*dm,rank,N));
  PetscCall(DMSetUp(*dm));
  PetscFunctionReturn(0);
}
