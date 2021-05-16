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
  PetscErrorCode         ierr;
  ISLocalToGlobalMapping ltog;
  PetscInt               i,rstart,rend,*cols;
  PetscScalar            *vals;

  PetscFunctionBegin;
  ierr = MatCreate(PetscObjectComm((PetscObject)dm),J);CHKERRQ(ierr);
  ierr = MatSetSizes(*J,red->n,red->n,red->N,red->N);CHKERRQ(ierr);
  ierr = MatSetType(*J,dm->mattype);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(*J,red->n,NULL);CHKERRQ(ierr);
  ierr = MatSeqBAIJSetPreallocation(*J,1,red->n,NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(*J,red->n,NULL,red->N-red->n,NULL);CHKERRQ(ierr);
  ierr = MatMPIBAIJSetPreallocation(*J,1,red->n,NULL,red->N-red->n,NULL);CHKERRQ(ierr);

  ierr = DMGetLocalToGlobalMapping(dm,&ltog);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(*J,ltog,ltog);CHKERRQ(ierr);
  ierr = MatSetDM(*J,dm);CHKERRQ(ierr);

  ierr = PetscMalloc2(red->N,&cols,red->N,&vals);CHKERRQ(ierr);
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

static PetscErrorCode DMDestroy_Redundant(DM dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectComposeFunction((PetscObject)dm,"DMRedundantSetSize_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)dm,"DMRedundantGetSize_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)dm,"DMSetUpGLVisViewer_C",NULL);CHKERRQ(ierr);
  /* This was originally freed in DMDestroy(), but that prevents reference counting of backend objects */
  ierr = PetscFree(dm->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCreateGlobalVector_Redundant(DM dm,Vec *gvec)
{
  PetscErrorCode         ierr;
  DM_Redundant           *red = (DM_Redundant*)dm->data;
  ISLocalToGlobalMapping ltog;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(gvec,2);
  *gvec = NULL;
  ierr  = VecCreate(PetscObjectComm((PetscObject)dm),gvec);CHKERRQ(ierr);
  ierr  = VecSetSizes(*gvec,red->n,red->N);CHKERRQ(ierr);
  ierr  = VecSetType(*gvec,dm->vectype);CHKERRQ(ierr);
  ierr  = DMGetLocalToGlobalMapping(dm,&ltog);CHKERRQ(ierr);
  ierr  = VecSetLocalToGlobalMapping(*gvec,ltog);CHKERRQ(ierr);
  ierr  = VecSetDM(*gvec,dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCreateLocalVector_Redundant(DM dm,Vec *lvec)
{
  PetscErrorCode ierr;
  DM_Redundant   *red = (DM_Redundant*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(lvec,2);
  *lvec = NULL;
  ierr  = VecCreate(PETSC_COMM_SELF,lvec);CHKERRQ(ierr);
  ierr  = VecSetSizes(*lvec,red->N,red->N);CHKERRQ(ierr);
  ierr  = VecSetType(*lvec,dm->vectype);CHKERRQ(ierr);
  ierr  = VecSetDM(*lvec,dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMLocalToGlobalBegin_Redundant(DM dm,Vec l,InsertMode imode,Vec g)
{
  PetscErrorCode    ierr;
  DM_Redundant      *red = (DM_Redundant*)dm->data;
  const PetscScalar *lv;
  PetscScalar       *gv;
  PetscMPIInt       rank;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank);CHKERRMPI(ierr);
  ierr = VecGetArrayRead(l,&lv);CHKERRQ(ierr);
  ierr = VecGetArray(g,&gv);CHKERRQ(ierr);
  switch (imode) {
  case ADD_VALUES:
  case MAX_VALUES:
  {
    void        *source;
    PetscScalar *buffer;
    PetscInt    i;
    if (rank == red->rank) {
#if defined(PETSC_HAVE_MPI_IN_PLACE)
      buffer = gv;
      source = MPI_IN_PLACE;
#else
      ierr   = PetscMalloc1(red->N,&buffer);CHKERRQ(ierr);
      source = buffer;
#endif
      if (imode == ADD_VALUES) for (i=0; i<red->N; i++) buffer[i] = gv[i] + lv[i];
#if !defined(PETSC_USE_COMPLEX)
      if (imode == MAX_VALUES) for (i=0; i<red->N; i++) buffer[i] = PetscMax(gv[i],lv[i]);
#endif
    } else source = (void*)lv;
    ierr = MPI_Reduce(source,gv,red->N,MPIU_SCALAR,(imode == ADD_VALUES) ? MPIU_SUM : MPIU_MAX,red->rank,PetscObjectComm((PetscObject)dm));CHKERRMPI(ierr);
#if !defined(PETSC_HAVE_MPI_IN_PLACE)
    if (rank == red->rank) {ierr = PetscFree(buffer);CHKERRQ(ierr);}
#endif
  } break;
  case INSERT_VALUES:
    ierr = PetscArraycpy(gv,lv,red->n);CHKERRQ(ierr);
    break;
  default: SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"InsertMode not supported");
  }
  ierr = VecRestoreArrayRead(l,&lv);CHKERRQ(ierr);
  ierr = VecRestoreArray(g,&gv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMLocalToGlobalEnd_Redundant(DM dm,Vec l,InsertMode imode,Vec g)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

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
    if (red->n) {ierr = PetscArraycpy(lv,gv,red->n);CHKERRQ(ierr);}
    ierr = MPI_Bcast(lv,red->N,MPIU_SCALAR,red->rank,PetscObjectComm((PetscObject)dm));CHKERRMPI(ierr);
    break;
  default: SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"InsertMode not supported");
  }
  ierr = VecRestoreArrayRead(g,&gv);CHKERRQ(ierr);
  ierr = VecRestoreArray(l,&lv);CHKERRQ(ierr);
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

static PetscErrorCode DMCreateColoring_Redundant(DM dm,ISColoringType ctype,ISColoring *coloring)
{
  DM_Redundant    *red = (DM_Redundant*)dm->data;
  PetscErrorCode  ierr;
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
  default: SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONG,"Unknown ISColoringType %d",(int)ctype);
  }
  ierr = PetscMalloc1(nloc,&colors);CHKERRQ(ierr);
  for (i=0; i<nloc; i++) colors[i] = i;
  ierr = ISColoringCreate(PetscObjectComm((PetscObject)dm),red->N,nloc,colors,PETSC_OWN_POINTER,coloring);CHKERRQ(ierr);
  ierr = ISColoringSetType(*coloring,ctype);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMRefine_Redundant(DM dmc,MPI_Comm comm,DM *dmf)
{
  PetscErrorCode ierr;
  PetscMPIInt    flag;
  DM_Redundant   *redc = (DM_Redundant*)dmc->data;

  PetscFunctionBegin;
  if (comm == MPI_COMM_NULL) {
    ierr = PetscObjectGetComm((PetscObject)dmc,&comm);CHKERRQ(ierr);
  }
  ierr = MPI_Comm_compare(PetscObjectComm((PetscObject)dmc),comm,&flag);CHKERRMPI(ierr);
  if (flag != MPI_CONGRUENT && flag != MPI_IDENT) SETERRQ(PetscObjectComm((PetscObject)dmc),PETSC_ERR_SUP,"cannot change communicators");
  ierr = DMRedundantCreate(comm,redc->rank,redc->N,dmf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCoarsen_Redundant(DM dmf,MPI_Comm comm,DM *dmc)
{
  PetscErrorCode ierr;
  PetscMPIInt    flag;
  DM_Redundant   *redf = (DM_Redundant*)dmf->data;

  PetscFunctionBegin;
  if (comm == MPI_COMM_NULL) {
    ierr = PetscObjectGetComm((PetscObject)dmf,&comm);CHKERRQ(ierr);
  }
  ierr = MPI_Comm_compare(PetscObjectComm((PetscObject)dmf),comm,&flag);CHKERRMPI(ierr);
  if (flag != MPI_CONGRUENT && flag != MPI_IDENT) SETERRQ(PetscObjectComm((PetscObject)dmf),PETSC_ERR_SUP,"cannot change communicators");
  ierr = DMRedundantCreate(comm,redf->rank,redf->N,dmc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCreateInterpolation_Redundant(DM dmc,DM dmf,Mat *P,Vec *scale)
{
  PetscErrorCode ierr;
  DM_Redundant   *redc = (DM_Redundant*)dmc->data;
  DM_Redundant   *redf = (DM_Redundant*)dmf->data;
  PetscMPIInt    flag;
  PetscInt       i,rstart,rend;

  PetscFunctionBegin;
  ierr = MPI_Comm_compare(PetscObjectComm((PetscObject)dmc),PetscObjectComm((PetscObject)dmf),&flag);CHKERRMPI(ierr);
  if (flag != MPI_CONGRUENT && flag != MPI_IDENT) SETERRQ(PetscObjectComm((PetscObject)dmf),PETSC_ERR_SUP,"cannot change communicators");
  if (redc->rank != redf->rank) SETERRQ(PetscObjectComm((PetscObject)dmf),PETSC_ERR_ARG_INCOMP,"Owning rank does not match");
  if (redc->N != redf->N) SETERRQ(PetscObjectComm((PetscObject)dmf),PETSC_ERR_ARG_INCOMP,"Global size does not match");
  ierr = MatCreate(PetscObjectComm((PetscObject)dmc),P);CHKERRQ(ierr);
  ierr = MatSetSizes(*P,redc->n,redc->n,redc->N,redc->N);CHKERRQ(ierr);
  ierr = MatSetType(*P,MATAIJ);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(*P,1,NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(*P,1,NULL,0,NULL);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(*P,&rstart,&rend);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) {ierr = MatSetValue(*P,i,i,1.0,INSERT_VALUES);CHKERRQ(ierr);}
  ierr = MatAssemblyBegin(*P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (scale) {ierr = DMCreateInterpolationScale(dmc,dmf,*P,scale);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*@
    DMRedundantSetSize - Sets the size of a densely coupled redundant object

    Collective on dm

    Input Parameter:
+   dm - redundant DM
.   rank - rank of process to own redundant degrees of freedom
-   N - total number of redundant degrees of freedom

    Level: advanced

.seealso DMDestroy(), DMCreateGlobalVector(), DMRedundantCreate(), DMRedundantGetSize()
@*/
PetscErrorCode DMRedundantSetSize(DM dm,PetscMPIInt rank,PetscInt N)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidType(dm,1);
  PetscValidLogicalCollectiveMPIInt(dm,rank,2);
  PetscValidLogicalCollectiveInt(dm,N,3);
  ierr = PetscTryMethod(dm,"DMRedundantSetSize_C",(DM,PetscMPIInt,PetscInt),(dm,rank,N));CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidType(dm,1);
  ierr = PetscUseMethod(dm,"DMRedundantGetSize_C",(DM,PetscMPIInt*,PetscInt*),(dm,rank,N));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMRedundantSetSize_Redundant(DM dm,PetscMPIInt rank,PetscInt N)
{
  DM_Redundant   *red = (DM_Redundant*)dm->data;
  PetscErrorCode ierr;
  PetscMPIInt    myrank;
  PetscInt       i,*globals;

  PetscFunctionBegin;
  ierr      = MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&myrank);CHKERRMPI(ierr);
  red->rank = rank;
  red->N    = N;
  red->n    = (myrank == rank) ? N : 0;

  /* mapping is setup here */
  ierr = PetscMalloc1(red->N,&globals);CHKERRQ(ierr);
  for (i=0; i<red->N; i++) globals[i] = i;
  ierr = ISLocalToGlobalMappingDestroy(&dm->ltogmap);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreate(PetscObjectComm((PetscObject)dm),1,red->N,globals,PETSC_OWN_POINTER,&dm->ltogmap);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  DM_Redundant   *red;

  PetscFunctionBegin;
  ierr     = PetscNewLog(dm,&red);CHKERRQ(ierr);
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

  ierr = PetscObjectComposeFunction((PetscObject)dm,"DMRedundantSetSize_C",DMRedundantSetSize_Redundant);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)dm,"DMRedundantGetSize_C",DMRedundantGetSize_Redundant);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)dm,"DMSetUpGLVisViewer_C",DMSetUpGLVisViewer_Redundant);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
    DMRedundantCreate - Creates a DM object, used to manage data for dense globally coupled variables

    Collective

    Input Parameter:
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(dm,4);
  ierr = DMCreate(comm,dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm,DMREDUNDANT);CHKERRQ(ierr);
  ierr = DMRedundantSetSize(*dm,rank,N);CHKERRQ(ierr);
  ierr = DMSetUp(*dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
