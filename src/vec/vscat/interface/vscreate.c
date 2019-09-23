/*
    The VECSCATTER (vec scatter) interface routines, callable by users.
*/

#include <petsc/private/vecscatterimpl.h>    /*I   "petscvec.h"    I*/

/* Logging support */
PetscClassId VEC_SCATTER_CLASSID;

PetscFunctionList VecScatterList              = NULL;
PetscBool         VecScatterRegisterAllCalled = PETSC_FALSE;

/*@C
  VecScatterSetType - Builds a vector scatter, for a particular vector scatter implementation.

  Collective on VecScatter

  Input Parameters:
+ vscat - The vector scatter object
- type - The name of the vector scatter type

  Options Database Key:
. -vecscatter_type <type> - Sets the vector scatter type; use -help for a list
                     of available types

  Notes:
  See "petsc/include/petscvec.h" for available vector scatter types (for instance, VECSCATTERMPI1, or VECSCATTERMPI3NODE).

  Use VecScatterDuplicate() to form additional vectors scatter of the same type as an existing vector scatter.

  Level: intermediate

.seealso: VecScatterGetType(), VecScatterCreate()
@*/
PetscErrorCode VecScatterSetType(VecScatter vscat, VecScatterType type)
{
  PetscBool      match;
  PetscErrorCode ierr;
  PetscErrorCode (*r)(VecScatter);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vscat, VEC_SCATTER_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)vscat, type, &match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  ierr = PetscFunctionListFind(VecScatterList,type,&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown vector scatter type: %s",type);

  if (vscat->ops->destroy) {
    ierr = (*vscat->ops->destroy)(vscat);CHKERRQ(ierr);
    vscat->ops->destroy = NULL;
  }

  ierr = (*r)(vscat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  VecScatterGetType - Gets the vector scatter type name (as a string) from the VecScatter.

  Not Collective

  Input Parameter:
. vscat  - The vector scatter

  Output Parameter:
. type - The vector scatter type name

  Level: intermediate

.seealso: VecScatterSetType(), VecScatterCreate()
@*/
PetscErrorCode VecScatterGetType(VecScatter vscat, VecScatterType *type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vscat, VEC_SCATTER_CLASSID,1);
  PetscValidPointer(type,2);
  ierr = VecScatterRegisterAll();CHKERRQ(ierr);
  *type = ((PetscObject)vscat)->type_name;
  PetscFunctionReturn(0);
}

/*@
  VecScatterSetFromOptions - Configures the vector scatter from the options database.

  Collective on VecScatter

  Input Parameter:
. vscat - The vector scatter

  Notes:
    To see all options, run your program with the -help option, or consult the users manual.
          Must be called before VecScatterSetUp() but before the vector scatter is used.

  Level: beginner


.seealso: VecScatterCreate(), VecScatterDestroy(), VecScatterSetUp()
@*/
PetscErrorCode VecScatterSetFromOptions(VecScatter vscat)
{
  PetscErrorCode ierr;
  PetscBool      opt;
  VecScatterType defaultType;
  char           typeName[256];
  PetscMPIInt    size;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vscat,VEC_SCATTER_CLASSID,1);

  ierr = PetscObjectOptionsBegin((PetscObject)vscat);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)vscat), &size);CHKERRQ(ierr);

  /* Handle vector type options */
  if (((PetscObject)vscat)->type_name) {
    defaultType = ((PetscObject)vscat)->type_name;
  } else {
    if (size > 1) defaultType = VECSCATTERMPI1;
    else defaultType = VECSCATTERSEQ;
  }

  ierr = VecScatterRegisterAll();CHKERRQ(ierr);
  ierr = PetscOptionsFList("-vecscatter_type","Vector Scatter type","VecScatterSetType",VecScatterList,defaultType,typeName,256,&opt);CHKERRQ(ierr);
  if (size > 1 && opt) {
    ierr = VecScatterSetType(vscat,typeName);CHKERRQ(ierr);
  } else {
    ierr = VecScatterSetType(vscat,defaultType);CHKERRQ(ierr);
  }

  vscat->beginandendtogether = PETSC_FALSE;
  ierr = PetscOptionsBool("-vecscatter_merge","Use combined (merged) vector scatter begin and end","VecScatterCreate",vscat->beginandendtogether,&vscat->beginandendtogether,NULL);CHKERRQ(ierr);
  if (vscat->beginandendtogether) {
    ierr = PetscInfo(vscat,"Using combined (merged) vector scatter begin and end\n");CHKERRQ(ierr);
  }

  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  VecScatterRegister -  Adds a new vector scatter component implementation

  Not Collective

  Input Parameters:
+ name        - The name of a new user-defined creation routine
- create_func - The creation routine itself

  Notes:
  VecScatterRegister() may be called multiple times to add several user-defined vectors

  Sample usage:
.vb
    VecScatterRegister("my_vecscatter",MyVecScatterCreate);
.ve

  Then, your vector scatter type can be chosen with the procedural interface via
.vb
    VecScatterCreate(MPI_Comm, VecScatter *);
    VecScatterSetType(VecScatter,"my_vectorscatter_name");
.ve
   or at runtime via the option
.vb
    -vecscatter_type my_vectorscatter_name
.ve

  Level: advanced

.seealso: VecScatterRegisterAll(), VecScatterRegisterDestroy()
@*/
PetscErrorCode VecScatterRegister(const char sname[], PetscErrorCode (*function)(VecScatter))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecInitializePackage();CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&VecScatterList,sname,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
/*@
   VecScatterCreate - Creates a vector scatter context.

   Collective on Vec

   Input Parameters:
+  xin - a vector that defines the shape (parallel data layout of the vector)
         of vectors from which we scatter
.  yin - a vector that defines the shape (parallel data layout of the vector)
         of vectors to which we scatter
.  ix - the indices of xin to scatter (if NULL scatters all values)
-  iy - the indices of yin to hold results (if NULL fills entire vector yin)

   Output Parameter:
.  newctx - location to store the new scatter context

   Options Database Keys:
.  -vecscatter_view         - Prints detail of communications
.  -vecscatter_view ::ascii_info    - Print less details about communication
.  -vecscatter_merge        - VecScatterBegin() handles all of the communication, VecScatterEnd() is a nop
                              eliminates the chance for overlap of computation and communication
-  -vecscatter_packtogether - Pack all messages before sending, receive all messages before unpacking
                              will make the results of scatters deterministic when otherwise they are not (it may be slower also).
.  -vecscatter_type sf      - Use the PetscSF implementation of vecscatter (Default). One can use PetscSF options to control the communication.

    Level: intermediate

  Notes:
   If both xin and yin are parallel, their communicator must be on the same
   set of processes, but their process order can be different.
   In calls to VecScatter() you can use different vectors than the xin and
   yin you used above; BUT they must have the same parallel data layout, for example,
   they could be obtained from VecDuplicate().
   A VecScatter context CANNOT be used in two or more simultaneous scatters;
   that is you cannot call a second VecScatterBegin() with the same scatter
   context until the VecScatterEnd() has been called on the first VecScatterBegin().
   In this case a separate VecScatter is needed for each concurrent scatter.

   Currently the MPI_Send() use PERSISTENT versions.
   (this unfortunately requires that the same in and out arrays be used for each use, this
    is why  we always need to pack the input into the work array before sending
    and unpack upon receiving instead of using MPI datatypes to avoid the packing/unpacking).

   Both ix and iy cannot be NULL at the same time.

   Use VecScatterCreateToAll() to create a vecscatter that copies an MPI vector to sequential vectors on all MPI ranks.
   Use VecScatterCreateToZero() to create a vecscatter that copies an MPI vector to a sequential vector on MPI rank 0.
   These special vecscatters have better performance than general ones.

.seealso: VecScatterDestroy(), VecScatterCreateToAll(), VecScatterCreateToZero(), PetscSFCreate()
@*/
PetscErrorCode VecScatterCreate(Vec xin,IS ix,Vec yin,IS iy,VecScatter *newctx)
{
  VecScatter        ctx;
  PetscErrorCode    ierr;
  PetscMPIInt       xsize,ysize,result;
  PetscBool         vseq = PETSC_TRUE;
  MPI_Comm          comm,xcomm,ycomm;

  PetscFunctionBegin;
  PetscValidPointer(newctx,5);
  *newctx = NULL;

  if (!ix && !iy) SETERRQ(PetscObjectComm((PetscObject)xin),PETSC_ERR_SUP,"Cannot pass default in for both input and output indices");

  /* Get comm from xin and yin */
  ierr = PetscObjectGetComm((PetscObject)xin,&xcomm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(xcomm,&xsize);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)yin,&ycomm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(ycomm,&ysize);CHKERRQ(ierr);
  vseq = ysize > 1 ? PETSC_FALSE : vseq;
  vseq = xsize > 1 ? PETSC_FALSE : vseq;
  if (xsize > 1 && ysize > 1) {
    ierr = MPI_Comm_compare(xcomm,ycomm,&result);CHKERRQ(ierr);
    if (result == MPI_UNEQUAL) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMECOMM,"VecScatterCreate: parallel vectors xin and yin must have identical/congruent/similar communicators");
  }

  /* If xsize==ysize, we use ycomm as comm of ctx, which does affect correctness of -vecscatter_type mpi1. But it does not
     affect -vecscatter_type sf, since sf internally uses the roots' comm (equal to xcomm when x is parallel) as the sf
     object's comm. One day when we degrade VecScatter from a petsc object to a simple wrapper around PetscSF, we can get
     rid of this confusion.
   */
  comm = xsize > ysize ? xcomm : ycomm;

  ierr = VecScatterInitializePackage();CHKERRQ(ierr);
  ierr = PetscHeaderCreate(ctx,VEC_SCATTER_CLASSID,"VecScatter","Vector Scatter","Vec",comm,VecScatterDestroy,VecScatterView);CHKERRQ(ierr);

  ctx->from_v  = xin;
  ctx->to_v    = yin;
  ctx->from_is = ix;
  ctx->to_is   = iy;
  ierr = VecGetLocalSize(xin,&ctx->from_n);CHKERRQ(ierr);
  ierr = VecGetLocalSize(yin,&ctx->to_n);CHKERRQ(ierr);

  /* Set default scatter type */
  ierr = VecScatterSetType(ctx,vseq ? VECSCATTERSEQ : VECSCATTERSF);CHKERRQ(ierr);

  ierr = VecScatterSetFromOptions(ctx);CHKERRQ(ierr);
  ierr = VecScatterSetUp(ctx);CHKERRQ(ierr);

  *newctx = ctx;
  PetscFunctionReturn(0);
}
