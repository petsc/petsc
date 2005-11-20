#include "zpetsc.h" 

void *PETSCNULLPOINTERADDRESS = PETSC_NULL;

/*MC
   PetscFortranAddr - a variable type in Fortran that can hold a
     regular C pointer.

   Notes: Used, for example, as the file argument in PetscFOpen()

   Level: beginner

.seealso:  PetscOffset, PetscInt
M*/
/*MC
   PetscOffset - a variable type in Fortran used with VecGetArray()
     and ISGetIndices()

   Level: beginner

.seealso:  PetscFortranAddr, PetscInt
M*/

/*
    This is code for translating PETSc memory addresses to integer offsets 
    for Fortran.
*/
char   *PETSC_NULL_CHARACTER_Fortran = 0;
void   *PETSC_NULL_INTEGER_Fortran   = 0;
void   *PETSC_NULL_OBJECT_Fortran    = 0;
void   *PETSC_NULL_Fortran           = 0;
void   *PETSC_NULL_SCALAR_Fortran    = 0;
void   *PETSC_NULL_DOUBLE_Fortran    = 0;
void   *PETSC_NULL_REAL_Fortran      = 0;
EXTERN_C_BEGIN
void   (*PETSC_NULL_FUNCTION_Fortran)(void) = 0;
EXTERN_C_END
size_t PetscIntAddressToFortran(PetscInt *base,PetscInt *addr)
{
  size_t tmp1 = (size_t) base,tmp2 = 0;
  size_t tmp3 = (size_t) addr;
  size_t itmp2;

#if !defined(PETSC_HAVE_CRAY90_POINTER)
  if (tmp3 > tmp1) {
    tmp2  = (tmp3 - tmp1)/sizeof(PetscInt);
    itmp2 = (size_t) tmp2;
  } else {
    tmp2  = (tmp1 - tmp3)/sizeof(PetscInt);
    itmp2 = -((size_t) tmp2);
  }
#else
  if (tmp3 > tmp1) {
    tmp2  = (tmp3 - tmp1);
    itmp2 = (size_t) tmp2;
  } else {
    tmp2  = (tmp1 - tmp3);
    itmp2 = -((size_t) tmp2);
  }
#endif

  if (base + itmp2 != addr) {
    (*PetscErrorPrintf)("PetscIntAddressToFortran:C and Fortran arrays are\n");
    (*PetscErrorPrintf)("not commonly aligned or are too far apart to be indexed \n");
    (*PetscErrorPrintf)("by an integer. Locations: C %uld Fortran %uld\n",tmp1,tmp3);
    MPI_Abort(PETSC_COMM_WORLD,1);
  }
  return itmp2;
}

PetscInt *PetscIntAddressFromFortran(PetscInt *base,size_t addr)
{
  return base + addr;
}

/*
       obj - PETSc object on which request is made
       base - Fortran array address
       addr - C array address
       res  - will contain offset from C to Fortran
       shift - number of bytes that prevent base and addr from being commonly aligned
       N - size of the array

*/
PetscErrorCode PetscScalarAddressToFortran(PetscObject obj,PetscScalar *base,PetscScalar *addr,PetscInt N,size_t *res)
{
  size_t   tmp1 = (size_t) base,tmp2 = tmp1/sizeof(PetscScalar);
  size_t   tmp3 = (size_t) addr;
  size_t   itmp2;
  PetscInt shift;

#if !defined(PETSC_HAVE_CRAY90_POINTER)
  if (tmp3 > tmp1) {  /* C is bigger than Fortran */
    tmp2  = (tmp3 - tmp1)/sizeof(PetscScalar);
    itmp2 = (size_t) tmp2;
    shift = (sizeof(PetscScalar) - (int)((tmp3 - tmp1) % sizeof(PetscScalar))) % sizeof(PetscScalar);
  } else {  
    tmp2  = (tmp1 - tmp3)/sizeof(PetscScalar);
    itmp2 = -((size_t) tmp2);
    shift = (int)((tmp1 - tmp3) % sizeof(PetscScalar));
  }
#else
  if (tmp3 > tmp1) {  /* C is bigger than Fortran */
    tmp2  = (tmp3 - tmp1);
    itmp2 = (size_t) tmp2;
  } else {  
    tmp2  = (tmp1 - tmp3);
    itmp2 = -((size_t) tmp2);
  }
  shift = 0;
#endif
  
  if (shift) { 
    /* 
        Fortran and C not PetscScalar aligned,recover by copying values into
        memory that is aligned with the Fortran
    */
    PetscErrorCode       ierr;
    PetscScalar          *work;
    PetscObjectContainer container;

    ierr = PetscMalloc((N+1)*sizeof(PetscScalar),&work);CHKERRQ(ierr); 

    /* shift work by that number of bytes */
    work = (PetscScalar*)(((char*)work) + shift);
    ierr = PetscMemcpy(work,addr,N*sizeof(PetscScalar));CHKERRQ(ierr);

    /* store in the first location in addr how much you shift it */
    ((PetscInt*)addr)[0] = shift;
 
    ierr = PetscObjectContainerCreate(PETSC_COMM_SELF,&container);CHKERRQ(ierr);
    ierr = PetscObjectContainerSetPointer(container,addr);CHKERRQ(ierr);
    ierr = PetscObjectCompose(obj,"GetArrayPtr",(PetscObject)container);CHKERRQ(ierr);

    tmp3 = (size_t) work;
    if (tmp3 > tmp1) {  /* C is bigger than Fortran */
      tmp2  = (tmp3 - tmp1)/sizeof(PetscScalar);
      itmp2 = (size_t) tmp2;
      shift = (sizeof(PetscScalar) - (int)((tmp3 - tmp1) % sizeof(PetscScalar))) % sizeof(PetscScalar);
    } else {  
      tmp2  = (tmp1 - tmp3)/sizeof(PetscScalar);
      itmp2 = -((size_t) tmp2);
      shift = (int)((tmp1 - tmp3) % sizeof(PetscScalar));
    }
    if (shift) {
      (*PetscErrorPrintf)("PetscScalarAddressToFortran:C and Fortran arrays are\n");
      (*PetscErrorPrintf)("not commonly aligned.\n");
      /* double/int doesn't work with ADIC */
      (*PetscErrorPrintf)("Locations/sizeof(PetscScalar): C %f Fortran %f\n",
                         ((PetscReal)tmp3)/(PetscReal)sizeof(PetscScalar),((PetscReal)tmp1)/(PetscReal)sizeof(PetscScalar));
      MPI_Abort(PETSC_COMM_WORLD,1);
    }
    ierr = PetscVerboseInfo(((void*)obj,"PetscScalarAddressToFortran:Efficiency warning, copying array in XXXGetArray() due\n\
    to alignment differences between C and Fortran\n"));CHKERRQ(ierr);
  }
  *res = itmp2;
  return 0;
}

/*
    obj - the PETSc object where the scalar pointer came from
    base - the Fortran array address
    addr - the Fortran offset from base
    N    - the amount of data

    lx   - the array space that is to be passed to XXXXRestoreArray()
*/     
PetscErrorCode PetscScalarAddressFromFortran(PetscObject obj,PetscScalar *base,size_t addr,PetscInt N,PetscScalar **lx)
{
  PetscErrorCode       ierr;
  PetscInt             shift;
  PetscObjectContainer container;
  PetscScalar          *tlx;

  ierr = PetscObjectQuery(obj,"GetArrayPtr",(PetscObject *)&container);CHKERRQ(ierr);
  if (container) {
    ierr  = PetscObjectContainerGetPointer(container,(void**)lx);CHKERRQ(ierr);
    tlx   = base + addr;

    shift = *(PetscInt*)*lx;
    ierr  = PetscMemcpy(*lx,tlx,N*sizeof(PetscScalar));CHKERRQ(ierr);
    tlx   = (PetscScalar*)(((char *)tlx) - shift);
    ierr = PetscFree(tlx);CHKERRQ(ierr);
    ierr = PetscObjectContainerDestroy(container);CHKERRQ(ierr);
    ierr = PetscObjectCompose(obj,"GetArrayPtr",0);CHKERRQ(ierr);
  } else {
    *lx = base + addr;
  }
  return 0;
}

#undef __FUNCT__  
#define __FUNCT__ "MPICCommToFortranComm"
/*@C
    MPICCommToFortranComm - Converts a MPI_Comm represented
    in C to one appropriate to pass to a Fortran routine.

    Not collective

    Input Parameter:
.   cobj - the C MPI_Comm

    Output Parameter:
.   fobj - the Fortran MPI_Comm

    Level: advanced

    Notes:
    MPICCommToFortranComm() must be called in a C/C++ routine.
    MPI 1 does not provide a standard for mapping between
    Fortran and C MPI communicators; this routine handles the
    mapping correctly on all machines.

.keywords: Fortran, C, MPI_Comm, convert, interlanguage

.seealso: MPIFortranCommToCComm()
@*/
PetscErrorCode MPICCommToFortranComm(MPI_Comm comm,int *fcomm)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  /* call to MPI_Comm_size() is for error checking on comm */
  ierr = MPI_Comm_size(comm,&size);
  if (ierr) SETERRQ(PETSC_ERR_ARG_CORRUPT ,"Invalid MPI communicator");

  *fcomm = PetscFromPointerComm(comm);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MPIFortranCommToCComm"
/*@C
    MPIFortranCommToCComm - Converts a MPI_Comm represented
    int Fortran (as an integer) to a MPI_Comm in C.

    Not collective

    Input Parameter:
.   fcomm - the Fortran MPI_Comm (an integer)

    Output Parameter:
.   comm - the C MPI_Comm

    Level: advanced

    Notes:
    MPIFortranCommToCComm() must be called in a C/C++ routine.
    MPI 1 does not provide a standard for mapping between
    Fortran and C MPI communicators; this routine handles the
    mapping correctly on all machines.

.keywords: Fortran, C, MPI_Comm, convert, interlanguage

.seealso: MPICCommToFortranComm()
@*/
PetscErrorCode MPIFortranCommToCComm(int fcomm,MPI_Comm *comm)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  *comm = (MPI_Comm)PetscToPointerComm(fcomm);
  /* call to MPI_Comm_size() is for error checking on comm */
  ierr = MPI_Comm_size(*comm,&size);
  if (ierr) SETERRQ(PETSC_ERR_ARG_CORRUPT,"Invalid MPI communicator");
  PetscFunctionReturn(0);
}



