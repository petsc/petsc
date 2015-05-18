#include <petsc/private/sfimpl.h>

#if !defined(PETSC_HAVE_MPI_TYPE_GET_ENVELOPE)
#define MPI_Type_get_envelope(datatype,num_ints,num_addrs,num_dtypes,combiner) (*(num_ints)=0,*(num_addrs)=0,*(num_dtypes)=0,*(combiner)=0,1);SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP_SYS,"Need an MPI-2 implementation")
#define MPI_Type_get_contents(datatype,num_ints,num_addrs,num_dtypes,ints,addrs,dtypes) (*(ints)=0,*(addrs)=0,*(dtypes)=0,1);SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP_SYS,"Need an MPI-2 implementation")
#endif
#if !defined(PETSC_HAVE_MPI_COMBINER_DUP)  /* We have no way to interpret output of MPI_Type_get_envelope without this. */
#  define MPI_COMBINER_DUP   0
#endif
#if !defined(PETSC_HAVE_MPI_COMBINER_CONTIGUOUS) && MPI_VERSION < 2
#  define MPI_COMBINER_CONTIGUOUS -1
#endif

#undef __FUNCT__
#define __FUNCT__ "MPIPetsc_Type_unwrap"
PetscErrorCode MPIPetsc_Type_unwrap(MPI_Datatype a,MPI_Datatype *atype)
{
  PetscMPIInt    nints,naddrs,ntypes,combiner;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Type_get_envelope(a,&nints,&naddrs,&ntypes,&combiner);CHKERRQ(ierr);
  if (combiner == MPI_COMBINER_DUP) {
    PetscMPIInt  ints[1];
    MPI_Aint     addrs[1];
    MPI_Datatype types[1];
    if (nints != 0 || naddrs != 0 || ntypes != 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Unexpected returns from MPI_Type_get_envelope()");
    ierr   = MPI_Type_get_contents(a,0,0,1,ints,addrs,types);CHKERRQ(ierr);
    *atype = types[0];
  } else *atype = a;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MPIPetsc_Type_compare"
PetscErrorCode MPIPetsc_Type_compare(MPI_Datatype a,MPI_Datatype b,PetscBool *match)
{
  PetscErrorCode ierr;
  MPI_Datatype   atype,btype;
  PetscMPIInt    aintcount,aaddrcount,atypecount,acombiner;
  PetscMPIInt    bintcount,baddrcount,btypecount,bcombiner;

  PetscFunctionBegin;
  ierr   = MPIPetsc_Type_unwrap(a,&atype);CHKERRQ(ierr);
  ierr   = MPIPetsc_Type_unwrap(b,&btype);CHKERRQ(ierr);
  *match = PETSC_FALSE;
  if (atype == btype) {
    *match = PETSC_TRUE;
    PetscFunctionReturn(0);
  }
  ierr = MPI_Type_get_envelope(atype,&aintcount,&aaddrcount,&atypecount,&acombiner);CHKERRQ(ierr);
  ierr = MPI_Type_get_envelope(btype,&bintcount,&baddrcount,&btypecount,&bcombiner);CHKERRQ(ierr);
  if (acombiner == bcombiner && aintcount == bintcount && aaddrcount == baddrcount && atypecount == btypecount && (aintcount > 0 || aaddrcount > 0 || atypecount > 0)) {
    PetscMPIInt  *aints,*bints;
    MPI_Aint     *aaddrs,*baddrs;
    MPI_Datatype *atypes,*btypes;
    PetscBool    same;
    ierr = PetscMalloc6(aintcount,&aints,bintcount,&bints,aaddrcount,&aaddrs,baddrcount,&baddrs,atypecount,&atypes,btypecount,&btypes);CHKERRQ(ierr);
    ierr = MPI_Type_get_contents(atype,aintcount,aaddrcount,atypecount,aints,aaddrs,atypes);CHKERRQ(ierr);
    ierr = MPI_Type_get_contents(btype,bintcount,baddrcount,btypecount,bints,baddrs,btypes);CHKERRQ(ierr);
    ierr = PetscMemcmp(aints,bints,aintcount*sizeof(aints[0]),&same);CHKERRQ(ierr);
    if (same) {
      ierr = PetscMemcmp(aaddrs,baddrs,aaddrcount*sizeof(aaddrs[0]),&same);CHKERRQ(ierr);
      if (same) {
        /* This comparison should be recursive */
        ierr = PetscMemcmp(atypes,btypes,atypecount*sizeof(atypes[0]),&same);CHKERRQ(ierr);
      }
    }
    ierr = PetscFree6(aints,bints,aaddrs,baddrs,atypes,btypes);CHKERRQ(ierr);
    if (same) *match = PETSC_TRUE;
    PetscFunctionReturn(0);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MPIPetsc_Type_compare_contig"
/* Check whether a was created via MPI_Type_contiguous from b
 *
 */
PetscErrorCode MPIPetsc_Type_compare_contig(MPI_Datatype a,MPI_Datatype b,PetscInt *n)
{
  PetscErrorCode ierr;
  MPI_Datatype   atype,btype;
  PetscMPIInt    aintcount,aaddrcount,atypecount,acombiner;

  PetscFunctionBegin;
  ierr = MPIPetsc_Type_unwrap(a,&atype);CHKERRQ(ierr);
  ierr = MPIPetsc_Type_unwrap(b,&btype);CHKERRQ(ierr);
  *n = PETSC_FALSE;
  if (atype == btype) {
    *n = 1;
    PetscFunctionReturn(0);
  }
  ierr = MPI_Type_get_envelope(atype,&aintcount,&aaddrcount,&atypecount,&acombiner);CHKERRQ(ierr);
  if (acombiner == MPI_COMBINER_CONTIGUOUS && aintcount >= 1) {
    PetscMPIInt  *aints;
    MPI_Aint     *aaddrs;
    MPI_Datatype *atypes;
    ierr = PetscMalloc3(aintcount,&aints,aaddrcount,&aaddrs,atypecount,&atypes);CHKERRQ(ierr);
    ierr = MPI_Type_get_contents(atype,aintcount,aaddrcount,atypecount,aints,aaddrs,atypes);CHKERRQ(ierr);
    if (atypes[0] == btype) *n = aints[0];
    ierr = PetscFree3(aints,aaddrs,atypes);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  PetscFunctionReturn(0);
}
