/*
    Code for allocating Unix shared memory on MPI rank 0 and later accessing it from other MPI processes
*/
#include <petsc/private/petscimpl.h>
#include <petscsys.h>

PetscBool PCMPIServerActive    = PETSC_FALSE; // PETSc is running in server mode
PetscBool PCMPIServerInSolve   = PETSC_FALSE; // A parallel server solve is occurring
PetscBool PCMPIServerUseShmget = PETSC_TRUE;  // Use Unix shared memory for distributing objects

#if defined(PETSC_HAVE_SHMGET)
  #include <sys/shm.h>
  #include <sys/mman.h>
  #include <errno.h>

typedef struct _PetscShmgetAllocation *PetscShmgetAllocation;
struct _PetscShmgetAllocation {
  void                 *addr; // address on this process; points to same physical address on all processes
  int                   shmkey, shmid;
  size_t                sz;
  PetscShmgetAllocation next;
};
static PetscShmgetAllocation allocations = NULL;

typedef struct {
  size_t shmkey[3];
  size_t sz[3];
} BcastInfo;

#endif

/*@C
  PetscShmgetAddressesFinalize - frees any shared memory that was allocated by `PetscShmgetAllocateArray()` but
  not deallocated with `PetscShmgetDeallocateArray()`

  Level: developer

  Notes:
  This prevents any shared memory allocated, but not deallocated, from remaining on the system and preventing
  its future use.

  If the program crashes outstanding shared memory allocations may remain.

.seealso: `PetscShmgetAllocateArray()`, `PetscShmgetDeallocateArray()`, `PetscShmgetUnmapAddresses()`
@*/
PetscErrorCode PetscShmgetAddressesFinalize(void)
{
  PetscFunctionBegin;
#if defined(PETSC_HAVE_SHMGET)
  PetscShmgetAllocation next = allocations, previous = NULL;

  while (next) {
    PetscCheck(!shmctl(next->shmid, IPC_RMID, NULL), PETSC_COMM_SELF, PETSC_ERR_SYS, "Unable to free shared memory key %d shmid %d %s, see PCMPIServerBegin()", next->shmkey, next->shmid, strerror(errno));
    previous = next;
    next     = next->next;
    PetscCall(PetscFree(previous));
  }
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* takes a void so can work bsan safe with PetscObjectContainerCompose() */
PetscErrorCode PCMPIServerAddressesDestroy(void **ctx)
{
  PCMPIServerAddresses *addresses = (PCMPIServerAddresses *)*ctx;

  PetscFunctionBegin;
#if defined(PETSC_HAVE_SHMGET)
  PetscCall(PetscShmgetUnmapAddresses(addresses->n, addresses->addr));
  PetscCall(PetscFree(addresses));
#else
  (void)addresses;
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscShmgetMapAddresses - given shared address on the first MPI process determines the
  addresses on the other MPI processes that map to the same physical memory

  Input Parameters:
+ comm       - the `MPI_Comm` to scatter the address
. n          - the number of addresses, each obtained on MPI process zero by `PetscShmgetAllocateArray()`
- baseaddres - the addresses on the first MPI process, ignored on all but first process

  Output Parameter:
. addres - the addresses on each MPI process, the array of void * must already be allocated

  Level: developer

  Note:
  This routine does nothing if `PETSC_HAVE_SHMGET` is not defined

.seealso: `PetscShmgetDeallocateArray()`, `PetscShmgetAllocateArray()`, `PetscShmgetUnmapAddresses()`
@*/
PetscErrorCode PetscShmgetMapAddresses(MPI_Comm comm, PetscInt n, const void **baseaddres, void **addres)
{
  PetscFunctionBegin;
#if defined(PETSC_HAVE_SHMGET)
  if (PetscGlobalRank == 0) {
    BcastInfo bcastinfo = {
      {0, 0, 0},
      {0, 0, 0}
    };
    for (PetscInt i = 0; i < n; i++) {
      PetscShmgetAllocation allocation = allocations;

      while (allocation) {
        if (allocation->addr == baseaddres[i]) {
          bcastinfo.shmkey[i] = allocation->shmkey;
          bcastinfo.sz[i]     = allocation->sz;
          addres[i]           = (void *)baseaddres[i];
          break;
        }
        allocation = allocation->next;
      }
      PetscCheck(allocation, comm, PETSC_ERR_PLIB, "Unable to locate PCMPI allocated shared address %p, see PCMPIServerBegin()", baseaddres[i]);
    }
    PetscCall(PetscInfo(NULL, "Mapping PCMPI Server array %p\n", addres[0]));
    PetscCallMPI(MPI_Bcast(&bcastinfo, 6, MPIU_SIZE_T, 0, comm));
  } else {
    BcastInfo bcastinfo = {
      {0, 0, 0},
      {0, 0, 0}
    };
    int    shmkey = 0;
    size_t sz     = 0;

    PetscCallMPI(MPI_Bcast(&bcastinfo, 6, MPIU_SIZE_T, 0, comm));
    for (PetscInt i = 0; i < n; i++) {
      PetscShmgetAllocation next = allocations, previous = NULL;

      shmkey = (int)bcastinfo.shmkey[i];
      sz     = bcastinfo.sz[i];
      while (next) {
        if (next->shmkey == shmkey) addres[i] = next->addr;
        previous = next;
        next     = next->next;
      }
      if (!next) {
        PetscShmgetAllocation allocation;
        PetscCall(PetscCalloc(sizeof(struct _PetscShmgetAllocation), &allocation));
        allocation->shmkey = shmkey;
        allocation->sz     = sz;
        allocation->shmid  = shmget(allocation->shmkey, allocation->sz, 0666);
        PetscCheck(allocation->shmid != -1, PETSC_COMM_SELF, PETSC_ERR_SYS, "Unable to map PCMPI shared memory key %d of size %d, see PCMPIServerBegin()", allocation->shmkey, (int)allocation->sz);
        allocation->addr = shmat(allocation->shmid, NULL, 0);
        PetscCheck(allocation->addr, PETSC_COMM_SELF, PETSC_ERR_SYS, "Unable to map PCMPI shared memory key %d, see PCMPIServerBegin()", allocation->shmkey);
        addres[i] = allocation->addr;
        if (previous) previous->next = allocation;
        else allocations = allocation;
      }
    }
  }
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscShmgetUnmapAddresses - given shared addresses on a MPI process unlink it

  Input Parameters:
+ n      - the number of addresses, each obtained on MPI process zero by `PetscShmgetAllocateArray()`
- addres - the addresses

  Level: developer

  Note:
  This routine does nothing if `PETSC_HAVE_SHMGET` is not defined

.seealso: `PetscShmgetDeallocateArray()`, `PetscShmgetAllocateArray()`, `PetscShmgetMapAddresses()`
@*/
PetscErrorCode PetscShmgetUnmapAddresses(PetscInt n, void **addres) PeNS
{
  PetscFunctionBegin;
#if defined(PETSC_HAVE_SHMGET)
  if (PetscGlobalRank > 0) {
    for (PetscInt i = 0; i < n; i++) {
      PetscShmgetAllocation next = allocations, previous = NULL;
      PetscBool             found = PETSC_FALSE;

      while (next) {
        if (next->addr == addres[i]) {
          PetscCheck(!shmdt(next->addr), PETSC_COMM_SELF, PETSC_ERR_SYS, "Unable to shmdt() location %s, see PCMPIServerBegin()", strerror(errno));
          if (previous) previous->next = next->next;
          else allocations = next->next;
          PetscCall(PetscFree(next));
          found = PETSC_TRUE;
          break;
        }
        previous = next;
        next     = next->next;
      }
      PetscCheck(found, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unable to find address %p to unmap, see PCMPIServerBegin()", addres[i]);
    }
  }
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscShmgetAllocateArray - allocates shared memory accessible by all MPI processes in the server

  Not Collective, only called on the first MPI process

  Input Parameters:
+ sz  - the number of elements in the array
- asz - the size of an entry in the array, for example `sizeof(PetscScalar)`

  Output Parameters:
. addr - the address of the array

  Level: developer

  Notes:
  Uses `PetscMalloc()` if `PETSC_HAVE_SHMGET` is not defined or the MPI linear solver server is not running

  Sometimes when a program crashes, shared memory IDs may remain, making it impossible to rerun the program.
  Use
.vb
  $PETSC_DIR/lib/petsc/bin/petscfreesharedmemory
.ve to free that memory. The Linux command `ipcrm --all` or macOS command `for i in $(ipcs -m | tail -$(expr $(ipcs -m | wc -l) - 3) | tr -s ' ' | cut -d" " -f3); do ipcrm -M $i; done`
  will also free the memory.

  Use the Unix command `ipcs -m` to see what memory IDs are currently allocated and `ipcrm -m ID` to remove a memory ID

  Under Apple macOS the following file must be copied to /Library/LaunchDaemons/sharedmemory.plist (ensure this file is owned by root and not the user)
  and the machine rebooted before using shared memory
.vb
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
 <key>Label</key>
 <string>shmemsetup</string>
 <key>UserName</key>
 <string>root</string>
 <key>GroupName</key>
 <string>wheel</string>
 <key>ProgramArguments</key>
 <array>
 <string>/usr/sbin/sysctl</string>
 <string>-w</string>
 <string>kern.sysv.shmmax=4194304000</string>
 <string>kern.sysv.shmmni=2064</string>
 <string>kern.sysv.shmseg=2064</string>
 <string>kern.sysv.shmall=131072000</string>
  </array>
 <key>KeepAlive</key>
 <false/>
 <key>RunAtLoad</key>
 <true/>
</dict>
</plist>
.ve

  Use the command
.vb
  /usr/sbin/sysctl -a | grep shm
.ve
  to confirm that the shared memory limits you have requested are available.

  Fortran Note:
  The calling sequence is `PetscShmgetAllocateArray[Scalar,Int](PetscInt start, PetscInt len, Petsc[Scalar,Int], pointer :: d1(:), ierr)`

  Developer Note:
  More specifically this uses `PetscMalloc()` if `!PCMPIServerUseShmget` || `!PCMPIServerActive` || `PCMPIServerInSolve`
  where `PCMPIServerInSolve` indicates that the solve is nested inside a MPI linear solver server solve and hence should
  not allocate the vector and matrix memory in shared memory.

.seealso: [](sec_pcmpi), `PCMPIServerBegin()`, `PCMPI`, `KSPCheckPCMPI()`, `PetscShmgetDeallocateArray()`
@*/
PetscErrorCode PetscShmgetAllocateArray(size_t sz, size_t asz, void *addr[])
{
  PetscFunctionBegin;
  if (!PCMPIServerUseShmget || !PCMPIServerActive || PCMPIServerInSolve) PetscCall(PetscMalloc(sz * asz, addr));
#if defined(PETSC_HAVE_SHMGET)
  else {
    PetscShmgetAllocation allocation;
    static int            shmkeys = 10;

    PetscCall(PetscCalloc(sizeof(struct _PetscShmgetAllocation), &allocation));
    allocation->shmkey = shmkeys++;
    allocation->sz     = sz * asz;
    allocation->shmid  = shmget(allocation->shmkey, allocation->sz, 0666 | IPC_CREAT);
    PetscCheck(allocation->shmid != -1, PETSC_COMM_SELF, PETSC_ERR_LIB, "Unable to schmget() of size %d with key %d %s see PetscShmgetAllocateArray()", (int)allocation->sz, allocation->shmkey, strerror(errno));
    allocation->addr = shmat(allocation->shmid, NULL, 0);
    PetscCheck(allocation->addr, PETSC_COMM_SELF, PETSC_ERR_LIB, "Unable to shmat() of shmid %d %s", allocation->shmid, strerror(errno));
  #if PETSC_SIZEOF_VOID_P == 8
    PetscCheck((uint64_t)allocation->addr != 0xffffffffffffffff, PETSC_COMM_SELF, PETSC_ERR_LIB, "shmat() of shmid %d returned 0xffffffffffffffff %s, see PCMPIServerBegin()", allocation->shmid, strerror(errno));
  #endif

    if (!allocations) allocations = allocation;
    else {
      PetscShmgetAllocation next = allocations;
      while (next->next) next = next->next;
      next->next = allocation;
    }
    *addr = allocation->addr;
    PetscCall(PetscInfo(NULL, "Allocating PCMPI Server array %p shmkey %d shmid %d size %d\n", *addr, allocation->shmkey, allocation->shmid, (int)allocation->sz));
  }
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscShmgetDeallocateArray - deallocates shared memory accessible by all MPI processes in the server

  Not Collective, only called on the first MPI process

  Input Parameter:
. addr - the address of array

  Level: developer

  Note:
  Uses `PetscFree()` if `PETSC_HAVE_SHMGET` is not defined or the MPI linear solver server is not running

  Fortran Note:
  The calling sequence is `PetscShmgetDeallocateArray[Scalar,Int](Petsc[Scalar,Int], pointer :: d1(:), ierr)`

.seealso: [](sec_pcmpi), `PCMPIServerBegin()`, `PCMPI`, `KSPCheckPCMPI()`, `PetscShmgetAllocateArray()`
@*/
PetscErrorCode PetscShmgetDeallocateArray(void *addr[])
{
  PetscFunctionBegin;
  if (!*addr) PetscFunctionReturn(PETSC_SUCCESS);
  if (!PCMPIServerUseShmget || !PCMPIServerActive || PCMPIServerInSolve) PetscCall(PetscFree(*addr));
#if defined(PETSC_HAVE_SHMGET)
  else {
    PetscShmgetAllocation next = allocations, previous = NULL;

    while (next) {
      if (next->addr == *addr) {
        PetscCall(PetscInfo(NULL, "Deallocating PCMPI Server array %p shmkey %d shmid %d size %d\n", *addr, next->shmkey, next->shmid, (int)next->sz));
        PetscCheck(!shmdt(next->addr), PETSC_COMM_SELF, PETSC_ERR_SYS, "Unable to shmdt() location %s, see PCMPIServerBegin()", strerror(errno));
        PetscCheck(!shmctl(next->shmid, IPC_RMID, NULL), PETSC_COMM_SELF, PETSC_ERR_SYS, "Unable to free shared memory addr %p key %d shmid %d %s, see PCMPIServerBegin()", *addr, next->shmkey, next->shmid, strerror(errno));
        *addr = NULL;
        if (previous) previous->next = next->next;
        else allocations = next->next;
        PetscCall(PetscFree(next));
        PetscFunctionReturn(PETSC_SUCCESS);
      }
      previous = next;
      next     = next->next;
    }
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unable to locate PCMPI allocated shared memory address %p", *addr);
  }
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(PETSC_USE_FORTRAN_BINDINGS)
  #include <petsc/private/ftnimpl.h>

  #if defined(PETSC_HAVE_FORTRAN_CAPS)
    #define petscshmgetallocatearrayscalar_   PETSCSHMGETALLOCATEARRAYSCALAR
    #define petscshmgetdeallocatearrayscalar_ PETSCSHMGETDEALLOCATEARRAYSCALAR
    #define petscshmgetallocatearrayint_      PETSCSHMGETALLOCATEARRAYINT
    #define petscshmgetdeallocatearrayint_    PETSCSHMGETDEALLOCATEARRAYINT
  #elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
    #define petscshmgetallocatearrayscalar_   petscshmgetallocatearrayscalar
    #define petscshmgetdeallocatearrayscalar_ petscshmgetdeallocatearrayscalar
    #define petscshmgetallocatearrayint_      petscshmgetallocatearrayint
    #define petscshmgetdeallocatearrayint_    petscshmgetdeallocatearrayint
  #endif

PETSC_EXTERN void petscshmgetallocatearrayscalar_(PetscInt *start, PetscInt *len, F90Array1d *a, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *aa;

  *ierr = PetscShmgetAllocateArray(*len, sizeof(PetscScalar), (void **)&aa);
  if (*ierr) return;
  *ierr = F90Array1dCreate(aa, MPIU_SCALAR, *start, *len, a PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void petscshmgetdeallocatearrayscalar_(F90Array1d *a, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *aa;

  *ierr = F90Array1dAccess(a, MPIU_SCALAR, (void **)&aa PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = PetscShmgetDeallocateArray((void **)&aa);
  if (*ierr) return;
  *ierr = F90Array1dDestroy(a, MPIU_SCALAR PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void petscshmgetallocatearrayint_(PetscInt *start, PetscInt *len, F90Array1d *a, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscInt *aa;

  *ierr = PetscShmgetAllocateArray(*len, sizeof(PetscInt), (void **)&aa);
  if (*ierr) return;
  *ierr = F90Array1dCreate(aa, MPIU_INT, *start, *len, a PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void petscshmgetdeallocatearrayint_(F90Array1d *a, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscInt *aa;

  *ierr = F90Array1dAccess(a, MPIU_INT, (void **)&aa PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = PetscShmgetDeallocateArray((void **)&aa);
  if (*ierr) return;
  *ierr = F90Array1dDestroy(a, MPIU_INT PETSC_F90_2PTR_PARAM(ptrd));
}

#endif
