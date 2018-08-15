#include <petscsys.h>        /*I  "petscsys.h"  I*/
#include <petsc/private/petscimpl.h>

struct _n_PetscShmComm {
  PetscMPIInt *globranks;       /* global ranks of each rank in the shared memory communicator */
  PetscMPIInt shmsize;          /* size of the shared memory communicator */
  MPI_Comm    globcomm,shmcomm; /* global communicator and shared memory communicator (a sub-communicator of the former) */
};

/*
   Private routine to delete internal tag/name shared memory communicator when a communicator is freed.

   This is called by MPI, not by users. This is called by MPI_Comm_free() when the communicator that has this  data as an attribute is freed.

   Note: this is declared extern "C" because it is passed to MPI_Comm_create_keyval()

*/
PETSC_EXTERN PetscMPIInt MPIAPI Petsc_DelComm_Shm(MPI_Comm comm,PetscMPIInt keyval,void *val,void *extra_state)
{
  PetscErrorCode  ierr;
  PetscShmComm p = (PetscShmComm)val;

  PetscFunctionBegin;
  ierr = PetscInfo1(0,"Deleting shared memory subcommunicator in a MPI_Comm %ld\n",(long)comm);CHKERRMPI(ierr);
  ierr = MPI_Comm_free(&p->shmcomm);CHKERRMPI(ierr);
  ierr = PetscFree(p->globranks);CHKERRMPI(ierr);
  ierr = PetscFree(val);CHKERRMPI(ierr);
  PetscFunctionReturn(MPI_SUCCESS);
}

/*@C
    PetscShmCommGet - Given a PETSc communicator returns a communicator of all ranks that share a common memory


    Collective on comm.

    Input Parameter:
.   globcomm - MPI_Comm

    Output Parameter:
.   pshmcomm - the PETSc shared memory communicator object

    Level: developer

    Notes:
    This should be called only with an PetscCommDuplicate() communictor

           When used with MPICH, MPICH must be configured with --download-mpich-device=ch3:nemesis

    Concepts: MPI subcomm^numbering

@*/
PetscErrorCode PetscShmCommGet(MPI_Comm globcomm,PetscShmComm *pshmcomm)
{
#ifdef PETSC_HAVE_MPI_PROCESS_SHARED_MEMORY
  PetscErrorCode   ierr;
  MPI_Group        globgroup,shmgroup;
  PetscMPIInt      *shmranks,i,flg;
  PetscCommCounter *counter;

  PetscFunctionBegin;
  ierr = MPI_Comm_get_attr(globcomm,Petsc_Counter_keyval,&counter,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(globcomm,PETSC_ERR_ARG_CORRUPT,"Bad MPI communicator supplied; must be a PETSc communicator");

  ierr = MPI_Comm_get_attr(globcomm,Petsc_ShmComm_keyval,pshmcomm,&flg);CHKERRQ(ierr);
  if (flg) PetscFunctionReturn(0);

  ierr        = PetscNew(pshmcomm);CHKERRQ(ierr);
  (*pshmcomm)->globcomm = globcomm;

  ierr = MPI_Comm_split_type(globcomm, MPI_COMM_TYPE_SHARED,0, MPI_INFO_NULL,&(*pshmcomm)->shmcomm);CHKERRQ(ierr);

  ierr = MPI_Comm_size((*pshmcomm)->shmcomm,&(*pshmcomm)->shmsize);CHKERRQ(ierr);
  ierr = MPI_Comm_group(globcomm, &globgroup);CHKERRQ(ierr);
  ierr = MPI_Comm_group((*pshmcomm)->shmcomm, &shmgroup);CHKERRQ(ierr);
  ierr = PetscMalloc1((*pshmcomm)->shmsize,&shmranks);CHKERRQ(ierr);
  ierr = PetscMalloc1((*pshmcomm)->shmsize,&(*pshmcomm)->globranks);CHKERRQ(ierr);
  for (i=0; i<(*pshmcomm)->shmsize; i++) shmranks[i] = i;
  ierr = MPI_Group_translate_ranks(shmgroup, (*pshmcomm)->shmsize, shmranks, globgroup, (*pshmcomm)->globranks);CHKERRQ(ierr);
  ierr = PetscFree(shmranks);CHKERRQ(ierr);
  ierr = MPI_Group_free(&globgroup);CHKERRQ(ierr);
  ierr = MPI_Group_free(&shmgroup);CHKERRQ(ierr);

  for (i=0; i<(*pshmcomm)->shmsize; i++) {
    ierr = PetscInfo2(NULL,"Shared memory rank %d global rank %d\n",i,(*pshmcomm)->globranks[i]);CHKERRQ(ierr);
  }
  ierr = MPI_Comm_set_attr(globcomm,Petsc_ShmComm_keyval,*pshmcomm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#else
  SETERRQ(globcomm, PETSC_ERR_SUP, "Shared memory communicators need MPI-3 package support.\nPlease upgrade your MPI or reconfigure with --download-mpich.");
#endif
}

/*@C
    PetscShmCommGlobalToLocal - Given a global rank returns the local rank in the shared memory communicator

    Input Parameters:
+   pshmcomm - the shared memory communicator object
-   grank    - the global rank

    Output Parameter:
.   lrank - the local rank, or MPI_PROC_NULL if it does not exist

    Level: developer

    Developer Notes:
    Assumes the pshmcomm->globranks[] is sorted

    It may be better to rewrite this to map multiple global ranks to local in the same function call

    Concepts: MPI subcomm^numbering

@*/
PetscErrorCode PetscShmCommGlobalToLocal(PetscShmComm pshmcomm,PetscMPIInt grank,PetscMPIInt *lrank)
{
  PetscMPIInt    low,high,t,i;
  PetscBool      flg = PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *lrank = MPI_PROC_NULL;
  if (grank < pshmcomm->globranks[0]) PetscFunctionReturn(0);
  if (grank > pshmcomm->globranks[pshmcomm->shmsize-1]) PetscFunctionReturn(0);
  ierr = PetscOptionsGetBool(NULL,NULL,"-noshared",&flg,NULL);CHKERRQ(ierr);
  if (flg) PetscFunctionReturn(0);
  low  = 0;
  high = pshmcomm->shmsize;
  while (high-low > 5) {
    t = (low+high)/2;
    if (pshmcomm->globranks[t] > grank) high = t;
    else low = t;
  }
  for (i=low; i<high; i++) {
    if (pshmcomm->globranks[i] > grank) PetscFunctionReturn(0);
    if (pshmcomm->globranks[i] == grank) {
      *lrank = i;
      PetscFunctionReturn(0);
    }
  }
  PetscFunctionReturn(0);
}

/*@C
    PetscShmCommLocalToGlobal - Given a local rank in the shared memory communicator returns the global rank

    Input Parameters:
+   pshmcomm - the shared memory communicator object
-   lrank    - the local rank in the shared memory communicator

    Output Parameter:
.   grank - the global rank in the global communicator where the shared memory communicator is built

    Level: developer

    Concepts: MPI subcomm^numbering
@*/
PetscErrorCode PetscShmCommLocalToGlobal(PetscShmComm pshmcomm,PetscMPIInt lrank,PetscMPIInt *grank)
{
  PetscFunctionBegin;
#ifdef PETSC_USE_DEBUG
  {
    PetscErrorCode ierr;
    if (lrank < 0 || lrank >= pshmcomm->shmsize) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"No rank %D in the shared memory communicator",lrank);CHKERRQ(ierr); }
  }
#endif
  *grank = pshmcomm->globranks[lrank];
  PetscFunctionReturn(0);
}

/*@C
    PetscShmCommGetMpiShmComm - Returns the MPI communicator that represents all processes with common shared memory

    Input Parameter:
.   pshmcomm - PetscShmComm object obtained with PetscShmCommGet()

    Output Parameter:
.   comm     - the MPI communicator

    Level: developer

@*/
PetscErrorCode PetscShmCommGetMpiShmComm(PetscShmComm pshmcomm,MPI_Comm *comm)
{
  PetscFunctionBegin;
  *comm = pshmcomm->shmcomm;
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_OPENMP) && defined(PETSC_HAVE_PTHREAD) && defined(PETSC_HAVE_MPI_PROCESS_SHARED_MEMORY) && defined(PETSC_HAVE_HWLOC)
#include <pthread.h>
#include <hwloc.h>
#include <omp.h>

struct _n_PetscOmpCtrl {
  MPI_Comm          omp_comm;        /* a shared memory communicator to spawn omp threads */
  MPI_Comm          omp_master_comm; /* a communicator to give to third party libraries */
  PetscMPIInt       omp_comm_size;   /* size of omp_comm, a kind of OMP_NUM_THREADS */
  PetscBool         is_omp_master;   /* rank 0's in omp_comm */
  MPI_Win           omp_win;         /* a shared memory window containing a barrier */
  pthread_barrier_t *barrier;        /* pointer to the barrier */
  hwloc_topology_t  topology;
  hwloc_cpuset_t    cpuset;          /* cpu bindings of omp master */
  hwloc_cpuset_t    omp_cpuset;      /* union of cpu bindings of ranks in omp_comm */
};

/* Allocate a shared pthread_barrier_t object in ctrl->omp_comm, set ctrl->barrier */
PETSC_STATIC_INLINE PetscErrorCode PetscOmpCtrlCreateBarrier(PetscOmpCtrl ctrl)
{
  PetscErrorCode        ierr;
  MPI_Aint              size;
  PetscMPIInt           disp_unit;
  void                  *baseptr;
  pthread_barrierattr_t attr;

  size = ctrl->is_omp_master ? sizeof(pthread_barrier_t) : 0;
  ierr = MPI_Win_allocate_shared(size,1,MPI_INFO_NULL,ctrl->omp_comm,&baseptr,&ctrl->omp_win);CHKERRQ(ierr);
  ierr = MPI_Win_shared_query(ctrl->omp_win,0,&size,&disp_unit,&baseptr);CHKERRQ(ierr);
  ctrl->barrier = (pthread_barrier_t*)baseptr;

  /* omp master initializes the barrier */
  if (ctrl->is_omp_master) {
    ierr = MPI_Comm_size(ctrl->omp_comm,&ctrl->omp_comm_size);CHKERRQ(ierr);
    ierr = pthread_barrierattr_init(&attr);CHKERRQ(ierr);
    ierr = pthread_barrierattr_setpshared(&attr,PTHREAD_PROCESS_SHARED);CHKERRQ(ierr); /* make the barrier also work for processes */
    ierr = pthread_barrier_init(ctrl->barrier,&attr,(unsigned int)ctrl->omp_comm_size);CHKERRQ(ierr);
    ierr = pthread_barrierattr_destroy(&attr);CHKERRQ(ierr);
  }

  /* the MPI_Barrier is to make sure the omp barrier is initialized before slaves use it */
  MPI_Barrier(ctrl->omp_comm);
  PetscFunctionReturn(0);
}

/* Destroy ctrl->barrier */
PETSC_STATIC_INLINE PetscErrorCode PetscOmpCtrlDestroyBarrier(PetscOmpCtrl ctrl)
{
  PetscErrorCode ierr;

  /* the MPI_Barrier is to make sure slaves have finished using the omp barrier before master destroys it */
  ierr = MPI_Barrier(ctrl->omp_comm);CHKERRQ(ierr);
  if (ctrl->is_omp_master) { ierr = pthread_barrier_destroy(ctrl->barrier);CHKERRQ(ierr); }
  ierr = MPI_Win_free(&ctrl->omp_win);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* create a PETSc OpenMP controler, which manages PETSc's interaction with OpenMP runtime */
PetscErrorCode PetscOmpCtrlCreate(MPI_Comm petsc_comm,PetscInt nthreads,PetscOmpCtrl *pctrl)
{
  PetscErrorCode        ierr;
  PetscOmpCtrl          ctrl;
  unsigned long         *cpu_ulongs=NULL;
  PetscInt              i,nr_cpu_ulongs;
  PetscShmComm          pshmcomm;
  MPI_Comm              shm_comm;
  PetscMPIInt           shm_rank,shm_comm_size,omp_rank,color;

  PetscFunctionBegin;
  ierr = PetscNew(&ctrl);CHKERRQ(ierr);

  /*=================================================================================
    Split petsc_comm into multiple omp_comms. Ranks in an omp_comm have access to
    physically shared memory. Rank 0 of each omp_comm is called an OMP master, and
    others are called slaves. OMP Masters make up a new comm called omp_master_comm,
    which is usually passed to third party libraries.
   ==================================================================================*/

  /* fetch the stored shared memory communicator */
  ierr = PetscShmCommGet(petsc_comm,&pshmcomm);CHKERRQ(ierr);
  ierr = PetscShmCommGetMpiShmComm(pshmcomm,&shm_comm);CHKERRQ(ierr);

  ierr = MPI_Comm_rank(shm_comm,&shm_rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(shm_comm,&shm_comm_size);CHKERRQ(ierr);

  if (nthreads < 1 || nthreads > shm_comm_size) SETERRQ2(petsc_comm,PETSC_ERR_ARG_OUTOFRANGE,"number of OpenMP threads %d can not be < 1 or > the MPI shared memory communicator size %d\n",nthreads,shm_comm_size);
  if (shm_comm_size % nthreads) { ierr = PetscPrintf(petsc_comm,"Warning: number of OpenMP threads %d is not a factor of the MPI shared memory communicator size %d, which may cause load-imbalance!\n",nthreads,shm_comm_size);CHKERRQ(ierr); }

  /* split shm_comm into a set of omp_comms with each of size nthreads. Ex., if
     shm_comm_size=16, nthreads=8, then ranks 0~7 get color 0 and ranks 8~15 get
     color 1. They are put in two omp_comms. Note that petsc_ranks may or may not
     be consecutive in a shm_comm, but shm_ranks always run from 0 to shm_comm_size-1.
     Use 0 as key so that rank ordering wont change in new comm.
   */
  color = shm_rank / nthreads;
  ierr  = MPI_Comm_split(shm_comm,color,0/*key*/,&ctrl->omp_comm);CHKERRQ(ierr);

  /* put rank 0's in omp_comms (i.e., master ranks) into a new comm - omp_master_comm */
  ierr = MPI_Comm_rank(ctrl->omp_comm,&omp_rank);CHKERRQ(ierr);
  if (!omp_rank) {
    ctrl->is_omp_master = PETSC_TRUE;  /* master */
    color = 0;
  } else {
    ctrl->is_omp_master = PETSC_FALSE; /* slave */
    color = MPI_UNDEFINED; /* to make slaves get omp_master_comm = MPI_COMM_NULL in MPI_Comm_split */
  }
  ierr = MPI_Comm_split(petsc_comm,color,0/*key*/,&ctrl->omp_master_comm);CHKERRQ(ierr); /* rank 0 in omp_master_comm is rank 0 in petsc_comm */

  /*=================================================================================
    Each omp_comm has a pthread_barrier_t in its shared memory, which is used to put
    slave ranks in sleep and idle their CPU, so that the master can fork OMP threads
    and run them on the idle CPUs.
   ==================================================================================*/
  ierr = PetscOmpCtrlCreateBarrier(ctrl);CHKERRQ(ierr);

  /*=================================================================================
    omp master logs its cpu binding (i.e., cpu set) and computes a new binding that
    is the union of the bindings of all ranks in the omp_comm
    =================================================================================*/
  ierr = hwloc_topology_init(&ctrl->topology);CHKERRQ(ierr);
#if HWLOC_API_VERSION >= 0x00020000
  /* to filter out unneeded info and have faster hwloc_topology_load */
  ierr = hwloc_topology_set_all_types_filter(ctrl->topology,HWLOC_TYPE_FILTER_KEEP_NONE);CHKERRQ(ierr);
  ierr = hwloc_topology_set_type_filter(ctrl->topology,HWLOC_OBJ_CORE,HWLOC_TYPE_FILTER_KEEP_ALL);CHKERRQ(ierr);
#endif
  ierr = hwloc_topology_load(ctrl->topology);CHKERRQ(ierr);

  ctrl->cpuset = hwloc_bitmap_alloc(); if (!ctrl->cpuset) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"hwloc_bitmap_alloc() failed\n");
  ierr = hwloc_get_cpubind(ctrl->topology,ctrl->cpuset, HWLOC_CPUBIND_PROCESS);CHKERRQ(ierr);

  /* hwloc main developer said they will add new APIs hwloc_bitmap_{nr,to,from}_ulongs in 2.1 to help us simplify the following bitmap pack/unpack code */
  nr_cpu_ulongs = (hwloc_bitmap_last(hwloc_topology_get_topology_cpuset (ctrl->topology))+sizeof(unsigned long)*8)/sizeof(unsigned long)/8;
  ierr = PetscMalloc1(nr_cpu_ulongs,&cpu_ulongs);CHKERRQ(ierr);
  if (nr_cpu_ulongs == 1) {
    cpu_ulongs[0] = hwloc_bitmap_to_ulong(ctrl->cpuset);
  } else {
    for (i=0; i<nr_cpu_ulongs; i++) cpu_ulongs[i] = hwloc_bitmap_to_ith_ulong(ctrl->cpuset,(unsigned)i);
  }

  ierr = MPI_Reduce(ctrl->is_omp_master ? MPI_IN_PLACE : cpu_ulongs, cpu_ulongs,nr_cpu_ulongs, MPI_UNSIGNED_LONG,MPI_BOR,0,ctrl->omp_comm);CHKERRQ(ierr);

  if (ctrl->is_omp_master) {
    ctrl->omp_cpuset = hwloc_bitmap_alloc(); if (!ctrl->omp_cpuset) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"hwloc_bitmap_alloc() failed\n");
    if (nr_cpu_ulongs == 1) {
#if HWLOC_API_VERSION >= 0x00020000
      ierr = hwloc_bitmap_from_ulong(ctrl->omp_cpuset,cpu_ulongs[0]);CHKERRQ(ierr);
#else
      hwloc_bitmap_from_ulong(ctrl->omp_cpuset,cpu_ulongs[0]);
#endif
    } else {
      for (i=0; i<nr_cpu_ulongs; i++)  {
#if HWLOC_API_VERSION >= 0x00020000
        ierr = hwloc_bitmap_set_ith_ulong(ctrl->omp_cpuset,(unsigned)i,cpu_ulongs[i]);CHKERRQ(ierr);
#else
        hwloc_bitmap_set_ith_ulong(ctrl->omp_cpuset,(unsigned)i,cpu_ulongs[i]);
#endif
      }
    }
  }

  /* all wait for the master to finish the initialization before using the barrier */
  ierr = MPI_Barrier(ctrl->omp_comm);CHKERRQ(ierr);
  ierr = PetscFree(cpu_ulongs);CHKERRQ(ierr);
  *pctrl = ctrl;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscOmpCtrlDestroy(PetscOmpCtrl *pctrl)
{
  PetscErrorCode  ierr;
  PetscOmpCtrl    ctrl = *pctrl;

  PetscFunctionBegin;
  hwloc_bitmap_free(ctrl->cpuset);
  hwloc_topology_destroy(ctrl->topology);
  PetscOmpCtrlDestroyBarrier(ctrl);
  ierr = MPI_Comm_free(&ctrl->omp_comm);CHKERRQ(ierr);
  if (ctrl->is_omp_master) {
    hwloc_bitmap_free(ctrl->omp_cpuset);
    ierr = MPI_Comm_free(&ctrl->omp_master_comm);CHKERRQ(ierr);
  }
  ierr = PetscFree(ctrl);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
    PetscOmpCtrlGetOmpComms - Get MPI communicators from a PetscOmpCtrl

    Input Parameter:
.   ctrl - a PetscOmpCtrl

    Output Parameter:
+   omp_comm         - a communicator that includes a master rank and slave ranks.
.   omp_master_comm  - on master ranks, return a communicator that include master ranks of each omp_comm;
                       on slave ranks, MPI_COMM_NULL will be return in reality.
-   is_omp_master    - true if the calling process is an OMP master rank.

    Level: developer
@*/
PetscErrorCode PetscOmpCtrlGetOmpComms(PetscOmpCtrl ctrl,MPI_Comm *omp_comm,MPI_Comm *omp_master_comm,PetscBool *is_omp_master)
{
  PetscFunctionBegin;
  if (omp_comm)        *omp_comm        = ctrl->omp_comm;
  if (omp_master_comm) *omp_master_comm = ctrl->omp_master_comm;
  if (is_omp_master)   *is_omp_master   = ctrl->is_omp_master;
  PetscFunctionReturn(0);
}

/* a barrier in the scope of an omp_comm. Not using MPI_Barrier since it keeps polling and does not free CPUs OMP wants to use */
PetscErrorCode PetscOmpCtrlBarrier(PetscOmpCtrl ctrl)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = pthread_barrier_wait(ctrl->barrier);
  if (ierr && ierr != PTHREAD_BARRIER_SERIAL_THREAD) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"pthread_barrier_wait failed within PetscOmpCtrlBarrier with return code %D\n", ierr);
  PetscFunctionReturn(0);
}

/* call this on master ranks before calling a library using OpenMP */
PetscErrorCode PetscOmpCtrlOmpRegionOnMasterBegin(PetscOmpCtrl ctrl)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = hwloc_set_cpubind(ctrl->topology,ctrl->omp_cpuset,HWLOC_CPUBIND_PROCESS);CHKERRQ(ierr);
  omp_set_num_threads(ctrl->omp_comm_size); /* may override OMP_NUM_THREAD in environment */
  PetscFunctionReturn(0);
}

/* call this on master ranks after leaving a library using OpenMP */
PetscErrorCode PetscOmpCtrlOmpRegionOnMasterEnd(PetscOmpCtrl ctrl)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = hwloc_set_cpubind(ctrl->topology,ctrl->cpuset,HWLOC_CPUBIND_PROCESS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#endif /* defined(PETSC_HAVE_PTHREAD) && .. */
