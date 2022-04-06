static const char help[] = "PetscSF Ping-pong test to measure MPI latency\n\n";

/*
  This is a simple test to measure the latency of MPI communication.
  The test is run with two processes.  The first process sends a message
  to the second process, and after having received the message, the second
  process sends a message back to the first process once.  The is repeated
  a number of times.  The latency is defined as half time of the round-trip.

  It mimics osu_latency from the OSU microbenchmarks (https://mvapich.cse.ohio-state.edu/benchmarks/).

  Usage: mpirun -n 2 ./ex18 -mtype <type>
  Other arguments have a default value that is also used in osu_latency.

  Examples:

  On Summit at OLCF:
    jsrun --smpiargs "-gpu" -n 2 -a 1 -c 7 -g 1 -r 2 -l GPU-GPU -d packed -b packed:7 ./ex18  -mtype cuda

  On Crusher at OLCF:
    srun -n2 -c32 --cpu-bind=map_cpu:0,1 --gpus-per-node=8 --gpu-bind=map_gpu:0,1 ./ex18 -mtype hip
*/

#include <petscsf.h>
#include <petscdevice.h>
#if defined(PETSC_HAVE_UNISTD_H)
#include <unistd.h>
#endif

/* Same values as OSU microbenchmarks */
#define LAT_LOOP_SMALL 10000
#define LAT_SKIP_SMALL 100
#define LAT_LOOP_LARGE 1000
#define LAT_SKIP_LARGE 10
#define LARGE_MESSAGE_SIZE 8192

static inline PetscErrorCode PetscMallocWithMemType(PetscMemType mtype,size_t size,void** ptr)
{
  PetscFunctionBegin;
  if (PetscMemTypeHost(mtype)) {
    #if defined(PETSC_HAVE_GETPAGESIZE)
      PetscCall(posix_memalign(ptr,getpagesize(),size));
    #else
      PetscCall(PetscMalloc(size,ptr));
    #endif
  }
#if defined(PETSC_HAVE_CUDA)
  else if (PetscMemTypeCUDA(mtype)) PetscCallCUDA(cudaMalloc(ptr,size));
#elif defined(PETSC_HAVE_HIP)
  else if (PetscMemTypeHIP(mtype))  PetscCallHIP(hipMalloc(ptr,size));
#endif
  PetscFunctionReturn(0);
}

static inline PetscErrorCode PetscFreeWithMemType_Private(PetscMemType mtype,void* ptr)
{
  PetscFunctionBegin;
  if (PetscMemTypeHost(mtype)) {free(ptr);}
#if defined(PETSC_HAVE_CUDA)
  else if (PetscMemTypeCUDA(mtype)) PetscCallCUDA(cudaFree(ptr));
#elif defined(PETSC_HAVE_HIP)
  else if (PetscMemTypeHIP(mtype))  PetscCallHIP(hipFree(ptr));
#endif
  PetscFunctionReturn(0);
}

/* Free memory and set ptr to NULL when succeeded */
#define PetscFreeWithMemType(t,p) ((p) && (PetscFreeWithMemType_Private((t),(p)) || ((p)=NULL,0)))

static inline PetscErrorCode PetscMemcpyFromHostWithMemType(PetscMemType mtype,void* dst, const void *src, size_t n)
{
  PetscFunctionBegin;
  if (PetscMemTypeHost(mtype)) PetscCall(PetscMemcpy(dst,src,n));
#if defined(PETSC_HAVE_CUDA)
  else if (PetscMemTypeCUDA(mtype)) PetscCallCUDA(cudaMemcpy(dst,src,n,cudaMemcpyHostToDevice));
#elif defined(PETSC_HAVE_HIP)
  else if (PetscMemTypeHIP(mtype))  PetscCallHIP(hipMemcpy(dst,src,n,hipMemcpyHostToDevice));
#endif
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscSF           sf[64];
  PetscLogDouble    t_start=0,t_end=0,time[64];
  PetscInt          i,j,n,nroots,nleaves,niter=100,nskip=10;
  PetscInt          maxn=512*1024; /* max 4M bytes messages */
  PetscSFNode       *iremote;
  PetscMPIInt       rank,size;
  PetscScalar       *rootdata=NULL,*leafdata=NULL,*pbuf,*ebuf;
  size_t            msgsize;
  PetscMemType      mtype = PETSC_MEMTYPE_HOST;
  char              mstring[16]={0};
  PetscBool         isCuda,isHip,isHost,set;
  PetscInt          skipSmall=-1,loopSmall=-1;
  MPI_Op            op = MPI_REPLACE;

  PetscCall(PetscInitialize(&argc,&argv,NULL,help));
  /* Must init the device first if one wants to call PetscGetMemType() without creating PETSc device objects */
#if defined(PETSC_HAVE_CUDA)
  PetscCall(PetscDeviceInitialize(PETSC_DEVICE_CUDA));
#elif defined(PETSC_HAVE_HIP)
  PetscCall(PetscDeviceInitialize(PETSC_DEVICE_HIP));
#endif
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCheck(size == 2,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"Must run with 2 processes");

  PetscCall(PetscOptionsGetInt(NULL,NULL,"-maxn",&maxn,NULL)); /* maxn PetscScalars */
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-skipSmall",&skipSmall,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-loopSmall",&loopSmall,NULL));

  PetscCall(PetscMalloc1(maxn,&iremote));
  PetscCall(PetscOptionsGetString(NULL,NULL,"-mtype",mstring,16,&set));
  if (set) {
    PetscCall(PetscStrcasecmp(mstring,"cuda",&isCuda));
    PetscCall(PetscStrcasecmp(mstring,"hip",&isHip));
    PetscCall(PetscStrcasecmp(mstring,"host",&isHost));

    if (isHost) mtype = PETSC_MEMTYPE_HOST;
    else if (isCuda) mtype = PETSC_MEMTYPE_CUDA;
    else if (isHip) mtype = PETSC_MEMTYPE_HIP;
    else SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Unknown memory type: %s",mstring);
  }

  PetscCall(PetscMallocWithMemType(mtype,sizeof(PetscScalar)*maxn,(void**)&rootdata));
  PetscCall(PetscMallocWithMemType(mtype,sizeof(PetscScalar)*maxn,(void**)&leafdata));

  PetscCall(PetscMalloc2(maxn,&pbuf,maxn,&ebuf));
  for (i=0; i<maxn; i++) {
    pbuf[i] = 123.0;
    ebuf[i] = 456.0;
  }

  for (n=1,i=0; n<=maxn; n*=2,i++) {
    PetscCall(PetscSFCreate(PETSC_COMM_WORLD,&sf[i]));
    PetscCall(PetscSFSetFromOptions(sf[i]));
    if (!rank) {
      nroots  = n;
      nleaves = 0;
    } else {
      nroots  = 0;
      nleaves = n;
      for (j=0; j<nleaves; j++) {
        iremote[j].rank  = 0;
        iremote[j].index = j;
      }
    }
    PetscCall(PetscSFSetGraph(sf[i],nroots,nleaves,NULL,PETSC_COPY_VALUES,iremote,PETSC_COPY_VALUES));
    PetscCall(PetscSFSetUp(sf[i]));
  }

  if (loopSmall > 0) {
    nskip = skipSmall;
    niter = loopSmall;
  } else {
    nskip = LAT_SKIP_SMALL;
    niter = LAT_LOOP_SMALL;
  }

  for (n=1,j=0; n<=maxn; n*=2,j++) {
    msgsize = sizeof(PetscScalar)*n;
    PetscCall(PetscMemcpyFromHostWithMemType(mtype,rootdata,pbuf,msgsize));
    PetscCall(PetscMemcpyFromHostWithMemType(mtype,leafdata,ebuf,msgsize));

    if (msgsize > LARGE_MESSAGE_SIZE) {
      nskip = LAT_SKIP_LARGE;
      niter = LAT_LOOP_LARGE;
    }
    PetscCallMPI(MPI_Barrier(MPI_COMM_WORLD));

    for (i=0; i<niter + nskip; i++) {
      if (i == nskip) {
       #if defined(PETSC_HAVE_CUDA)
        PetscCallCUDA(cudaDeviceSynchronize());
       #elif defined(PETSC_HAVE_HIP)
        PetscCallHIP(hipDeviceSynchronize());
       #endif
        PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
        t_start = MPI_Wtime();
      }
      PetscCall(PetscSFBcastWithMemTypeBegin(sf[j],MPIU_SCALAR,mtype,rootdata,mtype,leafdata,op));
      PetscCall(PetscSFBcastEnd(sf[j],MPIU_SCALAR,rootdata,leafdata,op));
      PetscCall(PetscSFReduceWithMemTypeBegin(sf[j],MPIU_SCALAR,mtype,leafdata,mtype,rootdata,op));
      PetscCall(PetscSFReduceEnd(sf[j],MPIU_SCALAR,leafdata,rootdata,op));
    }
   #if defined(PETSC_HAVE_CUDA)
    PetscCallCUDA(cudaDeviceSynchronize());
   #elif defined(PETSC_HAVE_HIP)
    PetscCallHIP(hipDeviceSynchronize());
   #endif
    PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
    t_end   = MPI_Wtime();
    time[j] = (t_end - t_start)*1e6 / (niter*2);
  }

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\t##  PetscSF Ping-pong test on %s ##\n  Message(Bytes) \t\tLatency(us)\n", mtype==PETSC_MEMTYPE_HOST? "Host" : "Device"));
  for (n=1,j=0; n<=maxn; n*=2,j++) {
    PetscCall(PetscSFDestroy(&sf[j]));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"%16" PetscInt_FMT " \t %16.4f\n",sizeof(PetscScalar)*n,time[j]));
  }

  PetscCall(PetscFree2(pbuf,ebuf));
  PetscCall(PetscFreeWithMemType(mtype,rootdata));
  PetscCall(PetscFreeWithMemType(mtype,leafdata));
  PetscCall(PetscFree(iremote));
  PetscCall(PetscFinalize());
  return 0;
}

/**TEST
  testset:
    # use small numbers to make the test cheap
    args: -maxn 4 -skipSmall 1 -loopSmall 1
    filter: grep "DOES_NOT_EXIST"
    output_file: output/empty.out
    nsize: 2

    test:
      args: -mtype host

    test:
      requires: cuda
      args: -mtype cuda

    test:
      requires: hip
      args: -mtype hip
TEST**/
