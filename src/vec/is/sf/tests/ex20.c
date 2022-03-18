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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscMemTypeHost(mtype)) {
    #if defined(PETSC_HAVE_GETPAGESIZE)
      ierr = posix_memalign(ptr,getpagesize(),size);CHKERRQ(ierr);
    #else
      ierr = PetscMalloc(size,ptr);CHKERRQ(ierr);
    #endif
  }
#if defined(PETSC_HAVE_CUDA)
  else if (PetscMemTypeCUDA(mtype)) {cudaError_t cerr = cudaMalloc(ptr,size);CHKERRCUDA(cerr);}
#elif defined(PETSC_HAVE_HIP)
  else if (PetscMemTypeHIP(mtype))  {hipError_t cerr  = hipMalloc(ptr,size);CHKERRHIP(cerr);}
#endif
  PetscFunctionReturn(0);
}

static inline PetscErrorCode PetscFreeWithMemType_Private(PetscMemType mtype,void* ptr)
{
  PetscFunctionBegin;
  if (PetscMemTypeHost(mtype)) {free(ptr);}
#if defined(PETSC_HAVE_CUDA)
  else if (PetscMemTypeCUDA(mtype)) {cudaError_t cerr = cudaFree(ptr);CHKERRCUDA(cerr);}
#elif defined(PETSC_HAVE_HIP)
  else if (PetscMemTypeHIP(mtype))  {hipError_t cerr  = hipFree(ptr);CHKERRHIP(cerr);}
#endif
  PetscFunctionReturn(0);
}

/* Free memory and set ptr to NULL when succeeded */
#define PetscFreeWithMemType(t,p) ((p) && (PetscFreeWithMemType_Private((t),(p)) || ((p)=NULL,0)))

static inline PetscErrorCode PetscMemcpyFromHostWithMemType(PetscMemType mtype,void* dst, const void *src, size_t n)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscMemTypeHost(mtype)) {ierr = PetscMemcpy(dst,src,n);CHKERRQ(ierr);}
#if defined(PETSC_HAVE_CUDA)
  else if (PetscMemTypeCUDA(mtype)) {cudaError_t cerr = cudaMemcpy(dst,src,n,cudaMemcpyHostToDevice);CHKERRCUDA(cerr);}
#elif defined(PETSC_HAVE_HIP)
  else if (PetscMemTypeHIP(mtype))  {hipError_t cerr  = hipMemcpy(dst,src,n,hipMemcpyHostToDevice);CHKERRHIP(cerr);}
#endif
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscErrorCode    ierr;
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

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  /* Must init the device first if one wants to call PetscGetMemType() without creating PETSc device objects */
#if defined(PETSC_HAVE_CUDA)
  ierr = PetscDeviceInitialize(PETSC_DEVICE_CUDA);CHKERRQ(ierr);
#elif defined(PETSC_HAVE_HIP)
  ierr = PetscDeviceInitialize(PETSC_DEVICE_HIP);CHKERRQ(ierr);
#endif
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  PetscCheck(size == 2,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"Must run with 2 processes");

  ierr = PetscOptionsGetInt(NULL,NULL,"-maxn",&maxn,NULL);CHKERRQ(ierr); /* maxn PetscScalars */
  ierr = PetscOptionsGetInt(NULL,NULL,"-skipSmall",&skipSmall,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-loopSmall",&loopSmall,NULL);CHKERRQ(ierr);

  ierr = PetscMalloc1(maxn,&iremote);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-mtype",mstring,16,&set);CHKERRQ(ierr);
  if (set) {
    ierr = PetscStrcasecmp(mstring,"cuda",&isCuda);CHKERRQ(ierr);
    ierr = PetscStrcasecmp(mstring,"hip",&isHip);CHKERRQ(ierr);
    ierr = PetscStrcasecmp(mstring,"host",&isHost);CHKERRQ(ierr);

    if (isHost) mtype = PETSC_MEMTYPE_HOST;
    else if (isCuda) mtype = PETSC_MEMTYPE_CUDA;
    else if (isHip) mtype = PETSC_MEMTYPE_HIP;
    else SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Unknown memory type: %s",mstring);
  }

  ierr = PetscMallocWithMemType(mtype,sizeof(PetscScalar)*maxn,(void**)&rootdata);CHKERRQ(ierr);
  ierr = PetscMallocWithMemType(mtype,sizeof(PetscScalar)*maxn,(void**)&leafdata);CHKERRQ(ierr);

  ierr = PetscMalloc2(maxn,&pbuf,maxn,&ebuf);CHKERRQ(ierr);
  for (i=0; i<maxn; i++) {
    pbuf[i] = 123.0;
    ebuf[i] = 456.0;
  }

  for (n=1,i=0; n<=maxn; n*=2,i++) {
    ierr = PetscSFCreate(PETSC_COMM_WORLD,&sf[i]);CHKERRQ(ierr);
    ierr = PetscSFSetFromOptions(sf[i]);CHKERRQ(ierr);
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
    ierr = PetscSFSetGraph(sf[i],nroots,nleaves,NULL,PETSC_COPY_VALUES,iremote,PETSC_COPY_VALUES);CHKERRQ(ierr);
    ierr = PetscSFSetUp(sf[i]);CHKERRQ(ierr);
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
    ierr = PetscMemcpyFromHostWithMemType(mtype,rootdata,pbuf,msgsize);CHKERRQ(ierr);
    ierr = PetscMemcpyFromHostWithMemType(mtype,leafdata,ebuf,msgsize);CHKERRQ(ierr);

    if (msgsize > LARGE_MESSAGE_SIZE) {
      nskip = LAT_SKIP_LARGE;
      niter = LAT_LOOP_LARGE;
    }
    ierr = MPI_Barrier(MPI_COMM_WORLD);CHKERRMPI(ierr);

    for (i=0; i<niter + nskip; i++) {
      if (i == nskip) {
       #if defined(PETSC_HAVE_CUDA)
        {cudaError_t cerr = cudaDeviceSynchronize();CHKERRCUDA(cerr);}
       #elif defined(PETSC_HAVE_HIP)
        {hipError_t  cerr = hipDeviceSynchronize();CHKERRHIP(cerr);}
       #endif
        ierr    = MPI_Barrier(PETSC_COMM_WORLD);CHKERRMPI(ierr);
        t_start = MPI_Wtime();
      }
      ierr = PetscSFBcastWithMemTypeBegin(sf[j],MPIU_SCALAR,mtype,rootdata,mtype,leafdata,op);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(sf[j],MPIU_SCALAR,rootdata,leafdata,op);CHKERRQ(ierr);
      ierr = PetscSFReduceWithMemTypeBegin(sf[j],MPIU_SCALAR,mtype,leafdata,mtype,rootdata,op);CHKERRQ(ierr);
      ierr = PetscSFReduceEnd(sf[j],MPIU_SCALAR,leafdata,rootdata,op);CHKERRQ(ierr);
    }
   #if defined(PETSC_HAVE_CUDA)
    {cudaError_t cerr = cudaDeviceSynchronize();CHKERRCUDA(cerr);}
   #elif defined(PETSC_HAVE_HIP)
    {hipError_t  cerr = hipDeviceSynchronize();CHKERRHIP(cerr);}
   #endif
    ierr    = MPI_Barrier(PETSC_COMM_WORLD);CHKERRMPI(ierr);
    t_end   = MPI_Wtime();
    time[j] = (t_end - t_start)*1e6 / (niter*2);
  }

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\t##  PetscSF Ping-pong test on %s ##\n  Message(Bytes) \t\tLatency(us)\n", mtype==PETSC_MEMTYPE_HOST? "Host" : "Device");CHKERRQ(ierr);
  for (n=1,j=0; n<=maxn; n*=2,j++) {
    ierr = PetscSFDestroy(&sf[j]);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"%16D \t %16.4f\n",sizeof(PetscScalar)*n,time[j]);CHKERRQ(ierr);
  }

  ierr = PetscFree2(pbuf,ebuf);CHKERRQ(ierr);
  ierr = PetscFreeWithMemType(mtype,rootdata);CHKERRQ(ierr);
  ierr = PetscFreeWithMemType(mtype,leafdata);CHKERRQ(ierr);
  ierr = PetscFree(iremote);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
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
