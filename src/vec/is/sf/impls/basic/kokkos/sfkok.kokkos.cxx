#include <../src/vec/is/sf/impls/basic/sfpack.h>

#include <Kokkos_Core.hpp>

using DeviceExecutionSpace = Kokkos::DefaultExecutionSpace;
using DeviceMemorySpace    = typename DeviceExecutionSpace::memory_space;
using HostMemorySpace      = Kokkos::HostSpace;

typedef Kokkos::View<char*,DeviceMemorySpace>       deviceBuffer_t;
typedef Kokkos::View<char*,HostMemorySpace>         HostBuffer_t;

typedef Kokkos::View<const char*,DeviceMemorySpace> deviceConstBuffer_t;
typedef Kokkos::View<const char*,HostMemorySpace>   HostConstBuffer_t;

/*====================================================================================*/
/*                             Regular operations                           */
/*====================================================================================*/
template<typename Type> struct Insert{KOKKOS_INLINE_FUNCTION Type operator()(Type& x,Type y) const {Type old = x; x  = y;             return old;}};
template<typename Type> struct Add   {KOKKOS_INLINE_FUNCTION Type operator()(Type& x,Type y) const {Type old = x; x += y;             return old;}};
template<typename Type> struct Mult  {KOKKOS_INLINE_FUNCTION Type operator()(Type& x,Type y) const {Type old = x; x *= y;             return old;}};
template<typename Type> struct Min   {KOKKOS_INLINE_FUNCTION Type operator()(Type& x,Type y) const {Type old = x; x  = PetscMin(x,y); return old;}};
template<typename Type> struct Max   {KOKKOS_INLINE_FUNCTION Type operator()(Type& x,Type y) const {Type old = x; x  = PetscMax(x,y); return old;}};
template<typename Type> struct LAND  {KOKKOS_INLINE_FUNCTION Type operator()(Type& x,Type y) const {Type old = x; x  = x && y;        return old;}};
template<typename Type> struct LOR   {KOKKOS_INLINE_FUNCTION Type operator()(Type& x,Type y) const {Type old = x; x  = x || y;        return old;}};
template<typename Type> struct LXOR  {KOKKOS_INLINE_FUNCTION Type operator()(Type& x,Type y) const {Type old = x; x  = !x != !y;      return old;}};
template<typename Type> struct BAND  {KOKKOS_INLINE_FUNCTION Type operator()(Type& x,Type y) const {Type old = x; x  = x & y;         return old;}};
template<typename Type> struct BOR   {KOKKOS_INLINE_FUNCTION Type operator()(Type& x,Type y) const {Type old = x; x  = x | y;         return old;}};
template<typename Type> struct BXOR  {KOKKOS_INLINE_FUNCTION Type operator()(Type& x,Type y) const {Type old = x; x  = x ^ y;         return old;}};
template<typename PairType> struct Minloc {
  KOKKOS_INLINE_FUNCTION PairType operator()(PairType& x,PairType y) const {
    PairType old = x;
    if (y.first < x.first) x = y;
    else if (y.first == x.first) x.second = PetscMin(x.second,y.second);
    return old;
  }
};
template<typename PairType> struct Maxloc {
  KOKKOS_INLINE_FUNCTION PairType operator()(PairType& x,PairType y) const {
    PairType old = x;
    if (y.first > x.first) x = y;
    else if (y.first == x.first) x.second = PetscMin(x.second,y.second); /* See MPI MAXLOC */
    return old;
  }
};

/*====================================================================================*/
/*                             Atomic operations                            */
/*====================================================================================*/
template<typename Type> struct AtomicInsert  {KOKKOS_INLINE_FUNCTION void operator()(Type& x,Type y) const {Kokkos::atomic_assign(&x,y);}};
template<typename Type> struct AtomicAdd     {KOKKOS_INLINE_FUNCTION void operator()(Type& x,Type y) const {Kokkos::atomic_add(&x,y);}};
template<typename Type> struct AtomicBAND    {KOKKOS_INLINE_FUNCTION void operator()(Type& x,Type y) const {Kokkos::atomic_and(&x,y);}};
template<typename Type> struct AtomicBOR     {KOKKOS_INLINE_FUNCTION void operator()(Type& x,Type y) const {Kokkos::atomic_or (&x,y);}};
template<typename Type> struct AtomicBXOR    {KOKKOS_INLINE_FUNCTION void operator()(Type& x,Type y) const {Kokkos::atomic_fetch_xor(&x,y);}};
template<typename Type> struct AtomicLAND    {KOKKOS_INLINE_FUNCTION void operator()(Type& x,Type y) const {const Type zero=0,one=~0; Kokkos::atomic_and(&x,y?one:zero);}};
template<typename Type> struct AtomicLOR     {KOKKOS_INLINE_FUNCTION void operator()(Type& x,Type y) const {const Type zero=0,one=1;  Kokkos::atomic_or (&x,y?one:zero);}};
template<typename Type> struct AtomicMult    {KOKKOS_INLINE_FUNCTION void operator()(Type& x,Type y) const {Kokkos::atomic_fetch_mul(&x,y);}};
template<typename Type> struct AtomicMin     {KOKKOS_INLINE_FUNCTION void operator()(Type& x,Type y) const {Kokkos::atomic_fetch_min(&x,y);}};
template<typename Type> struct AtomicMax     {KOKKOS_INLINE_FUNCTION void operator()(Type& x,Type y) const {Kokkos::atomic_fetch_max(&x,y);}};
/* TODO: struct AtomicLXOR  */
template<typename Type> struct AtomicFetchAdd{KOKKOS_INLINE_FUNCTION Type operator()(Type& x,Type y) const {return Kokkos::atomic_fetch_add(&x,y);}};

/* Map a thread id to an index in root/leaf space through a series of 3D subdomains. See PetscSFPackOpt. */
static KOKKOS_INLINE_FUNCTION PetscInt MapTidToIndex(const PetscInt *opt,PetscInt tid)
{
  PetscInt        i,j,k,m,n,r;
  const PetscInt  *offset,*start,*dx,*dy,*X,*Y;

  n      = opt[0];
  offset = opt + 1;
  start  = opt + n + 2;
  dx     = opt + 2*n + 2;
  dy     = opt + 3*n + 2;
  X      = opt + 5*n + 2;
  Y      = opt + 6*n + 2;
  for (r=0; r<n; r++) {if (tid < offset[r+1]) break;}
  m = (tid - offset[r]);
  k = m/(dx[r]*dy[r]);
  j = (m - k*dx[r]*dy[r])/dx[r];
  i = m - k*dx[r]*dy[r] - j*dx[r];

  return (start[r] + k*X[r]*Y[r] + j*X[r] + i);
}

/*====================================================================================*/
/*  Wrappers for Pack/Unpack/Scatter kernels. Function pointers are stored in 'link'         */
/*====================================================================================*/

/* Suppose user calls PetscSFReduce(sf,unit,...) and <unit> is an MPI data type made of 16 PetscReals, then
   <Type> is PetscReal, which is the primitive type we operate on.
   <bs>   is 16, which says <unit> contains 16 primitive types.
   <BS>   is 8, which is the maximal SIMD width we will try to vectorize operations on <unit>.
   <EQ>   is 0, which is (bs == BS ? 1 : 0)

  If instead, <unit> has 8 PetscReals, then bs=8, BS=8, EQ=1, rendering MBS below to a compile time constant.
  For the common case in VecScatter, bs=1, BS=1, EQ=1, MBS=1, the inner for-loops below will be totally unrolled.
*/
template<typename Type,PetscInt BS,PetscInt EQ>
static PetscErrorCode Pack(PetscSFLink link,PetscInt count,PetscInt start,PetscSFPackOpt opt,const PetscInt *idx,const void *data_,void *buf_)
{
  const PetscInt          *iopt = opt ? opt->array : NULL;
  const PetscInt          M = EQ ? 1 : link->bs/BS, MBS=M*BS; /* If EQ, then MBS will be a compile-time const */
  const Type              *data = static_cast<const Type*>(data_);
  Type                    *buf = static_cast<Type*>(buf_);
  DeviceExecutionSpace    exec;

  PetscFunctionBegin;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceExecutionSpace>(exec,0,count),KOKKOS_LAMBDA(PetscInt tid) {
    /* iopt != NULL ==> idx == NULL, i.e., the indices have patterns but not contiguous;
       iopt == NULL && idx == NULL ==> the indices are contiguous;
     */
    PetscInt t = (iopt? MapTidToIndex(iopt,tid) : (idx? idx[tid] : start+tid))*MBS;
    PetscInt s = tid*MBS;
    for (int i=0; i<MBS; i++) buf[s+i] = data[t+i];
  });
  PetscFunctionReturn(0);
}

template<typename Type,class Op,PetscInt BS,PetscInt EQ>
static PetscErrorCode UnpackAndOp(PetscSFLink link,PetscInt count,PetscInt start,PetscSFPackOpt opt,const PetscInt *idx,void *data_,const void *buf_)
{
  Op                      op;
  const PetscInt          *iopt = opt ? opt->array : NULL;
  const PetscInt          M = EQ ? 1 : link->bs/BS, MBS=M*BS;
  Type                    *data = static_cast<Type*>(data_);
  const Type              *buf = static_cast<const Type*>(buf_);
  DeviceExecutionSpace    exec;

  PetscFunctionBegin;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceExecutionSpace>(exec,0,count),KOKKOS_LAMBDA(PetscInt tid) {
    PetscInt t = (iopt? MapTidToIndex(iopt,tid) : (idx? idx[tid] : start+tid))*MBS;
    PetscInt s = tid*MBS;
    for (int i=0; i<MBS; i++) op(data[t+i],buf[s+i]);
  });
  PetscFunctionReturn(0);
}

template<typename Type,class Op,PetscInt BS,PetscInt EQ>
static PetscErrorCode FetchAndOp(PetscSFLink link,PetscInt count,PetscInt start,PetscSFPackOpt opt,const PetscInt *idx,void *data,void *buf)
{
  Op                      op;
  const PetscInt          *ropt = opt ? opt->array : NULL;
  const PetscInt          M = EQ ? 1 : link->bs/BS, MBS=M*BS;
  Type                    *rootdata = static_cast<Type*>(data),*leafbuf=static_cast<Type*>(buf);
  DeviceExecutionSpace    exec;

  PetscFunctionBegin;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceExecutionSpace>(exec,0,count),KOKKOS_LAMBDA(PetscInt tid) {
    PetscInt r = (ropt? MapTidToIndex(ropt,tid) : (idx? idx[tid] : start+tid))*MBS;
    PetscInt l = tid*MBS;
    for (int i=0; i<MBS; i++) leafbuf[l+i] = op(rootdata[r+i],leafbuf[l+i]);
  });
  PetscFunctionReturn(0);
}

template<typename Type,class Op,PetscInt BS,PetscInt EQ>
static PetscErrorCode ScatterAndOp(PetscSFLink link,PetscInt count,PetscInt srcStart,PetscSFPackOpt srcOpt,const PetscInt *srcIdx,const void *src_,PetscInt dstStart,PetscSFPackOpt dstOpt,const PetscInt *dstIdx,void *dst_)
{
  PetscInt                srcx=0,srcy=0,srcX=0,srcY=0,dstx=0,dsty=0,dstX=0,dstY=0;
  const PetscInt          M = (EQ) ? 1 : link->bs/BS, MBS=M*BS;
  const Type              *src = static_cast<const Type*>(src_);
  Type                    *dst = static_cast<Type*>(dst_);
  DeviceExecutionSpace    exec;

  PetscFunctionBegin;
  /* The 3D shape of source subdomain may be different than that of the destination, which makes it difficult to use CUDA 3D grid and block */
  if (srcOpt)       {srcx = srcOpt->dx[0]; srcy = srcOpt->dy[0]; srcX = srcOpt->X[0]; srcY = srcOpt->Y[0]; srcStart = srcOpt->start[0]; srcIdx = NULL;}
  else if (!srcIdx) {srcx = srcX = count; srcy = srcY = 1;}

  if (dstOpt)       {dstx = dstOpt->dx[0]; dsty = dstOpt->dy[0]; dstX = dstOpt->X[0]; dstY = dstOpt->Y[0]; dstStart = dstOpt->start[0]; dstIdx = NULL;}
  else if (!dstIdx) {dstx = dstX = count; dsty = dstY = 1;}

  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceExecutionSpace>(exec,0,count),KOKKOS_LAMBDA(PetscInt tid) {
    PetscInt i,j,k,s,t;
    Op       op;
    if (!srcIdx) { /* src is in 3D */
      k = tid/(srcx*srcy);
      j = (tid - k*srcx*srcy)/srcx;
      i = tid - k*srcx*srcy - j*srcx;
      s = srcStart + k*srcX*srcY + j*srcX + i;
    } else { /* src is contiguous */
      s = srcIdx[tid];
    }

    if (!dstIdx) { /* 3D */
      k = tid/(dstx*dsty);
      j = (tid - k*dstx*dsty)/dstx;
      i = tid - k*dstx*dsty - j*dstx;
      t = dstStart + k*dstX*dstY + j*dstX + i;
    } else { /* contiguous */
      t = dstIdx[tid];
    }

    s *= MBS;
    t *= MBS;
    for (i=0; i<MBS; i++) op(dst[t+i],src[s+i]);
  });
  PetscFunctionReturn(0);
}

/* Specialization for Insert since we may use memcpy */
template<typename Type,PetscInt BS,PetscInt EQ>
static PetscErrorCode ScatterAndInsert(PetscSFLink link,PetscInt count,PetscInt srcStart,PetscSFPackOpt srcOpt,const PetscInt *srcIdx,const void *src_,PetscInt dstStart,PetscSFPackOpt dstOpt,const PetscInt *dstIdx,void *dst_)
{
  PetscErrorCode          ierr;
  const Type              *src = static_cast<const Type*>(src_);
  Type                    *dst = static_cast<Type*>(dst_);
  DeviceExecutionSpace    exec;

  PetscFunctionBegin;
  if (!count) PetscFunctionReturn(0);
  /*src and dst are contiguous */
  if ((!srcOpt && !srcIdx) && (!dstOpt && !dstIdx) && src != dst) {
    size_t sz = count*link->unitbytes;
    deviceBuffer_t      dbuf(reinterpret_cast<char*>(dst+dstStart*link->bs),sz);
    deviceConstBuffer_t sbuf(reinterpret_cast<const char*>(src+srcStart*link->bs),sz);
    Kokkos::deep_copy(exec,dbuf,sbuf);
  } else {
    ierr = ScatterAndOp<Type,Insert<Type>,BS,EQ>(link,count,srcStart,srcOpt,srcIdx,src,dstStart,dstOpt,dstIdx,dst);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

template<typename Type,class Op,PetscInt BS,PetscInt EQ>
static PetscErrorCode FetchAndOpLocal(PetscSFLink link,PetscInt count,PetscInt rootstart,PetscSFPackOpt rootopt,const PetscInt *rootidx,void *rootdata_,PetscInt leafstart,PetscSFPackOpt leafopt,const PetscInt *leafidx,const void *leafdata_,void *leafupdate_)
{
  Op                      op;
  const PetscInt          M = (EQ) ? 1 : link->bs/BS, MBS = M*BS;
  const PetscInt          *ropt = rootopt ? rootopt->array : NULL;
  const PetscInt          *lopt = leafopt ? leafopt->array : NULL;
  Type                    *rootdata = static_cast<Type*>(rootdata_),*leafupdate = static_cast<Type*>(leafupdate_);
  const Type              *leafdata = static_cast<const Type*>(leafdata_);
  DeviceExecutionSpace    exec;

  PetscFunctionBegin;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceExecutionSpace>(exec,0,count),KOKKOS_LAMBDA(PetscInt tid) {
    PetscInt r = (ropt? MapTidToIndex(ropt,tid) : (rootidx? rootidx[tid] : rootstart+tid))*MBS;
    PetscInt l = (lopt? MapTidToIndex(lopt,tid) : (leafidx? leafidx[tid] : leafstart+tid))*MBS;
    for (int i=0; i<MBS; i++) leafupdate[l+i] = op(rootdata[r+i],leafdata[l+i]);
  });
  PetscFunctionReturn(0);
}

/*====================================================================================*/
/*  Init various types and instantiate pack/unpack function pointers                  */
/*====================================================================================*/
template<typename Type,PetscInt BS,PetscInt EQ>
static void PackInit_RealType(PetscSFLink link)
{
  /* Pack/unpack for remote communication */
  link->d_Pack              = Pack<Type,BS,EQ>;
  link->d_UnpackAndInsert   = UnpackAndOp<Type,Insert<Type>,BS,EQ>;
  link->d_UnpackAndAdd      = UnpackAndOp<Type,Add<Type>   ,BS,EQ>;
  link->d_UnpackAndMult     = UnpackAndOp<Type,Mult<Type>  ,BS,EQ>;
  link->d_UnpackAndMin      = UnpackAndOp<Type,Min<Type>   ,BS,EQ>;
  link->d_UnpackAndMax      = UnpackAndOp<Type,Max<Type>   ,BS,EQ>;
  link->d_FetchAndAdd       = FetchAndOp <Type,Add<Type>   ,BS,EQ>;
  /* Scatter for local communication */
  link->d_ScatterAndInsert  = ScatterAndInsert<Type,BS,EQ>; /* Has special optimizations */
  link->d_ScatterAndAdd     = ScatterAndOp<Type,Add<Type>    ,BS,EQ>;
  link->d_ScatterAndMult    = ScatterAndOp<Type,Mult<Type>   ,BS,EQ>;
  link->d_ScatterAndMin     = ScatterAndOp<Type,Min<Type>    ,BS,EQ>;
  link->d_ScatterAndMax     = ScatterAndOp<Type,Max<Type>    ,BS,EQ>;
  link->d_FetchAndAddLocal  = FetchAndOpLocal<Type,Add <Type>,BS,EQ>;
  /* Atomic versions when there are data-race possibilities */
  link->da_UnpackAndInsert  = UnpackAndOp<Type,AtomicInsert<Type>  ,BS,EQ>;
  link->da_UnpackAndAdd     = UnpackAndOp<Type,AtomicAdd<Type>     ,BS,EQ>;
  link->da_UnpackAndMult    = UnpackAndOp<Type,AtomicMult<Type>    ,BS,EQ>;
  link->da_UnpackAndMin     = UnpackAndOp<Type,AtomicMin<Type>     ,BS,EQ>;
  link->da_UnpackAndMax     = UnpackAndOp<Type,AtomicMax<Type>     ,BS,EQ>;
  link->da_FetchAndAdd      = FetchAndOp <Type,AtomicFetchAdd<Type>,BS,EQ>;

  link->da_ScatterAndInsert = ScatterAndOp<Type,AtomicInsert<Type>,BS,EQ>;
  link->da_ScatterAndAdd    = ScatterAndOp<Type,AtomicAdd<Type>   ,BS,EQ>;
  link->da_ScatterAndMult   = ScatterAndOp<Type,AtomicMult<Type>  ,BS,EQ>;
  link->da_ScatterAndMin    = ScatterAndOp<Type,AtomicMin<Type>   ,BS,EQ>;
  link->da_ScatterAndMax    = ScatterAndOp<Type,AtomicMax<Type>   ,BS,EQ>;
  link->da_FetchAndAddLocal = FetchAndOpLocal<Type,AtomicFetchAdd<Type>,BS,EQ>;
}

template<typename Type,PetscInt BS,PetscInt EQ>
static void PackInit_IntegerType(PetscSFLink link)
{
  link->d_Pack              = Pack<Type,BS,EQ>;
  link->d_UnpackAndInsert   = UnpackAndOp<Type,Insert<Type> ,BS,EQ>;
  link->d_UnpackAndAdd      = UnpackAndOp<Type,Add<Type>    ,BS,EQ>;
  link->d_UnpackAndMult     = UnpackAndOp<Type,Mult<Type>   ,BS,EQ>;
  link->d_UnpackAndMin      = UnpackAndOp<Type,Min<Type>    ,BS,EQ>;
  link->d_UnpackAndMax      = UnpackAndOp<Type,Max<Type>    ,BS,EQ>;
  link->d_UnpackAndLAND     = UnpackAndOp<Type,LAND<Type>   ,BS,EQ>;
  link->d_UnpackAndLOR      = UnpackAndOp<Type,LOR<Type>    ,BS,EQ>;
  link->d_UnpackAndLXOR     = UnpackAndOp<Type,LXOR<Type>   ,BS,EQ>;
  link->d_UnpackAndBAND     = UnpackAndOp<Type,BAND<Type>   ,BS,EQ>;
  link->d_UnpackAndBOR      = UnpackAndOp<Type,BOR<Type>    ,BS,EQ>;
  link->d_UnpackAndBXOR     = UnpackAndOp<Type,BXOR<Type>   ,BS,EQ>;
  link->d_FetchAndAdd       = FetchAndOp <Type,Add<Type>    ,BS,EQ>;

  link->d_ScatterAndInsert  = ScatterAndInsert<Type,BS,EQ>;
  link->d_ScatterAndAdd     = ScatterAndOp<Type,Add<Type>   ,BS,EQ>;
  link->d_ScatterAndMult    = ScatterAndOp<Type,Mult<Type>  ,BS,EQ>;
  link->d_ScatterAndMin     = ScatterAndOp<Type,Min<Type>   ,BS,EQ>;
  link->d_ScatterAndMax     = ScatterAndOp<Type,Max<Type>   ,BS,EQ>;
  link->d_ScatterAndLAND    = ScatterAndOp<Type,LAND<Type>  ,BS,EQ>;
  link->d_ScatterAndLOR     = ScatterAndOp<Type,LOR<Type>   ,BS,EQ>;
  link->d_ScatterAndLXOR    = ScatterAndOp<Type,LXOR<Type>  ,BS,EQ>;
  link->d_ScatterAndBAND    = ScatterAndOp<Type,BAND<Type>  ,BS,EQ>;
  link->d_ScatterAndBOR     = ScatterAndOp<Type,BOR<Type>   ,BS,EQ>;
  link->d_ScatterAndBXOR    = ScatterAndOp<Type,BXOR<Type>  ,BS,EQ>;
  link->d_FetchAndAddLocal  = FetchAndOpLocal<Type,Add<Type>,BS,EQ>;

  link->da_UnpackAndInsert  = UnpackAndOp<Type,AtomicInsert<Type>,BS,EQ>;
  link->da_UnpackAndAdd     = UnpackAndOp<Type,AtomicAdd<Type>   ,BS,EQ>;
  link->da_UnpackAndMult    = UnpackAndOp<Type,AtomicMult<Type>  ,BS,EQ>;
  link->da_UnpackAndMin     = UnpackAndOp<Type,AtomicMin<Type>   ,BS,EQ>;
  link->da_UnpackAndMax     = UnpackAndOp<Type,AtomicMax<Type>   ,BS,EQ>;
  link->da_UnpackAndLAND    = UnpackAndOp<Type,AtomicLAND<Type>  ,BS,EQ>;
  link->da_UnpackAndLOR     = UnpackAndOp<Type,AtomicLOR<Type>   ,BS,EQ>;
  link->da_UnpackAndBAND    = UnpackAndOp<Type,AtomicBAND<Type>  ,BS,EQ>;
  link->da_UnpackAndBOR     = UnpackAndOp<Type,AtomicBOR<Type>   ,BS,EQ>;
  link->da_UnpackAndBXOR    = UnpackAndOp<Type,AtomicBXOR<Type>  ,BS,EQ>;
  link->da_FetchAndAdd      = FetchAndOp <Type,AtomicFetchAdd<Type>,BS,EQ>;

  link->da_ScatterAndInsert = ScatterAndOp<Type,AtomicInsert<Type>,BS,EQ>;
  link->da_ScatterAndAdd    = ScatterAndOp<Type,AtomicAdd<Type>   ,BS,EQ>;
  link->da_ScatterAndMult   = ScatterAndOp<Type,AtomicMult<Type>  ,BS,EQ>;
  link->da_ScatterAndMin    = ScatterAndOp<Type,AtomicMin<Type>   ,BS,EQ>;
  link->da_ScatterAndMax    = ScatterAndOp<Type,AtomicMax<Type>   ,BS,EQ>;
  link->da_ScatterAndLAND   = ScatterAndOp<Type,AtomicLAND<Type>  ,BS,EQ>;
  link->da_ScatterAndLOR    = ScatterAndOp<Type,AtomicLOR<Type>   ,BS,EQ>;
  link->da_ScatterAndBAND   = ScatterAndOp<Type,AtomicBAND<Type>  ,BS,EQ>;
  link->da_ScatterAndBOR    = ScatterAndOp<Type,AtomicBOR<Type>   ,BS,EQ>;
  link->da_ScatterAndBXOR   = ScatterAndOp<Type,AtomicBXOR<Type>  ,BS,EQ>;
  link->da_FetchAndAddLocal = FetchAndOpLocal<Type,AtomicFetchAdd<Type>,BS,EQ>;
}

#if defined(PETSC_HAVE_COMPLEX)
template<typename Type,PetscInt BS,PetscInt EQ>
static void PackInit_ComplexType(PetscSFLink link)
{
  link->d_Pack             = Pack<Type,BS,EQ>;
  link->d_UnpackAndInsert  = UnpackAndOp<Type,Insert<Type>,BS,EQ>;
  link->d_UnpackAndAdd     = UnpackAndOp<Type,Add<Type>   ,BS,EQ>;
  link->d_UnpackAndMult    = UnpackAndOp<Type,Mult<Type>  ,BS,EQ>;
  link->d_FetchAndAdd      = FetchAndOp <Type,Add<Type>   ,BS,EQ>;

  link->d_ScatterAndInsert = ScatterAndInsert<Type,BS,EQ>;
  link->d_ScatterAndAdd    = ScatterAndOp<Type,Add<Type>   ,BS,EQ>;
  link->d_ScatterAndMult   = ScatterAndOp<Type,Mult<Type>  ,BS,EQ>;
  link->d_FetchAndAddLocal = FetchAndOpLocal<Type,Add<Type>,BS,EQ>;

  link->da_UnpackAndInsert = UnpackAndOp<Type,AtomicInsert<Type> ,BS,EQ>;
  link->da_UnpackAndAdd    = UnpackAndOp<Type,AtomicAdd<Type>    ,BS,EQ>;
  link->da_UnpackAndMult   = UnpackAndOp<Type,AtomicMult<Type>   ,BS,EQ>;
  link->da_FetchAndAdd     = FetchAndOp<Type,AtomicFetchAdd<Type>,BS,EQ>;

  link->da_ScatterAndInsert = ScatterAndOp<Type,AtomicInsert<Type>,BS,EQ>;
  link->da_ScatterAndAdd    = ScatterAndOp<Type,AtomicAdd<Type>   ,BS,EQ>;
  link->da_ScatterAndMult   = ScatterAndOp<Type,AtomicMult<Type>  ,BS,EQ>;
  link->da_FetchAndAddLocal = FetchAndOpLocal<Type,AtomicFetchAdd<Type>,BS,EQ>;
}
#endif

template<typename Type>
static void PackInit_PairType(PetscSFLink link)
{
  link->d_Pack             = Pack<Type,1,1>;
  link->d_UnpackAndInsert  = UnpackAndOp<Type,Insert<Type>,1,1>;
  link->d_UnpackAndMaxloc  = UnpackAndOp<Type,Maxloc<Type>,1,1>;
  link->d_UnpackAndMinloc  = UnpackAndOp<Type,Minloc<Type>,1,1>;

  link->d_ScatterAndInsert = ScatterAndOp<Type,Insert<Type>,1,1>;
  link->d_ScatterAndMaxloc = ScatterAndOp<Type,Maxloc<Type>,1,1>;
  link->d_ScatterAndMinloc = ScatterAndOp<Type,Minloc<Type>,1,1>;
  /* Atomics for pair types are not implemented yet */
}

template<typename Type,PetscInt BS,PetscInt EQ>
static void PackInit_DumbType(PetscSFLink link)
{
  link->d_Pack             = Pack<Type,BS,EQ>;
  link->d_UnpackAndInsert  = UnpackAndOp<Type,Insert<Type>,BS,EQ>;
  link->d_ScatterAndInsert = ScatterAndInsert<Type,BS,EQ>;
  /* Atomics for dumb types are not implemented yet */
}

/*
  Kokkos::DefaultExecutionSpace(stream) is a reference counted pointer object. It has a bug
  that one is not able to repeatedly create and destroy the object. SF's original design was each
  SFLink has a stream (NULL or not) and hence an execution space object. The bug prevents us from
  destroying multiple SFLinks with NULL stream and the default execution space object. To avoid
  memory leaks, SF_Kokkos only supports NULL stream, which is also petsc's default scheme. SF_Kokkos
  does not do its own new/delete. It just uses Kokkos::DefaultExecutionSpace(), which is a singliton
  object in Kokkos.
*/
/*
static PetscErrorCode PetscSFLinkDestroy_Kokkos(PetscSFLink link)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
*/

/* Some device-specific utilities */
static PetscErrorCode PetscSFLinkSyncDevice_Kokkos(PetscSFLink link)
{
  PetscFunctionBegin;
  Kokkos::fence();
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFLinkSyncStream_Kokkos(PetscSFLink link)
{
  DeviceExecutionSpace    exec;
  PetscFunctionBegin;
  exec.fence();
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFLinkMemcpy_Kokkos(PetscSFLink link,PetscMemType dstmtype,void* dst,PetscMemType srcmtype,const void*src,size_t n)
{
  DeviceExecutionSpace    exec;

  PetscFunctionBegin;
  if (!n) PetscFunctionReturn(0);
  if (PetscMemTypeHost(dstmtype) && PetscMemTypeHost(srcmtype)) {
    PetscErrorCode ierr = PetscMemcpy(dst,src,n);CHKERRQ(ierr);
  } else {
    if (PetscMemTypeDevice(dstmtype) && PetscMemTypeHost(srcmtype)) {
      deviceBuffer_t       dbuf(static_cast<char*>(dst),n);
      HostConstBuffer_t    sbuf(static_cast<const char*>(src),n);
      Kokkos::deep_copy(exec,dbuf,sbuf);
      PetscErrorCode ierr = PetscLogCpuToGpu(n);CHKERRQ(ierr);
    } else if (PetscMemTypeHost(dstmtype) && PetscMemTypeDevice(srcmtype)) {
      HostBuffer_t         dbuf(static_cast<char*>(dst),n);
      deviceConstBuffer_t  sbuf(static_cast<const char*>(src),n);
      Kokkos::deep_copy(exec,dbuf,sbuf);
      PetscErrorCode ierr = PetscLogGpuToCpu(n);CHKERRQ(ierr);
    } else if (PetscMemTypeDevice(dstmtype) && PetscMemTypeDevice(srcmtype)) {
      deviceBuffer_t       dbuf(static_cast<char*>(dst),n);
      deviceConstBuffer_t  sbuf(static_cast<const char*>(src),n);
      Kokkos::deep_copy(exec,dbuf,sbuf);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSFMalloc_Kokkos(PetscMemType mtype,size_t size,void** ptr)
{
  PetscFunctionBegin;
  if (PetscMemTypeHost(mtype)) {PetscErrorCode ierr = PetscMalloc(size,ptr);CHKERRQ(ierr);}
  else if (PetscMemTypeDevice(mtype)) {
    if (!PetscKokkosInitialized) { PetscErrorCode ierr = PetscKokkosInitializeCheck();CHKERRQ(ierr); }
    *ptr = Kokkos::kokkos_malloc<DeviceMemorySpace>(size);
  } else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Wrong PetscMemType %d", (int)mtype);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSFFree_Kokkos(PetscMemType mtype,void* ptr)
{
  PetscFunctionBegin;
  if (PetscMemTypeHost(mtype)) {PetscErrorCode ierr = PetscFree(ptr);CHKERRQ(ierr);}
  else if (PetscMemTypeDevice(mtype)) {Kokkos::kokkos_free<DeviceMemorySpace>(ptr);}
  else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Wrong PetscMemType %d",(int)mtype);
  PetscFunctionReturn(0);
}

/* Destructor when the link uses MPI for communication */
static PetscErrorCode PetscSFLinkDestroy_Kokkos(PetscSF sf,PetscSFLink link)
{
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  for (int i=PETSCSF_LOCAL; i<=PETSCSF_REMOTE; i++) {
    ierr = PetscSFFree(sf,PETSC_MEMTYPE_DEVICE,link->rootbuf_alloc[i][PETSC_MEMTYPE_DEVICE]);CHKERRQ(ierr);
    ierr = PetscSFFree(sf,PETSC_MEMTYPE_DEVICE,link->leafbuf_alloc[i][PETSC_MEMTYPE_DEVICE]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* Some fields of link are initialized by PetscSFPackSetUp_Host. This routine only does what needed on device */
PetscErrorCode PetscSFLinkSetUp_Kokkos(PetscSF sf,PetscSFLink link,MPI_Datatype unit)
{
  PetscErrorCode     ierr;
  PetscInt           nSignedChar=0,nUnsignedChar=0,nInt=0,nPetscInt=0,nPetscReal=0;
  PetscBool          is2Int,is2PetscInt;
#if defined(PETSC_HAVE_COMPLEX)
  PetscInt           nPetscComplex=0;
#endif

  PetscFunctionBegin;
  if (link->deviceinited) PetscFunctionReturn(0);
  ierr = PetscKokkosInitializeCheck();CHKERRQ(ierr);
  ierr = MPIPetsc_Type_compare_contig(unit,MPI_SIGNED_CHAR,  &nSignedChar);CHKERRQ(ierr);
  ierr = MPIPetsc_Type_compare_contig(unit,MPI_UNSIGNED_CHAR,&nUnsignedChar);CHKERRQ(ierr);
  /* MPI_CHAR is treated below as a dumb type that does not support reduction according to MPI standard */
  ierr = MPIPetsc_Type_compare_contig(unit,MPI_INT,  &nInt);CHKERRQ(ierr);
  ierr = MPIPetsc_Type_compare_contig(unit,MPIU_INT, &nPetscInt);CHKERRQ(ierr);
  ierr = MPIPetsc_Type_compare_contig(unit,MPIU_REAL,&nPetscReal);CHKERRQ(ierr);
#if defined(PETSC_HAVE_COMPLEX)
  ierr = MPIPetsc_Type_compare_contig(unit,MPIU_COMPLEX,&nPetscComplex);CHKERRQ(ierr);
#endif
  ierr = MPIPetsc_Type_compare(unit,MPI_2INT,&is2Int);CHKERRQ(ierr);
  ierr = MPIPetsc_Type_compare(unit,MPIU_2INT,&is2PetscInt);CHKERRQ(ierr);

  if (is2Int) {
    PackInit_PairType<Kokkos::pair<int,int>>(link);
  } else if (is2PetscInt) { /* TODO: when is2PetscInt and nPetscInt=2, we don't know which path to take. The two paths support different ops. */
    PackInit_PairType<Kokkos::pair<PetscInt,PetscInt>>(link);
  } else if (nPetscReal) {
    if      (nPetscReal == 8) PackInit_RealType<PetscReal,8,1>(link); else if (nPetscReal%8 == 0) PackInit_RealType<PetscReal,8,0>(link);
    else if (nPetscReal == 4) PackInit_RealType<PetscReal,4,1>(link); else if (nPetscReal%4 == 0) PackInit_RealType<PetscReal,4,0>(link);
    else if (nPetscReal == 2) PackInit_RealType<PetscReal,2,1>(link); else if (nPetscReal%2 == 0) PackInit_RealType<PetscReal,2,0>(link);
    else if (nPetscReal == 1) PackInit_RealType<PetscReal,1,1>(link); else if (nPetscReal%1 == 0) PackInit_RealType<PetscReal,1,0>(link);
  } else if (nPetscInt && sizeof(PetscInt) == sizeof(llint)) {
    if      (nPetscInt == 8) PackInit_IntegerType<llint,8,1>(link); else if (nPetscInt%8 == 0) PackInit_IntegerType<llint,8,0>(link);
    else if (nPetscInt == 4) PackInit_IntegerType<llint,4,1>(link); else if (nPetscInt%4 == 0) PackInit_IntegerType<llint,4,0>(link);
    else if (nPetscInt == 2) PackInit_IntegerType<llint,2,1>(link); else if (nPetscInt%2 == 0) PackInit_IntegerType<llint,2,0>(link);
    else if (nPetscInt == 1) PackInit_IntegerType<llint,1,1>(link); else if (nPetscInt%1 == 0) PackInit_IntegerType<llint,1,0>(link);
  } else if (nInt) {
    if      (nInt == 8) PackInit_IntegerType<int,8,1>(link); else if (nInt%8 == 0) PackInit_IntegerType<int,8,0>(link);
    else if (nInt == 4) PackInit_IntegerType<int,4,1>(link); else if (nInt%4 == 0) PackInit_IntegerType<int,4,0>(link);
    else if (nInt == 2) PackInit_IntegerType<int,2,1>(link); else if (nInt%2 == 0) PackInit_IntegerType<int,2,0>(link);
    else if (nInt == 1) PackInit_IntegerType<int,1,1>(link); else if (nInt%1 == 0) PackInit_IntegerType<int,1,0>(link);
  } else if (nSignedChar) {
    if      (nSignedChar == 8) PackInit_IntegerType<char,8,1>(link); else if (nSignedChar%8 == 0) PackInit_IntegerType<char,8,0>(link);
    else if (nSignedChar == 4) PackInit_IntegerType<char,4,1>(link); else if (nSignedChar%4 == 0) PackInit_IntegerType<char,4,0>(link);
    else if (nSignedChar == 2) PackInit_IntegerType<char,2,1>(link); else if (nSignedChar%2 == 0) PackInit_IntegerType<char,2,0>(link);
    else if (nSignedChar == 1) PackInit_IntegerType<char,1,1>(link); else if (nSignedChar%1 == 0) PackInit_IntegerType<char,1,0>(link);
  }  else if (nUnsignedChar) {
    if      (nUnsignedChar == 8) PackInit_IntegerType<unsigned char,8,1>(link); else if (nUnsignedChar%8 == 0) PackInit_IntegerType<unsigned char,8,0>(link);
    else if (nUnsignedChar == 4) PackInit_IntegerType<unsigned char,4,1>(link); else if (nUnsignedChar%4 == 0) PackInit_IntegerType<unsigned char,4,0>(link);
    else if (nUnsignedChar == 2) PackInit_IntegerType<unsigned char,2,1>(link); else if (nUnsignedChar%2 == 0) PackInit_IntegerType<unsigned char,2,0>(link);
    else if (nUnsignedChar == 1) PackInit_IntegerType<unsigned char,1,1>(link); else if (nUnsignedChar%1 == 0) PackInit_IntegerType<unsigned char,1,0>(link);
#if defined(PETSC_HAVE_COMPLEX)
  } else if (nPetscComplex) {
    if      (nPetscComplex == 8) PackInit_ComplexType<Kokkos::complex<PetscReal>,8,1>(link); else if (nPetscComplex%8 == 0) PackInit_ComplexType<Kokkos::complex<PetscReal>,8,0>(link);
    else if (nPetscComplex == 4) PackInit_ComplexType<Kokkos::complex<PetscReal>,4,1>(link); else if (nPetscComplex%4 == 0) PackInit_ComplexType<Kokkos::complex<PetscReal>,4,0>(link);
    else if (nPetscComplex == 2) PackInit_ComplexType<Kokkos::complex<PetscReal>,2,1>(link); else if (nPetscComplex%2 == 0) PackInit_ComplexType<Kokkos::complex<PetscReal>,2,0>(link);
    else if (nPetscComplex == 1) PackInit_ComplexType<Kokkos::complex<PetscReal>,1,1>(link); else if (nPetscComplex%1 == 0) PackInit_ComplexType<Kokkos::complex<PetscReal>,1,0>(link);
#endif
  } else {
    MPI_Aint lb,nbyte;
    ierr = MPI_Type_get_extent(unit,&lb,&nbyte);CHKERRMPI(ierr);
    if (lb != 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Datatype with nonzero lower bound %ld\n",(long)lb);
    if (nbyte % sizeof(int)) { /* If the type size is not multiple of int */
      if      (nbyte == 4) PackInit_DumbType<char,4,1>(link); else if (nbyte%4 == 0) PackInit_DumbType<char,4,0>(link);
      else if (nbyte == 2) PackInit_DumbType<char,2,1>(link); else if (nbyte%2 == 0) PackInit_DumbType<char,2,0>(link);
      else if (nbyte == 1) PackInit_DumbType<char,1,1>(link); else if (nbyte%1 == 0) PackInit_DumbType<char,1,0>(link);
    } else {
      nInt = nbyte / sizeof(int);
      if      (nInt == 8) PackInit_DumbType<int,8,1>(link); else if (nInt%8 == 0) PackInit_DumbType<int,8,0>(link);
      else if (nInt == 4) PackInit_DumbType<int,4,1>(link); else if (nInt%4 == 0) PackInit_DumbType<int,4,0>(link);
      else if (nInt == 2) PackInit_DumbType<int,2,1>(link); else if (nInt%2 == 0) PackInit_DumbType<int,2,0>(link);
      else if (nInt == 1) PackInit_DumbType<int,1,1>(link); else if (nInt%1 == 0) PackInit_DumbType<int,1,0>(link);
    }
  }

  link->SyncDevice   = PetscSFLinkSyncDevice_Kokkos;
  link->SyncStream   = PetscSFLinkSyncStream_Kokkos;
  link->Memcpy       = PetscSFLinkMemcpy_Kokkos;
  link->Destroy      = PetscSFLinkDestroy_Kokkos;
  link->deviceinited = PETSC_TRUE;
  PetscFunctionReturn(0);
}
