# file: ex2pycuda.py
from petsc4py import PETSc as petsc
import numpy

def ScreeningAvg(fwk, key, stage, avg):
    if avg is None:
        avg = PETSc.Mat().create(fwk.comm)
        fwk.registerDependence("Electrolyte", key)
        return avg
    if stage == "init":
        assert isinstance(avg,PETSc.Mat)
        # Extract the "DensityField" component, and from it, the mesh (DA) and vectors of mesh spacings and ion radii
        # The DA supports vectors with d+1 degrees of freedom: d ion species + Gamma (local screening parameter).
        # The screening average operator averages the first d components (ion densities),
        # over a ball of radius constructed using the densities and Gamma.
        e     = fwk.getComponent("Electrolyte")
        da    = e.query("mesh")
        comm  = da.getComm()
        if comm.getSize() > 1:
            raise Exception("Serial only is supported right now.")
        h     = e.query("h")
        radii = e.query("radii")
        #
        N   = da.getSizes()
        dim = da.getDim()
        if dim != 3:
            raise Exception("mesh must be 3D")
        meshSize = N[0]*N[1]*N[2]
        numSpecies = radii.getSize()
        #
        matSizes = [meshSize*numSpecies, meshSize*(numSpecies+1)]
        avg.create(comm)
        avg.setSizes(matSizes)
        avg.setType('python')
        avg.setPythonContext(MatScreeningAvg())
        # Now store the da, the radii and the mesh spacing vector h by composing them with c
        avg.compose("mesh", da)
        avg.compose("radii", radii)
        avg.compose("h",h)
        # Also allocate a vector to hold pointwise screening radii; store it too
        R = PETSc.Vec()
        R.createSeq(meshSize,comm=comm)
        avg.compose("R",R)
    return avg



class MatScreeningAvg:

    def __init__(self):
        pass
    
    def create(self, A):
        pass

    def mult(self, A, x, y):
        "y <- A * x"
        da       = A.query("mesh")
        R        = A.query("R").getArray()
        radii    = A.query("radii").getArray()
        h        = A.query("h").getArray()
        #
        N              = da.getSizes()
        dof            = da.getDof()
        M              = (N[0],N[1],N[2],dof)
        xx = x[...].reshape(M, order='C')
        yy = y[...].reshape(N, order='C')
        # call PyCUDA stuff
        doScreeningAvg(yy,xx,h,radii,R)
        
# In most Python subroutines we place the output argument first,
# in part, because later arguments can have default values, be keyword arguments, etc.
        
import time
import pycuda
import pycuda.autoinit
from pycuda import gpuarray
import pycuda.driver as drv
from pycuda.reduction import ReductionKernel
from pycuda.compiler import SourceModule

import numpy.random as rnd

from math import *

def doScreeningAveraging(rhoAvg, rhoGamma, h, R, radii):
    M          = rhoGamma.shape
    dim        = [M[0], M[1], M[2]]
    numSpecies = M[3]-1
    # 
    rho        = rhoGamma[:,:,:,:numSpecies]
    Gamma      = rhoGamma[:,:,:,numSpecies]
    #calculate R
    R_filter = calculateR(R, Gamma,rho,radii)
    #for each density calculate the locally averaged density
    for sp in range(numSpecies):
        doScreeningAvgSingle(rhoAvg[:,:,:,sp], rho[:, :, :, sp], h, R)


def calculateR(R, Gamma, rho, radii):
    n_sp = len(radii)
    dim = Gamma.shape
    for i in range(dim[0]):
        for j in range(dim[1]):
            for k in range(dim[2]):
                r_avg = 0.
                rho_avg = 0.
                for sp in range(n_sp):
                    r_avg += radii[sp]*rho[i, j, k, sp]
                    rho_avg += rho[i, j, k, sp]
                if (rho_avg == 0):
                    r_avg = 0.
                else:
                    r_avg = r_avg/rho_avg
                R[i, j, k] = r_avg + 1. / (2.*Gamma[i, j, k]) 


# Calculates the pointwise local average of each species' density by using the GPU
#
# Arguments:
# Output:
#  rhoAvg - the locally averaged density
# Input:
#  rho    - the input density
#  radius - the pointwise screening radius
# Optional:
#  nblocks - the number of blocks to use on the GPU
#  x_per - the size of the per-kernel realspace blocks done
#
def doScreeningAvgSingle(rhoAvg, rho, h, R, nBlocks=screening_blks, x_per=x_per_kernel, k_per=k_per_kernel):

    dim = R.shape
    
    rhohat_r, rhohat_i = fft_field(rho)

    rhohat_r_gpu = gpuarray.to_gpu_async(numpy.array(rhohat_r, dtype=numpy.float32))
    rhohat_i_gpu = gpuarray.to_gpu_async(numpy.array(rhohat_i, dtype=numpy.float32))

    #t1 = time.time()

    strm = drv.Stream()
    
    R_gpu = gpuarray.to_gpu_async(numpy.array(R, dtype=numpy.float32))

    rho_t = gpuarray.empty((x_per, x_per, x_per, nBlocks), dtype=numpy.float32)

    k_0 = numpy.array((0, 0, 0), dtype = numpy.int32)
    k_size = numpy.array((dim[0], dim[1], dim[2]), dtype=numpy.int32)
    l = numpy.array((dim[0]*h[0], dim[1]*h[1], dim[2]*h[2]), dtype=numpy.float32)

    rho_s = gpuarray.to_gpu_async(numpy.array(numpy.zeros_like(rho), dtype=numpy.float32))
    for k in range(0, dim[2], x_per):
        for j in range(0, dim[1], x_per):
            for i in range(0, dim[0], x_per):                        
                x_0 = numpy.array((i, j, k), dtype=numpy.int32)
                x_size = numpy.array((x_per, x_per, x_per), dtype=numpy.int32)
                calculateScreeningAvgBlock(rho_t, x_0, R_gpu, rhohat_r_gpu, rhohat_i_gpu, k_0, l, k_size, stream=strm)
                accumulateScreeningAvgBlock(rho_s, rho_t, x_0, stream = strm)
    strm.synchronize()
    #t2 = time.time()
    #print "first kernel took %0.3f ms" % ((t2 - t1)*1000)
    drv.memcpy_dtoh_asymc(rhoAvg,rho_s,strm)


#just takes the fft and then normalizes it.
def fft_field(rho):
    rhohat = numpy.fft.fftn(rho)
    n_x, n_y, n_z = rho.shape
    for i in range(n_x):
        for j in range(n_y):
            for k in range(n_z):
                rhohat[i, j, k] = rhohat[i, j, k] / (n_x*n_y*n_z)
    return numpy.real(rhohat), numpy.imag(rhohat)

#number of blocks / threads to use and points to do at a time
screening_blks = 512
screening_threads = 256
reduction_threads = screening_blks
x_per_kernel = 16

k_per_kernel = n #use to limit the size of each summation

def calculateScreeningAvgBlock(rho_block, x_0, B, A_r, A_i, k_0, l, n, threads = screening_threads, stream = None, REAL=numpy.float32):
    '''Applies the screening averaging operator to a small piece of physical space.
      Arguments:
      Output:
        rho_block - an array with the shape of the physical space block and the number of blocks (float32)
      Input:
        x_0   - the x offset, in integers, of this particular block in the whole physical space (int32)
        B     - the whole realspace screening radius function (float32)
        A_r   - the real component of rhohat (float32)
        A_i   - the imaginary component of rhohat (float32)
        k_0   - the offset into a potentially larger Fourier space block (int32)
        l     - the sizes of the physical domain (float32, float32, float32)
        n     - grid point counts in each direction  (int32, int32, int32)
       stream - the serializing stream'''

    code = """
    
#include<math_constants.h>

//problem parameters

//the per-thread stride
#define k_blk 512

//the number of 
#define n_k_blks 

//problem geometry defines

#define nx n[0]
#define ny n[1]
#define nz n[2]

#define nxx n_x[0]
#define nxy n_x[1]
#define nxz n_x[2]

#define x0x x_0[0]
#define x0y x_0[1]
#define x0z x_0[2]

#define nkx n_k[0]
#define nky n_k[1]
#define nkz n_k[2]

#define k0x k_0[0]
#define k0y k_0[1]
#define k0z k_0[2]

#define lx l[0]
#define ly l[1]
#define lz l[2]

#define k_size nkx*nky*nkz
#define x_size nxx*nxy*nxz

//problem mathematics defines

#define PI CUDART_PI

//CUDA block/thread manipulation defines

#define tid threadIdx.x
#define bid blockIdx.x
#define t_end blockDim.x
#define b_end gridDim.x

__global__ void calculateScreeningAvgBlock(float * Rho_s, //output, of size prod(n_x)*gridDim.x
                                           int * x_0,     //x offset, integers
                                           int * n_x,     //number of space steps to take in each
				           float * B,     //screening radius, size prod(n_x)
				           float * A_r,   //fhat_real, size prod(n_k)
				           float * A_i,   //fhat_imag, size prod(n_k)
                                           int * k_0,     //k offset, integers
                                           int * n_k,     //number of k steps in each direction
                                           float * l,     //length scale in each direction
				           int * n)       //total problem size in each direction
{     
#warning start debug line numbers from here
//in this version: each block gets assigned a range of frequency space and then a per-block reduction occurs.
//The combination of the overall data happens in realspace

  //fourier space buffers
  __shared__ float A_cr[k_blk];
  __shared__ float A_ci[k_blk];
  __shared__ float F_x[k_blk];
  __shared__ float F_y[k_blk];
  __shared__ float F_z[k_blk];
  __shared__ float F_f[k_blk];

  //have each block zero out the portion of memory it will be writing to
  for (int i = tid; i < x_size; i += t_end) {
    Rho_s[bid + i*b_end] = 0.;
  }
  __syncthreads();
  const float ilx = 1. / lx;
  const float ily = 1. / ly;
  const float ilz = 1. / lz;
  const float inx = 1. / nx;
  const float iny = 1. / ny;
  const float inz = 1. / nz;

  double sum;
  float x;
  float y;
  float z;
  float R;
            
  //distribute fourier space to the various blocks
  for (int k_i = bid*k_blk; k_i < k_size; k_i += b_end*k_blk) {  //loop over block-radix

    for (int k_blk_i = tid; (k_blk_i < k_blk) && (k_blk_i + k_i < k_size); k_blk_i += t_end) { //loop over thread-radix (1s place!)
      //copy 
      A_cr[k_blk_i] = A_r[k_blk_i + k_i];
      A_ci[k_blk_i] = A_i[k_blk_i + k_i];

      int _i_x = ((k_blk_i + k_i) / (nz*ny)) + k0x;
      int _i_y = (((k_blk_i + k_i - _i_x*ny*nz) / nz)) + k0y;
      int _i_z = ((k_blk_i + k_i - _i_y*nz - _i_x*ny*nz)) + k0z;

      _i_x = _i_x < nx / 2  ? _i_x : -nx + _i_x;
      _i_y = _i_y < ny / 2  ? _i_y : -ny + _i_y;
      _i_z = _i_z < nz / 2  ? _i_z : -nz + _i_z;


      float f_x = 2.*ilx*PI*_i_x;
      float f_y = 2.*ily*PI*_i_y;
      float f_z = 2.*ilz*PI*_i_z;

      F_x[k_blk_i] = f_x;
      F_y[k_blk_i] = f_y;
      F_z[k_blk_i] = f_z;
      F_f[k_blk_i] = sqrt(f_x*f_x + f_y*f_y + f_z*f_z);
    }
    //synchronize the threads because they all read from every spot in shared memory

    __syncthreads();
    for (int x_i = tid; x_i < x_size; x_i += t_end) {
      //calculate the x, y, and z values of this block from the offset
      int i_x = (x_i / (nxz*nxy));
      int i_y = (x_i - i_x*nxy*nxz) / nxz;
      int i_z = (x_i - i_y*nxz - i_x*nxy*nxz);
      i_x += x0x;
      i_y += x0y;
      i_z += x0z;
      if (i_x < nx && i_y < ny && i_z < nz) {  
        x = inx*lx*(i_x);
        y = iny*ly*(i_y);
        z = inz*lz*(i_z);
        R = B[i_x*ny*nz + i_y*nz + i_z];
        sum = 0.;
        for (int k_blk_i = 0; (k_blk_i < k_blk) && (k_blk_i + k_i < k_size); k_blk_i++) {
            float k_x = F_x[k_blk_i];
            float k_y = F_y[k_blk_i];
            float k_z = F_z[k_blk_i];
            float f = F_f[k_blk_i];
            float ai = A_ci[k_blk_i];
            float ar = A_cr[k_blk_i];
            float fdotx = k_x*x + k_y*y + k_z*z;
            float Rf = R*f;
            //float iRf = 1. / Rf;
            float Rf2 = Rf*Rf;
            float Rf3 = Rf2*Rf;
            float sinRf;
            float cosRf;
            float sinfx;
            float cosfx;
            __sincosf(Rf, &sinRf, &cosRf);
            __sincosf(fdotx, &sinfx, &cosfx);
            sum += (-sinfx*ai + ar*cosfx)*(Rf > 0. ? 3.*(sinRf/Rf3 - cosRf/Rf2) : 1.);
        }
        __syncthreads();
        Rho_s[bid + b_end*x_i] += sum;
      }
    }
  }
}
"""

    avg_funct = SourceModule(code).get_function("calculateScreeningAvgBlock")

    n_x = numpy.array(rho_block.shape[0:3], dtype=numpy.int32)
    n_k = numpy.array(A_r.shape, dtype=numpy.int32)
    n_x_total = numpy.array(B.shape, dtype=numpy.int32)
    nBlocks = rho_block.shape[3]

    avg_funct(rho_block,
                 drv.In(x_0),
                 drv.In(n_x),
                 B,
                 A_r,
                 A_i,
                 drv.In(k_0),
                 drv.In(n_k),
                 drv.In(l),
                 drv.In(n_x_total),
                 stream = stream,
                 grid=(nBlocks, 1),
                 block=(threads, 1, 1))

def accumulateScreeningAvgBlock(rho, rho_block, x0, stream=None, threads = 256):#
    '''Takes a block from the previous calculation and reduces it to a block of realspace by#
    contracting out the block dimension and sticking the whole thing into a bigger array
    Arguments:
    Output:
      rho       - averaged density over the whole physical mesh
    Input:
      rho_block - a physical-space-block-size by number-of-blocks array
      x0        - the offset of the block into realspace
    Optional:
      stream    - the serializing stream
      threads   - number of threads per block'''

    code = '''
#define n nx*ny*nz

#define n_accum 256

#define i_x blockIdx.x
#define i_y blockIdx.y
#define tid threadIdx.x
#define bn gridDim.x
#define tn blockDim.x

__global__ void accumulateScreeningAvgBlock(float * B,  //Big array
                                          float * A,  //Small block
                                          int b,      //number of blocks
                                          int ax,     //x size of A
                                          int ay,     //y size of A
                                          int az,     //z size of A
                                          int bx,     //x size of B
                                          int by,     //y size of B
                                          int bz,     //z size of B
                                          int x0,     //offset of A into B
                                          int y0,     // .. 
                                          int z0) {   // ..

  __shared__ float accum[n_accum];
  double sum = 0.;


  for (int i_z = 0; i_z < az; i_z++) {
    int off = b*(i_z + (i_x*ay + i_y)*az);
    for (int i_accum = tid; i_accum < n_accum; i_accum += tn) {
      sum = 0.;
      //c_kah = 0.;
      for (int i_b = i_accum; i_b < b; i_b += n_accum) {
        
        //y_kah = A[i_b + off] + c_kah;
        //t_kah = sum + y_kah;
        //c_kah = (t_kah - sum) - y_kah;
        //sum = t_kah;
        sum += A[i_b + off];
      }
      accum[i_accum] = sum;
    }
    for(int stride = n_accum / 2; stride > 0; stride >>= 1){
      __syncthreads();
      for(int iAccum = tid; iAccum < stride; iAccum += tn) {
        accum[iAccum] += accum[stride + iAccum];
      }
    }
    if (tid == 0) if (i_z + z0 < bz && i_y + y0 < by && i_x + x0 < bx) B[i_z + z0 + ((i_x + x0)*by + i_y + y0)*bz] = accum[0];
  }
}
    '''
    #blk = block.get()
    s = rho_block.shape
    t = rho.shape
    ax = s[0]
    ay = s[1]
    az = s[2]
    bx = t[0]
    by = t[1]
    bz = t[2]
    xx = x0[0]
    xy = x0[1]
    xz = x0[2]
    blks = s[3]
    sum_funct = SourceModule(code).get_function("accumulateScreeningAvgBlock")
    sum_funct(rho, rho_block, numpy.int32(blks),
              numpy.int32(ax), numpy.int32(ay), numpy.int32(az),
              numpy.int32(bx), numpy.int32(by), numpy.int32(bz),
              numpy.int32(xx), numpy.int32(xy), numpy.int32(xz),
              block=(threads, 1, 1),
              grid=(ax, ay),
              stream=stream)





