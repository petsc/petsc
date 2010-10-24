# file: ex2pycuda.py
from petsc4py import PETSc as petsc
import numpy

class ScreeningAvg:
    @staticmethod
    def init(a):
        fwk = a.getParent()
        key = a.getName()
        fwk.registerDependence("Electrolyte",key)
        e = fwk.getComponent("Electrolyte")
        a.compose("Electrolyte", e)
        
    @staticmethod
    def setup(a):
        # Extract the "DensityField" component, and from it, the mesh (DMDA) and vectors of mesh spacings and ion radii
        # The DMDA supports vectors with d+1 degrees of freedom: d ion species + Gamma (local screening parameter).
        # The screening average operator averages the first d components (ion densities),
        # over a ball of radius constructed using the densities and Gamma.
        e     = a.query("Electrolyte")
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
        ##
        avg = petsc.Mat()
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
        a.compose("averagingOperator",avg)



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
        doScreeningAveraging(yy,xx,h,radii,R)
        
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
    calculateR(R, Gamma,rho,radii)
    #for each density calculate the locally averaged density
    for sp in range(numSpecies):
        doScreeningAveragingSingleGPU(rhoAvg[:,:,:,sp], rho[:, :, :, sp], h, R)


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

# Calculates the normalized FFT of a density.
def densityFFT(rho):
    rhohat = numpy.fft.fftn(rho)
    n_x, n_y, n_z = rho.shape
    for i in range(n_x):
        for j in range(n_y):
            for k in range(n_z):
                rhohat[i, j, k] = rhohat[i, j, k] / (n_x*n_y*n_z)
    return numpy.real(rhohat), numpy.imag(rhohat)

# Calculates the pointwise local average of each species' density by using GPU
#
# Arguments:
# Output:
#  rhoAvg - the locally averaged density
# Input:
#  rho    - the input density
#  radius - the pointwise screening radius
#
def doScreeningAveragingSingleGPU(rhoAvg, rho, h, R):
    N             = numpy.array(R.shape, dtype=numpy.int32)
    blockCount    = 512 # = (2^3)^3 = 8^3
    bockSize     = 32 # threads per block: assume 8 SPs, dual-clocked, dual-issue
    n             = numpy.array([(N[0]+8)/8,(N[1]+8)/8,(N[2]+8)/8], dtype=numpy.int32) # brick size
    #
    #
    # rho_avg_inflated_brick_gpu is an 'inflated' brick of physical space output values, with a blockCount inflation factor
    rho_avg_inflated_brick_gpu = gpuarray.empty((n[0],n[1], n[2], blockCount), dtype=numpy.float32)
    #
    rhohat_r, rhohat_i = densityFFT(rho)
    # FIX: move FFT up and apply to ALL species at once (can numpy do that?)
    #      better yet, use cudafft
    rhohat_r_gpu = gpuarray.to_gpu_async(numpy.array(rhohat_r, dtype=numpy.float32))
    rhohat_i_gpu = gpuarray.to_gpu_async(numpy.array(rhohat_i, dtype=numpy.float32))
    #
    R_gpu = gpuarray.to_gpu_async(numpy.array(R, dtype=numpy.float32))
    L = numpy.array((N[0]*h[0], N[1]*h[1], N[2]*h[2]), dtype=numpy.float32)
    #
    strm = drv.Stream()
    #
    # rho_avg_gpu is the set of 'reduced' physical space values
    # Each brick of rho_s_gpu values is generated by taking a corresponding rho_avg_inflated_brick_gpu
    # and 'reducing' blockCount consecutive values of the brick to one rho_avg_gpu value
    rho_avg_gpu = gpuarray.to_gpu_async(numpy.array(numpy.zeros_like(rho), dtype=numpy.float32))
    for k in range(0, N[2], n[2]):
        for j in range(0, N[1], n[1]:
            for i in range(0, N[0], n[2]):                        
                x0 = numpy.array((i, j, k), dtype=numpy.int32)
                x1 = numpy.array((i+n[0], j+n[1], k+n[2]), dtype=numpy.int32)
                calculateScreeningAvgInflatedBrickGPU(rho_avg_inflated_brick_gpu, x0, x1, rhohat_r_gpu, rhohat_i_gpu, R_gpu, L, N, blockSize=blockSize, blockCount=blockCount,stream=strm)
                # blockCount used above now becomes blockSize, since we want to have as many threads working on reduction, as there are elements to reduce
                accumulateScreeningAvgGPU(rho_avg_gpu, rho_avg_inflated_brick_gpu, x0, x1, blockSize=blockCount, stream = strm)
    strm.synchronize()
    drv.memcpy_dtoh_asymc(rhoAvg,rho_avg_gpu,strm)



def calculateScreeningAvgInflatedBrickGPU(rho_avg_inflated_brick_gpu, x0, x1, rhohat_r_gpu, rhohat_i_gpu, R_gpu, N, L, blockSize, blockCount, stream = None):
    '''
      Notes:
         This routine is implements a (partial) action of the screening average matrix onto a density vector.
       The rows of the matrix are labeled by the points of a real space mesh (a subset of \Z^3),
       and the columns are labeled by the points of an "isomorphic" FFT-space (called Fourier space in the sequel) mesh.
       This routine applies only a subset of rows of the matrix corresponding to the '[i0, i1)' brick of
       of the mesh (i0,i1 \in \Z^3).  The brick has dimensions  'n = (nx,ny,nz) = i1-i0', and the total
       number of matrix rows in the brick is nx*ny*nz. The algorithm is a natural extension of the classical
       matrix-matrix multiplication example, using shared memory.
         During the application, we use 'blockCount' CUDA blocks, each with 'blockSize' threads in it.
       Each block contributes to the action of EACH matrix row on the density vector, by contracting the two over
       a subset of columns, effectively carrying out the row-density inner product in parallel.
       The contributions to each row-density inner product from different CUDA blocks are NOT summed,
       but returned separately, resulting in an 'inflated' array of return vlues: each row value is replaced by
       'blockCount' consecutive partial values, for or a total of 'nx*ny*nz*blockCount' (block index is varying most
       rapidly).
         Within each thread block only one thread contributes to the action of a given row, but each thread may
       contribute to multiple row-density contractions: rows with indices equal mod blockSize are assigned to the same
       thread, that is, rows are assigned to threads with stride 'blockSize'.
         Columns are assigned to the CUDA threads the following way: chunks of column indices in 'chunkSize' increments
       are assigned to thread blocks repeatedly, until all columns are exhausted.  Within each block the columns from
       each chunk are distributed to threads mod 'blockSize'.
      Arguments:
      Output:
        rho_inflated_brick_gpu - an array with the shape of the physical space brick and the number of CUDA blocks - float32[nx,ny,nz,blockCount]
      Input:
        i0             - the starting multi-index of this physical space brick;    int32[3]
        i1             - the ending   multi-index of this physical space brick;    int32[3]
        rhohat_r_gpu   - "amplitude", the real component of rhohat;                float32[Nx,Ny,Nz]
        rhohat_i_gpu   - "amplitude", the imaginary component of rhohat;           float32[Nx,Ny,Nz]
        R_gpu          - the whole real space screening radius function;           float32[Nx,Ny,Nz]
        N              - grid point counts in each direction;                      int32[3]
        L              - the sizes of the physical domain;                         float32[3]
        blockCount     - number of CUDA blocks in the grid to use;                 int
        threadCount    - number of CUDA threads per CUDA block to use;             int
        stream         - the serializing stream
        '''

    code = """
    
#include<math_constants.h>

__global__ void calculateScreeningAvgInflatedBrick(
                                           float *rho_inflated_brick, // averaged density inflated brick;         float32[nx*ny*nz*blockCount]
                                           int   *i0,                 // brick start multi-index;                 int32[3]
                                           int   *i1,                 // brick end   multi-index;                 int32[3]
				           float *rho_hat_r,          // density FFT, real part;                  float32[N[0]*N[1]*N[2]]
				           float *rho_hat_i,          // density FFT, imaginary part;             float32[N[0]*N[1]*N[2]]
                                           float *R,                  // screening radius;                        float32[N[0]*N[1]*N[2]]
                                           int   *N,                  // number of mesh points in each direction; int32[3]
                                           float *L)                  // physical domain size in each direction;  float32[3]
{     
#warning start debug line numbers from here


//// Geometric parameters

// Whole physical domain
#define Nx N[0]
#define Ny N[1]
#define Nz N[2]

#define Lx L[0]
#define Ly L[1]
#define Lz L[2]

// Physical space brick
#define nx (i1[0] - i0[0])
#define ny (i1[1] - i0[1])
#define nz (i1[2] - i0[2])

#define ix0 i0[0]
#define iy0 i0[1]
#define iz0 i0[2]

#define brickSize nx*ny*nz
#define meshSize  Nx*Ny*Nz
#define chunkSize 512


//// Math defines

#define PI CUDART_PI

//// CUDA block/thread defines

#define threadId    threadIdx.x
#define blockId     blockIdx.x
#define blockSize   blockDim.x // 'threadsPerBlock'
#define blockCount  gridDim.x

  //// Fourier space buffers
  // Amplitudes
  __shared__ float A_r[blockSize];
  __shared__ float A_i[blockSize];
  // Frequencies
  __shared__ float F_x[blockSize];
  __shared__ float F_y[blockSize];
  __shared__ float F_z[blockSize];
  __shared__ float F_f[blockSize];


  // Have each block zero out the portion of memory it will be writing to
  for (int i = threadId; i < brickSize; i += blockSize) {
    rho_inflated_brick[blockId + i*blockCount] = 0.;
  }
  __syncthreads();


  double sum;
  float x;
  float y;
  float z;
  float R;
            
  // Loop over chunks of columns in the whole Fourier space (of size 'meshSize')
  for (int ch = blockId*chunkSize; ch < meshSize; ch += blockCount*chunkSize) {  // one chunk per block, then repeat, until all columns in the mesh are exhausted
     // Loop over columns within each chunk
     for (int t = threadId; (t < chunkSize) && (ch + j < meshSize); t += blockSize) { // one column per thread, then repeat, until all columns in the chunk are exhausted
      int j = ch + t; // column index
      // Copy a chunk of density data  
      A_r[t] = rhohat_r[j];
      A_i[t] = rhohat_i[j];

      // Calculate frequencies corresponding to column j in Fourier space
      int jx = (j/ (nz*ny));
      int jy = (((j - jx*ny*nz) / nz));
      int jz = ((j - jy*nz - jx*ny*nz));

      jx = jx < Nx / 2  ? jx : -Nx + jx;
      jy = jy < Ny / 2  ? jy : -Ny + jy;
      jz = jz < Nz / 2  ? jz : -Nz + jz;


      float f_x = (2.*PI*jx)/L[0];
      float f_y = (2.*PI*jy)/L[1];
      float f_z = (2.*PI*jz)/L[2];

      F_x[t] = f_x;
      F_y[t] = f_y;
      F_z[t] = f_z;
      F_f[t] = sqrt(f_x*f_x + f_y*f_y + f_z*f_z);
    }
    // Synchronize the threads to ensure that writing to ALL of the shared memory is done before proceeding on to the next stage.
    // This is because in the next stage EVERY thread will need to read from ALL of the shared memory,


    __syncthreads();
    // Loop over rows
    for (int i = threadId; i < brickSize; i += blockSize) { // one row per thread, then repeat, until all rows in the brick are exhausted
      // Calculate the x, y, and z values of this row from its index i
      // The x,y,z directions vary in this order -- z,y,x -- from the most to the least rapidly varying.
      int ix = (i / (nz*ny));
      int iy = (i - ix*ny*nz)/ nz;
      int iz = (i - iy*nz - ix*ny*nz);
      ix += ix0;
      iy += iy0;
      iz += iz0;
      if (ix < Nx && iy < Ny && iz < Nz) {
        x = (Lx*ix)/Nx;
        y = (Ly*iy)/Ny;
        z = (Lz*iz)/Nz;
        // Screening radius at point (x,y,z)
        Rxyz = R[ix*Ny*Nz + iy*Nz + iz];
        sum = 0.;
        for (int c = 0; (c < chunkSize) && (ch+c < meshSize); c++) { // loop over ALL columns in chunk starting at ch
            float kx = F_x[c];
            float ky = F_y[c];
            float kz = F_z[c];
            float f  = F_f[c];
            float ai = A_i[c];
            float ar = A_r[c];
            float fdotx = kx*x + ky*y + kz*z;
            float Rf = Rxyz*f;
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
        rho_inflated_brick[blockId + blockCount*x_i] += sum;
      }
    }
  }
}
"""
    avg_funct = SourceModule(code).get_function("calculateScreeningAvgInflatedBrick")
    avg_funct(rho_inflated_brick_gpu,
                 drv.In(x0),
                 drv.In(x1),
                 rhohat_r_gpu,
                 rhohat_i_gpu,
                 R_gpu,
                 drv.In(L),
                 drv.In(N),
                 stream = stream,
                 grid=(blockCount, 1),
                 block=(blockSize, 1, 1))
                       

def accumulateScreeningAvgGPU(rho_gpu, rho_inflated_brick_gpu, x0, x1, blockSize, stream=None):#
    '''
    Notes:
        Takes a block from the previous calculation and reduces it to a block of realspace by
      contracting out the block dimension and sticking the whole thing into a bigger array
    Arguments:
    Output:
      rho_gpu                - averaged density over the whole physical mesh
    Input:
      rho_inflated_brick_gpu - a physical-space-block-size by number-of-blocks array
      x0                     - the start of the real space brick (multi-index)
      x1                     - the end of the real space brick (multi-index)
    Optional:
      blockSize - the number of threads per block
      stream    - the serializing stream
      '''
    
    code = '''

#define n_accum 256

#define i_x blockIdx.x
#define i_y blockIdx.y
#define tid threadIdx.x
#define bn gridDim.x
#define tn blockDim.x

__global__ void accumulateScreeningAvg(   float * B,  //Big array
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
    s = rho_inflated_brick_gpu.shape
    t = rho_gpu.shape
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
    sum_funct = SourceModule(code).get_function("accumulateScreeningAvg")
    sum_funct(rho, rho_block, numpy.int32(blks),
              numpy.int32(ax), numpy.int32(ay), numpy.int32(az),
              numpy.int32(bx), numpy.int32(by), numpy.int32(bz),
              numpy.int32(xx), numpy.int32(xy), numpy.int32(xz),
              block=(blockSize, 1, 1),
              grid=(ax, ay),
              stream=stream)


def doScreeningAveragingSingleCPU(rhoAvg, rho, h, R):
    l_x = self.Lx, l_y = self.Ly, l_z = self.Lz
    rhohat = numpy.fft.fftn(numpy.array(rho, dtype=numpy.complex128))
    n_x = rho.shape[0]
    n_y = rho.shape[1]
    n_z = rho.shape[2]
    f = numpy.array(numpy.zeros(3), dtype=numpy.float64)
    rhohat_conv = numpy.zeros_like(rhohat)
    for i in range(-n_x / 2, n_x / 2):
        for j in range(-n_y / 2, n_y / 2):
            for k in range(-n_z / 2, n_z / 2):
                f[0] = (2.*pi*(1. / l_x)*i)
                f[1] = (2.*pi*(1. / l_y)*j)
                f[2] = (2.*pi*(1. / l_z)*k)
                ff = (sqrt(f[0]**2 + f[1]**2 + f[2]**2))
                Rf = (radius*ff)
                Rf2 = (Rf*Rf)
                Rf3 = (Rf*Rf2)
                #centered at zero, the convolving function is:
                #3*(-cos(R*f) / (R*f)**2 + sin(R*f) / (R*f)**3
                if (Rf == 0.):
                    rhohat_conv[i, j, k] = (rhohat[i, j, k])
                else:
                    rhohat_conv[i, j, k] = (rhohat[i, j, k]*3.*(-cos(Rf) / Rf2 + sin(Rf) / Rf3))
    rhoAvg[:] = numpy.fft.ifftn(rhohat_conv)




class ScreeningAveragingTest:
    '''Tests the screening density vs. a convolution.'''
    import matplotlib.pyplot as pyplot
    def __init__(self, Lx=1.,Ly=3.,Lz=1.,Nx=32,Ny=32,Nz=32,Rconst = 0.1)
      # the constant screening radius used in the test
      self.Rconst = Rconst
      # The number of mesh points in each direction
      self.N = numpy.array([Nx,Ny,Nz])
      #the size of the domain
      self.L = numpy.array([Lx,Ly,Lz])
      #
      self.dim = len(self.N.shape)

    @property
    def Lx(self):
        return self.L[0]
    @property
    def Ly(self):
        return self.L[1]
    @property
    def Lz(self):
        return self.L[2]

    @property
    def Nx(self):
        return self.N[0]
    @property
    def Ny(self):
        return self.N[1]
    @property
    def Nz(self):
        return self.N[2]

    def plot_data(self,grid, cut = None):
        '''Plotting function'''
        Nx = self.Nx, Ny = self.Ny, Nz = self.Nz
        # Make plot with vertical (default) colorbar
        fig = pyplot.figure()
        ax = fig.add_subplot(111)
        dim = len(grid.shape)
        if (dim == 3):
            n_grid = numpy.array(numpy.zeros((self.Nx, self.Ny)))
            if cut is None:
                cut = Nz/2
            for i in range(Nx):
                for j in range(Nx):
                    n_grid[i, j] = grid[i, j, cut]
        else: 
            n_grid = grid
        cax = ax.imshow(n_grid, interpolation='nearest')
        cb = pyplot.colorbar(cax)    

    def linear_radial_screening_radius(self,x):
        '''The screening radius is linear in the distrance from the center of the physical sample.'''
        m_x = self.Lx / 2.
        m_y = self.Ly / 2.
        m_z = self.Lz / 2.
            return (sqrt((x[0] - m_x)**2 + (x[1] - m_y)**2 + (x[2] - m_z)**2))*0.25;

    def constant_screening_radius(self, x, radius=self.Rconst):
        '''Constant screening radius'''
        return radius

    def cos3d(self,x):
        '''a simple function to test against'''
        return 1.0*cos(2.*pi*x[0])*cos(2.*pi*x[1])*cos(2.*pi*x[2])

    def cos3dhat(self,xi):
        v = self.Nx*self.Ny*self.Nz
        epsilon = 0.0001
        for xi_i in xi:
            if fabs(xi_i - 2.0*pi) < epsilon:
                v *= 0.5
            elif fabs(xi_i + 2.0*pi) < epsilon:
                v *= 0.5
            else:
                v *= 0.0
        return v
    
    def field_function_x(self,x):
        return x[0]

    def constant_field_function(self,x):
        return 1.

    def random_field_function(self,x):
        return rnd.randn()

    def run(self):
        n_x = self.Nx, n_y = self.Ny, n_z = self.Nz
        l_x = self.Lx, l_y = self.Ly, l_z = self.Lz        
        h = numpy.array([l_x / n_x, l_y / n_y, l_z / n_z])
        ff = self.cos3d
        x = numpy.zeros(3)
        rho = numpy.array(numpy.zeros((n_x, n_y, n_z)), dtype=numpy.float64)
        R = numpy.array(numpy.zeros((n_x, n_y, n_z)), dtype=numpy.float64)
        # create the physical space grids
        for i in range(n_x):
            for j in range(n_y):
                for k in range(n_z):
                    x[0] = h[0]*i 
                    x[1] = h[1]*j 
                    x[2] = h[2]*k
                    R[i, j, k] = constant_screening_radius(x)
                    rho[i,j,k] = ff(x)
                    #print homogenization_radius_function(x)

        # Test the error on the Fourier transform
        rhohat = numpy.array(numpy.zeros((n_x, n_y, n_z)), dtype=numpy.complex64)
        for k in range(-n_z / 2, n_z / 2):
            for j in range(-n_y / 2, n_y / 2):
                for i in range(-n_x / 2, n_x / 2):
                    x[0] = 2.*pi / l_x * i
                    x[1] = 2.*pi / l_y * j
                    x[2] = 2.*pi / l_z * k
                    rhohat[i, j, k] = cos3dhat(x)
        print "ifft(rhohat) test", numpy.linalg.norm(A - numpy.fft.ifftn(rhohat))
        print "fft(ifft(rho)) test", numpy.linalg.norm(A - numpy.fft.ifftn(numpy.fft.fftn(rho)))
        print "Running the screening integral tests ..."
        # print numpy.fft.fftn(rhohat)

        # Run the FFT-based averaging test
        rho_c = numpy.zeros_like(rho)
        doScreeningAveragingSingleCPU(rho_c, rho, h, R)

        # Run the GPU screening averaging test
        rho_s = numpy.zeros_like(rho)
        doScreeningAveragingSingleGPU(rho_s, rho,, R)
        # rho_t = run_reduction_example(A, B)

        print "CPU vs. GPU L^2 error:", numpy.linalg.norm(rho_c   - rho_s)
        # plot_data(rho_t)
        # plot_data(rho_s)
        # pyplot.show()
    
if __name__== "__main__":
    ScreeningAveragingTest().run()

