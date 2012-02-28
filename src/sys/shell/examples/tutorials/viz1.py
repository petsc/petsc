from petsc4py import PETSc
import sys,time


class Viz:
    '''A crude Python vizualization engine.  If your mayavi works, 
       uncomment the corresponding lines in the viewRho method to
       use that for vizualization instead.'''
    @staticmethod
    def init(v):
        pass

    @staticmethod
    def viewRho(v):
        import numpy
        # Then extract objects attached to v:
        #   a DA "mesh" and a Vec "rho"
        da       = v.query("mesh")
        rho      = v.query("rho").getArray()
        # Get DA parameters and make sure rho conforms to them
        N              = da.getSizes()
        dim            = da.getDim()
        d              = da.getDof()-1
        assert dim    == 3
        assert rho.size == N[0]*N[1]*N[2]*(d)
        # Reshape the Vec's data array to conform to the DMDA shape
        shape = list(N)
        shape.append(d)
        rho = rho.reshape(shape)
        # Plot all of the components of rho over the da
        # Use a contour plot in 3D 
        for s in range(d):
            #from enthought.mayavi import mlab as mlab
            #mlab.contour3d(rho[:,:,:,s])
            sys.stdout.write(str(rho[:,:,:,s]))
            sys.stdout.flush()
            time.sleep(1)
            #mlab.clf()
        time.sleep(3)



