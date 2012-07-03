from petsc4py import PETSc
import time
import sys

class Viz:
    '''A crude Python vizualization engine.  If your mayavi works, 
       uncomment the corresponding lines in the viewRho method to
       use that for vizualization instead.'''
    @staticmethod
    def init(v):
        # Extract sibling f with key "Field"
        f = v.visitor.getComponent("Field")
        if f is None:
         raise Exception("No Field component in the visitor shell: '"+str(v.visitor.getName()) +"'")
        # Then extract objects attached to f:
        #   a DMDA and a Vec
        da       = v.compose("mesh",f.query("mesh"))
        rho      = v.compose("rho", f.query("rho"))
        pass
 
    @staticmethod
    def viewRho(v):
        import numpy
        # Get mesh and rho
        da  = v.query("mesh")
        rho = v.query("rho").getArray()
        # Get DMDA parameters and make sure rho conforms to them
        N              = da.getSizes()
        dim            = da.getDim()
        d              = da.getDof()-1
        assert dim    == 3
        assert rho.size == N[0]*N[1]*N[2]*d
        # Reshape the Vec's data array to conform to the DMDA shape
        shape = list(N)
        shape.append(d)
        rho = rho.reshape(shape)
        # Print ll of the components of rho over the da
        # If you have mayavi, uncomment the stuff below to use a contour plot in 3D instead
        for s in range(d):
            #from enthought.mayavi import mlab as mlab
            #mlab.contour3d(rho[:,:,:,s])
            print(rho[:,:,:,s])
            time.sleep(1)
            #mlab.clf()
        time.sleep(3)



