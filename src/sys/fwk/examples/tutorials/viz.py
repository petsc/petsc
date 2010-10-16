from petsc4py import PETSc
from enthought.mayavi import mlab as mlab
import time

class Viz:
    @staticmethod
    def init(v):
        fwk = v.getParent()
        key = v.getName()
        fwk.registerDependence("Electrolyte",key)
        e = fwk.getComponent("Electrolyte")
        v.compose("Electrolyte", e)
        
    @staticmethod
    def viewRho(v):
        import numpy
        # Extract a PetscObject e with key "Electrolyte"
        e = v.query("Electrolyte")
        if e is None:
            raise Exception("No Electrolyte object found in the Viz framework: '"+str(v.getName()) +"'")
        # Then extract objects attached to e:
        #   a DMDA and a Vec
        da       = e.query("mesh")
        rho      = e.query("rho").getArray()
        # Get DMDA parameters and make sure rho conforms to them
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
            mlab.contour3d(rho[:,:,:,s])
            time.sleep(1)
            mlab.clf()
        time.sleep(3)
        
    @staticmethod
    def viewRhoGamma(v):
        import numpy
        # Extract a PetscObject e with key "Electrolyte"
        e = v.query("Electrolyte")
        # Then extract objects attached to e:
        #   a DMDA and a Vec
        da       = e.query("mesh")
        rhoGamma = e.query("rhoGamma").getArray()
        # Get DMDA parameters and make sure rhoGamma conforms to them
        N              = da.getSizes()
        dim            = da.getDim()
        d              = da.getDof()-1
        assert dim    == 3
        assert rhoGamma.size == N[0]*N[1]*N[2]*(d+1)
        # Reshape the Vec's data array to conform to the DMDA shape
        shape = list(N)
        shape.append(d+1)
        rhoGamma = rhoGamma.reshape(shape)
        # Plot all of the components of rhoGamma -- rho and Gamma -- over the da
        # Use a contour plot in 3D 
        for s in range(d+1):
            mlab.contour3d(rhoGamma[:,:,:,s])
            time.sleep(1)
            mlab.clf()
        time.sleep(3)


