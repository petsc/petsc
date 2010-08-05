from petsc4py import PETSc
import time

def Viz(fwk, key, stage, v):
    if v is None:
        # Create a Fwk instance to serve as the component
        v = PETSc.Fwk().create(fwk.comm)
        fwk.registerDependence("Electrolyte",key)
        return v
    if stage == "init":
        assert isinstance(v, PETSc.Fwk)
    if stage == "viewRho":
        import numpy
        # Extract a PetscObject e with key "Electrolyte"
        e = fwk.getComponent("Electrolyte")
        # Then extract objects attached to e:
        #   a DA and a Vec
        da       = e.query("mesh")
        rho      = e.query("rho").getArray()
        # Get DA parameters and make sure rho conforms to them
        N              = da.getSizes()
        dim            = da.getDim()
        d              = da.getDof()-1
        assert dim    == 3
        assert rho.size == N[0]*N[1]*N[2]*(d)
        # Reshape the Vec's data array to conform to the DA shape
        shape = list(N)
        shape.append(d)
        rho = rho.reshape(shape)
        # Plot all of the components of rho over the da
        # Use a contour plot in 3D 
        from enthought.mayavi import mlab as mlab
        for s in range(d):
            mlab.contour3d(rho[:,:,:,s])
            time.sleep(1)
            mlab.clf()
        time.sleep(3)
    if stage == "viewRhoGamma":
        import numpy
        # Extract a PetscObject e with key "Electrolyte"
        e = fwk.getComponent("Electrolyte")
        # Then extract objects attached to e:
        #   a DA and a Vec
        da       = e.query("mesh")
        rhoGamma = e.query("rhoGamma").getArray()
        # Get DA parameters and make sure rhoGamma conforms to them
        N              = da.getSizes()
        dim            = da.getDim()
        d              = da.getDof()-1
        assert dim    == 3
        assert rhoGamma.size == N[0]*N[1]*N[2]*(d+1)
        # Reshape the Vec's data array to conform to the DA shape
        shape = list(N)
        shape.append(d+1)
        rhoGamma = rhoGamma.reshape(shape)
        # Plot all of the components of rhoGamma -- rho and Gamma -- over the da
        # Use a contour plot in 3D 
        from enthought.mayavi import mlab as mlab
        for s in range(d+1):
            mlab.contour3d(rhoGamma[:,:,:,s])
            time.sleep(1)
            mlab.clf()
        time.sleep(3)
    return v
