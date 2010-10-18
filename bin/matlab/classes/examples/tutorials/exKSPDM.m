%%
%
%  Solves a linear system where the DM object manages the mesh--solver interactions
%
%   Set the Matlab path and initialize PETSc
path(path,'../../')
PetscInitialize({'-ksp_monitor'});
%%
%   Open a viewer to display PETSc objects
viewer = PetscViewer();
viewer.SetType('ascii');
%%
%   Create a DM object
da = DMDACreate2d(DM.NONPERIODIC,DM.STENCIL_BOX,4,4,1,1,1,1,0,0);
%%
%%
%   Create the linear solver, tell it the DM
ksp = KSP();
ksp.SetType('gmres');
ksp.SetDM(da);
ksp.SetFromOptions();
ksp.Solve(b,x);
x.View(viewer);
ksp.View(viewer);
%%
%   Free PETSc objects and Shutdown PETSc
%
da.Destroy();
ksp.Destroy();
viewer.Destroy();
PetscFinalize();
