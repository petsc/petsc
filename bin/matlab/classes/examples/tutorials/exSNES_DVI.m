%%
% Solves a nonlinear variational inequality where the user manages the mesh--solver interactions
%
% This is a translation of snes/examples/tutorials/ex8.c
%
%   Set the Matlab path and initialize PETSc
path(path,'../../')
PetscInitialize({'-snes_monitor','-ksp_monitor'});
%%
%   Open a viewer to display PETSc objects
viewer = PetscViewer();
viewer.SetType('ascii');
%%
%  Create DM to manage the grid and get work vectors
dm = PetscDMDACreate2d(PetscDM.NONPERIODIC,PetscDM.STENCIL_BOX,-4,-4,PetscObject.DECIDE,PetscObject.DECIDE,1,1);
x  = dm.CreateGlobalVector();
r  = x.Duplicate();
J  = dm.GetMatrix('aij');
%%
%  Create the nonlinear solver 
snes = PetscSNES();
snes.SetType('ls');
%%
%  Provide a function 
snes.SetFunction(r,'snesfunction',0);
type snesfunction.m
%%
%  Provide a function that evaluates the Jacobian
snes.SetJacobian(J,J,'snesjacobian',0);
type snesjacobian.m
%%
%   Set VI bounds
xl = x.Duplicate();
xb = x.Duplicate();
for i=1:length(xl)
  xl(i) = -100000000;
  xb(i) = +100000000;
end
snes.VISetVariableBounds(xl,xb);

%  Solve the nonlinear system
snes.SetFromOptions();
snes.Solve(x);
x.View(viewer);
snes.View(viewer);
%%
%   Free PETSc objects and Shutdown PETSc
%
r.Destroy();
x.Destroy();
J.Destroy();
snes.Destroy();
dm.Destroy();
viewer.Destroy();
PetscFinalize();
