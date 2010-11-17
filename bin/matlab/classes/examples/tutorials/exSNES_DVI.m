%%
% Solves a nonlinear variational inequality where the user manages the mesh--solver interactions
%
% This is a translation of snes/examples/tests/ex8.c
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
user.mx = 4;user.my = 4;
user.dm = PetscDMDACreate2d(PetscDM.NONPERIODIC,PetscDM.STENCIL_BOX,user.mx,user.my,PetscObject.DECIDE,PetscObject.DECIDE,1,1);
x  = user.dm.CreateGlobalVector();
r  = x.Duplicate();
J  = user.dm.GetMatrix('aij');
for(i = 1:length(x(:))/2)
    x(2*i-1:2*i) = ones(2,1);
end
%%
%  Set Boundary conditions
[user] = MSA_BoundaryConditions(user);
%%
%  Create the nonlinear solver 
snes = PetscSNES();
snes.SetType('vi');
%%
%  Provide a function 
snes.SetFunction(r,'snesfunction',0);
type snesfunction.m
%%
%  Provide a function that evaluates the Jacobian
snes.SetJacobian(J,J,'snesjacobian',0);
type snesjacobian.m
%%
%  Initialize guess
x.Set(1.0);
%%
%   Set VI bounds
xl = x.Duplicate();
xu = x.Duplicate();
xl.Set(-100000000);
xu.Set(100000000);

snes.VISetVariableBounds(xl,xu);

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
user.dm.Destroy();
viewer.Destroy();
PetscFinalize();
