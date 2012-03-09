%%
% Solves a nonlinear variational inequality where the user manages the mesh--solver interactions
%
% This is a translation of snes/examples/tests/ex8.c
%
%   Set the Matlab path and initialize PETSc
more on
figure(1),clf;figure(2),clf;
path(path,'../../')
PetscInitialize({'-snes_vi_monitor','-ksp_monitor','-snes_vi_type','ss'});
%%
%  Create DM to manage the grid and get work vectors
user.mx = 10;user.my = 10;
user.dm = PetscDMDACreate2d(PetscDM.NONPERIODIC,PetscDM.STENCIL_BOX,user.mx,user.my,Petsc.DECIDE,Petsc.DECIDE,1,1);
x  = user.dm.CreateGlobalVector();
r  = x.Duplicate();
J  = user.dm.CreateMatrix('aij');

%%
%  Set Boundary conditions
[user] = MSA_BoundaryConditions(user);
%%
%  Set initial guess
[x] = MSA_InitialGuess(user,x);
%%
%  Create the nonlinear solver 
snes = PetscSNES();
snes.SetType('vi');

%%
%  Set minimum surface area problem function routine
snes.SetFunction(r,'snesdvi_function',user);
type snesdvi_function

%%
%  Set minimum surface area problem jacobian routine
snes.SetJacobian(J,J,'snesdvi_jacobian',user);
type snesdvi_jacobian

%%
%  Set solution monitoring routine
snes.MonitorSet('snesdvi_monitor',user);
type snesdvi_monitor

%%
%   Set VI bounds
xl = x.Duplicate();
xu = x.Duplicate();
xl.Set(-10000000);
xu.Set(100000000);
snes.VISetVariableBounds(xl,xu);    

%%
%  Solve the nonlinear system
snes.SetFromOptions();
snes.Solve(x);
x.View;
snes.View;

%%
%   Free PETSc objects and shutdown PETSc
%
user.bottom.Destroy();
user.top.Destroy();
user.right.Destroy();
user.left.Destroy()
r.Destroy();
x.Destroy();
xl.Destroy();
xu.Destroy();
user.dm.Destroy();
J.Destroy();
snes.Destroy();
PetscFinalize();
more off
