%%
% Solves a nonlinear variational inequality where the user manages the mesh--solver interactions
%
% This is a translation of snes/examples/tests/ex10.c
%
%   Set the Matlab path and initialize PETSc
path(path,'../../')
PetscInitialize({'-snes_monitor','-ksp_monitor'});
%%
%   Open a viewer to display PETSc objects
viewer = PetscViewer();
viewer.SetType('ascii');
%%
%   Create work vectors to manage the grid
user.mx = 10;user.my = 10;
N = user.mx*user.my;
x  = PetscVec();
x.SetType('seq');
x.SetSizes(N,N);
r  = x.Duplicate();
J  = PetscMat();
J.SetType('aij');
J.SetSizes(N,N,N,N);

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
snes.SetFunction(r,'snesdvi2_function',user);


%%
%  Set minimum surface area problem jacobian routine
snes.SetJacobian(J,J,'snesdvi2_jacobian',user);

%%
%  Set solution monitoring routine
snes.MonitorSet('snesdvi2_monitor',user);
figure(1),clf;figure(2),clf;
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
x.View(viewer);
snes.View(viewer);

%%
%   Free PETSc objects and Shutdown PETSc
%
user.bottom.Destroy();
user.top.Destroy();
user.right.Destroy();
user.left.Destroy();
r.Destroy();
x.Destroy();
J.Destroy();
snes.Destroy();
viewer.Destroy();
PetscFinalize();
