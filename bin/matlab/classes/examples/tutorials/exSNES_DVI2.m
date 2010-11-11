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
%  Create DM to manage the grid and get work vectors
user.mx = 4;user.my = 4;
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
%  Initialize guess
[x] = MSA_InitialGuess(user,x);
%%
%  Create the nonlinear solver 
snes = PetscSNES();
snes.SetType('vi');
%%
%  Provide a function 
snes.SetFunction(r,'snesdvi2_function',user);
type snesdvi2_function.m
%%
%  Provide a function that evaluates the Jacobian
snes.SetJacobian(J,J,'snesdvi2_jacobian',user);
type snesdvi2_jacobian.m

%%
%   Set VI bounds
xl = x.Duplicate();
xu = x.Duplicate();
xl.Set(-100000000);
xu.Set(1000000000);

snes.VISetVariableBounds(xl,xu);    

%%
%  Solve the nonlinear system
snes.SetFromOptions();
snes.Solve(x);
x.View(viewer);
snes.View(viewer);
%%
% Create surface plot
x_sol = reshape(x(:),user.mx,user.my);
surf(x_sol);
% hold on
% bdry = [user.bottom(:)',user.top(:)',user.right(:)',user.left(:)'];
% surf(bdry);

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
