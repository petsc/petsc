
%
%  Matlab graphics routine that may be used with the main program.  This uses
%  sockets for faster visualization than writing intermediate output to a file.
%
%  Usage Instructions:
%  - First, compile Matlab programs in petsc/src/viewer/impls/matlab/ via
%      make BOPT=g openport closeport receive 
%    to generate the executables openport.mex4, closeport.mex4, and receive.mex4
%  - Be sure to set the MATLABPATH environmental variable with a command such as
%      setenv MATLABPATH $PETSC_DIR/src/viewer/impls/matlab
%    (or append this to your existing Matlab path)
%  - Then run both of the following:
%      - this script (mscript.m) in Matlab (via mscript)
%      - the example program on any machine, via
%            main -f <grid_name> -viewer_matlab_machine machine_name -matlab_graphics
%        for example,
%            main -f grids/B -viewer_matlab_machine merlin.mcs.anl.gov -matlab_graphics
%
% Open the connection to the MPI program
p = openport;        

% Receive the vertex information (x,y) coordinates of vertices
cell_coords = receive(p);
% Receive list of cell df (2*nv velocity plus np pressure  per cell)
cell_df = receive(p);

%get the total df count
dfcount = receive(p);

nv = 9;
np = 4;
nt = 2*nv+np;

M = dfcount;

df_x = zeros(M);
df_y = zeros(M);
sol_x = zeros(M);
sol_y = zeros(M);
x_index = zeros(M);
y_index = zeros(M);

cell_df = cell_df + 1;

%  Extract the x and y coords
N = length(cell_coords);
cells_x = cell_coords(1:2:N);
cells_y = cell_coords(2:2:N);
for i=1:N,
  for j = 1:nv
    x_index

for i=1:N,
  for j = 1:nv
      df_x[cell_df[nt*(i-1)+2*(j-1)+1]] = cell_coords[2*nv*(i-1) + 2*(j-1) + 1];
      df_y[cell_df[nt*(i-1)+2*(j-1)+1]] = cell_coords[2*nv*(i-1) + 2*(j-1) + 2];

      df_x[cell_df[nt*(i-1)+2*(j-1)+2]] = cell_coords[2*nv*(i-1) + 2*(j-1) + 1];
      df_y[cell_df[nt*(i-1)+2*(j-1)+2]] = cell_coords[2*nv*(i-1) + 2*(j-1) + 2];
  end
end

% Put each nv into each row 
N = N/(2*nv);
cellx = zeros(nv,N);
celly = zeros(nv,N);
cellpx = zeros(np,N);
cellpy = zeros(np,N);

for i=1:N,
  cellx(:,i) = cells_x(nv*(i-1)+1:nv*i);
  celly(:,i) = cells_y(nv*(i-1)+1 :nv*i);
end
for i=1:N,
  cellpx(:,i) = cellx(1:2:8, i);
  cellpy(:,i) = celly(1:2:8, i);
end
% now extract the solution at each point
cellz3 = zeros(np,N);

flag = receive(p);

while flag > .5
  % Receive the solution vector
  solution = receive(p);
  flag = receive(p);	

% pressure
  for i=1:N,	
    pval = solution(cell_df(nt*i-np +1:nt*i));
    cellz3(:,i) = pval;
  end
  figure(3)
  fill3(cellpx,cellpy,cellz3,cellz3)

% velocity field
  for i=1:N,
   
    sol


end

% Close connection to MPI program
closeport(p);
