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


%  Extract the x and y coords
N = length(cell_coords);
cells_x = cell_coords(1:2:N);
cells_y = cell_coords(2:2:N);

nv = 9;
np = 4;
nt = 2*nv+np;

% each cell becomes 4 square cells, except for pressure
N = N/(2*nv);  % N is number of cells

cellx = zeros(4,4*N);
celly = zeros(4,4*N);
cellpx = zeros(4,N);
cellpy = zeros(4,N);

for i=1:N,
  t = 4*(i-1);
  cellx(:,t+1) = [cells_x(t+1), cells_x(t+2), cells_x(t+9), cells_x(t+8)];
  celly(:,t+1) = [cells_y(t+1), cells_y(t+2), cells_y(t+9), cells_y(t+8)];
 
 cellx(:,t+2) = [cells_x(t+2), cells_x(t+3), cells_x(t+4), cells_x(t+9)];
 



  celly(:,i) = cells_y(nv*(i-1)+1 :nv*i);
end
for i=1:N,
  cellpx(:,i) = cellx(1:2:8, i);
  cellpy(:,i) = celly(1:2:8, i);
end

cell_df = cell_df + 1;

% now extract the solution at each point
cellz1 = zeros(nv,N);
cellz2 = zeros(nv,N);
cellz3 = zeros(np,N);

flag = receive(p);

while flag > .5
  % Receive the solution vector
  solution = receive(p);
  flag = receive(p);	
  for i=1:N,	
    cellz1(:,i) = solution(cell_df(   nt*(i-1)+1: 2 :nt*i- np-1 ));
    cellz2(:,i) = solution(cell_df(   nt*(i-1)+2: 2 :nt*i -np ));
    pval = solution(cell_df(nt*i-np +1:nt*i));
    cellz3(:,i) = pval;
  end
%
  figure(1)
  fill3(cellx,celly,cellz1,cellz1)
%
  figure(2)
  fill3(cellx,celly,cellz2,cellz2)
%
  figure(3)
  fill3(cellpx,cellpy,cellz3,cellz3)
end

% Close connection to MPI program
closeport(p);
