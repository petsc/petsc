function bscript(np,options)
%
%  Matlab graphics routine that may be used with the main program.  This uses
%  sockets for faster visualization than writing intermediate output to a file.
%
%  Usage Instructions:
%  - Be sure to set the MATLABPATH environmental variable with a command such as
%      setenv MATLABPATH $PETSC_DIR/src/sys/viewer/impls/socket/matlab
%    (or append this to your existing Matlab path)
%
%    Run bscript(np,'extra options')
%      np - number of processors to run parallel program
%      any extra options to the program as a single text string
%
%
if (nargin < 2) options = ''; end;
if (nargin < 1) np      = 1; end;

% Start the parallel program
launch(['main -matlab_graphics ' options],np);

% Open the connection to the MPI program
p = openport;        

% Receive the vertex information (x,y) coordinates of vertices
cell_coords = receive(p);

% Receive list of cell vertices
cell_df = receive(p);

%  Extract the x and y coords
N = length(cell_coords);
cells_x = cell_coords(1:2:N);
cells_y = cell_coords(2:2:N);

nv = 4;

% Put each nv into each row 
N = N/(2*nv);
cellx = zeros(nv,N);
celly = zeros(nv,N);

for i=1:N,
  cellx(:,i) = cells_x(nv*(i-1)+1:nv*i);
  celly(:,i) = cells_y(nv*(i-1)+1 :nv*i);
end
cell_df = cell_df + 1;

% now extract the solution at each point
cellz = zeros(nv,N);
  
% Receive the solution vector
solution = receive(p);
for i=1:N,	
    cellz(:,i) = solution(cell_df( nv*(i-1)+1 :nv*i ));
end
%
figure(1);
fill3(cellx,celly,cellz,cellz);

% Close connection to MPI program
closeport(p);
