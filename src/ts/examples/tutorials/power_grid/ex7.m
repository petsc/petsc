%
%  Loads the output from ex7 (option -ts_monitor_solution_binary) and puts it into a useful
%  format for visualizing. Removes the first 49 y grid lines since they have noise from the initial
%  conditions. For example:
%
%    contourf(A{49})   and contourf(B{49})
%
A = PetscBinaryRead('binaryoutput','cell',10000);
l = size(A); l = l(2);
n = size(A{1}); n = sqrt(n(1));
for i=1:l; A{i} = reshape(A{i},n,n); end
B = A;

for i=1:l
        for j=1:n; A{i}(:,j) = min(A{i}(:,j)); end
        for j=1:n; B{i}(:,j) = B{i}(:,j) - min(B{i}(:,j)); end
end

for i=1:l
        A{i} = A{i}(:,50:n);
        B{i} = B{i}(:,50:n);
end
