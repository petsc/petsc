H = inf(nf_const*(max(dfo(to_solve))+1),length(to_solve),SolverNumber);
for np = to_solve
    for s = 1:SolverNumber
        H(1:length(Results{s,np}.H),np,s) = Results{s,np}.H;     
    end
end
h = perf_profile(H,1e-3,0);
legend(h,{Results{1,1}.alg, Results{2,1}.alg});
saveas(gca,'perf.png');

