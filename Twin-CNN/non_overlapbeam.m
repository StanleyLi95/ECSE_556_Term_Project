%To create non-overlap paths
function index = non_overlapbeam(N,L,K)
    x = randperm(N); % disorganize the number from 1 to N
    y = x(1:L*K); % N <= L*K
    index = reshape(y,L,K); %rearrange to a matrix of L*K
end
