% Steering vectors
% "azimuth" is "theta" in the paper in radians
% Produce N*1 complex matrix
function a=array_respones(azimuth,N,d,lambda)
    a=[];
    for i=1:length(azimuth)
        %merge each elements by rows
        a=[a (sqrt(1/N)*exp(1i*[0:N-1]*2*pi*d*sin(azimuth(i))/lambda)).'];
    end

end