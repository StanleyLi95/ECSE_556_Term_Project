function [H, At, Ar, AOD, AOA, BETA, delay] = generate_channel_H_LIS(Nt,Nr,L,fs,fc,M,Num_users)
    %% Path delay
    % delay_max = 20e-9;
    % delay = delay_max*rand(1,L);
    delay = [[1.78032951094980e-08,...
              9.55821708146159e-09,...
              7.35131305345536e-09,...
              4.73002944592621e-09,...
              3.13224555779004e-09,...
              8.52959370256540e-09,...
              3.91263028442759e-10,...
              1.18337584383972e-08,...
              8.14978983666684e-09,...
              7.66654915622931e-10]];
    
    lambda_c = 3e8/fc; % wavelength
    dt = lambda_c/2;
    dr = dt;
    H = zeros(Nr,Nt,Num_users,M);

    for u=1:1:Num_users
        beta(1:L) = exp(1i*2*pi*rand(1,L));
        %%% DoA

        AoA = 2*rand(1,L) - 1;
        for m = 1 :  M
            %     f = fc;
            f(m) = fc + fs/M*(m-1-(M-1)/2);
            lambda(m) = 3e8/f(m);
            for l = 1 : L
                At(:,l,u) = 0;
                Ar(:,l,u) = array_respones(AoA(l),Nr,dr,lambda(1));
                H(:,:,u,m) = H(:,:,u,m) + beta(l)*exp(-1i*2*pi*f(m)*delay(l))* Ar(:,l,u);
            end
            H(:,:,u,m) = sqrt(Nt*Nr)*H(:,:,u,m);            
        end
        
        %% Collect Data
        AOD(u,:) = 0;
        AOA(u,:) = AoA;
        BETA(u,:) = beta;
    end
end 