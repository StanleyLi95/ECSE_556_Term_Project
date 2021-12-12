
clear;  close all;   clc;


%% ----------------------------- System Parameters -------------------------
% Num_bands = 3;   % Number of subcarriers
M = 64; % number of Tx antennas.
% M2 = 2^8;
L = 100; % number of LIS antennas. L=100;
K = 8; % Number of users
% T = 1024;
Ns = 1; % number of data streams.
% P = L;
NtRF = 10;%opts.Ns*opts.Num_users; % number of paths at Tx.
NrRF = 10;%opts.Ns; % number of paths at Rx.
% Ns = NtRF/opts.Num_users; % number of paths per user
train_size = 100; %Training size 

opts.Num_paths = 10; %Number of channel paths
opts.fc = 60*10^9; %carrier frequency
opts.BW = 4*10^9; %bandwidth
opts.fs = opts.BW;  % sampling frequency  fs=2M(BW)
opts.selectOutputAsPhases = 1;
opts.snr_param = [40]; % SNR in dB.
opts.Nreal = 30; % number of realizations.
opts.Nchannels = 10; % number of channel matrices for the input data, make it 100 for better sampling the data space.
opts.fixedUsers = 0;
opts.fixedChannelGain = 0;

opts.noiseLevelHdB_CE = [ 25 30 40 50]; % dB.
opts.inputH = 1;
opts.inputRy = 0;
timeGenerate = tic; %启动秒表计时器
rng(4096); %使用种子4096初始化梅森旋转生成器

Num_paths = opts.Num_paths;
Nch = opts.Nchannels;
Nreal = opts.Nreal;
N = Nreal*Nch*K;

Z = repmat(struct('channel_dc',zeros(M,1,K)),1,N ); %重复数组副本, initiate direct channels 

snr = db2pow(opts.snr_param);

F = dftmtx(M);% DFT_R = dftmtx(Nr);

jjHB = 1;
jjCE = 1;
jjA2 = 1;
X = eye(M); % pilot signal
X2 = eye(M*L);
V = eye(L); % reflect beamforming data.
doaMismatchList = linspace(0,10,Nch); %等差数列

%% Process
for nch = 1:Nch
    %% Generate channels
        % H, BS and LIS
        [H, At, Ar, DoA, AoA, beta, delay] = generate_channel_H(L,M,Num_paths,opts.fs,opts.fc,1,1);
        paramH(nch,1).At = At;
        paramH(nch,1).Ar = Ar;
        paramH(nch,1).DoA = DoA;
        paramH(nch,1).AoA = AoA;
        paramH(nch,1).beta = beta;
        % H_LIS_G, LIS and Users 
        [h_lis, At, Ar, DoA, AoA, beta, delay] = generate_channel_H_LIS(1,L,Num_paths,opts.fs,opts.fc,1,K);
        paramH_LIS(nch,1).At = At;
        paramH_LIS(nch,1).Ar = Ar;
        paramH_LIS(nch,1).DoA = DoA;
        paramH_LIS(nch,1).AoA = AoA;
        paramH_LIS(nch,1).beta = beta;
        % H_DC, BS and Users
        [h_dc, At, Ar, DoA, AoA, beta, delay] = generate_channel_H_LIS(1,M,Num_paths,opts.fs,opts.fc,1,K); 
        paramH_DC(nch,1).At = At;
        paramH_DC(nch,1).Ar = Ar;
        paramH_DC(nch,1).DoA = DoA;
        paramH_DC(nch,1).AoA = AoA;
        paramH_DC(nch,1).beta = beta;

   %% TEST 
   % cascaded channel.
    G = zeros(M,L,K);
    for kk = 1:K
        G(:,:,kk) = H* diag(h_lis(:,1,kk));
    end
    
    %Normalization 
    %h_lis = h_lis/norm(h_lis(:)); 
    %G = G/norm(G(:)); 
    %h_dc = h_dc/norm(h_dc(:)); 

    %% Channel estimation
    timeGenerate = tic;
    for nr = 1:Nreal
        snrIndex_CE = ceil(nr/(Nreal/size(opts.noiseLevelHdB_CE,2)));
%         snrChannel = opts.noiseLevelHdB_CE(snrIndex_CE);
        snrChannel = -10;
       
        S = 1/sqrt(2)*(randn(K,M) + 1i*randn(K,M));

        for kk = 1:K % number of users.
            y_dc(kk,:) = awgn( h_dc(:,1,kk)'*X, snrChannel,'measured'  ); % direct channel data.

            h_dc_e(:,kk) = (y_dc(kk,:)*pinv(X))'; % direct channel LS.

            vG = []; h_dc_kron = [];
            for p = 1:L % for each LIS components. estimate cascaded channel
                v = V(:,p);
                vG = [vG v'*G(:,:,kk)'];
                h_dc_kron = [h_dc_kron h_dc(:,1,kk)'];
            end
            
            y_cc(:,:,kk) = reshape(awgn( (h_dc_kron + vG )*X2  ,snrChannel,'measured'),[M,L]); % cascaded channel data.

            
            %% test.

            %%
%             H*V*h_lis(:,1,kk) - G(:,:,kk)*diag(V)
            
            R_dc(:,:,nr,kk) = reshape(y_dc(kk,:),[sqrt(M) sqrt(M)]); % (direct channel) data to feed ChannelNet_k
            R_cc(:,:,nr,kk) = y_cc(:,:,kk); % (cascaded channel) data to feed ChannelNet_k

        end
%         toc(timeGenerate)
    end
    %% Channel Estimation. Training data for A3
    %     jj = 1;
    for kk = 1:K % input-output pair of the DL model.K--the user number
        for nr = 1:Nreal%realization number
            CENet{1,1}.X_dc(:,:,1,jjCE) = real(R_dc(:,:,nr,kk)); % input.
            CENet{1,1}.X_dc(:,:,2,jjCE) = imag(R_dc(:,:,nr,kk)); % input.
            CENet{1,1}.X_cc(:,:,1,jjCE) = real(R_cc(:,:,nr,kk)); % input.
            CENet{1,1}.X_cc(:,:,2,jjCE) = imag(R_cc(:,:,nr,kk)); % input.

            channel_dc = h_dc(:,1,kk); % output. dc
            channel_cc = G(:,:,kk);% output. cc
            CENet{1,1}.Y_dc(jjCE,:) = [real(channel_dc(:)); imag(channel_dc(:))]; % output.
            CENet{1,1}.Y_cc(jjCE,:) = [real(channel_cc(:)); imag(channel_cc(:))]; % output.
            
            Z(1,jjCE).h_dc = h_dc;
            Z(1,jjCE).G = G;
            jjCE = jjCE + 1;
            
            keepIndex(jjCE) = [nch]; 
        end
    end
    %%
    nch
end % nch

timeGenerate = toc(timeGenerate);

save('Channel_Estimation_Net_10.mat_-10dB', 'CENet', '-v7.3');

