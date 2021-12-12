clear;  close all;   clc;
% addpath('./AltMin/Narrowband');
% addpath('./TwoStage vs ZF vs ACE vs AS_final - two-bit');
% addpath(genpath('./AltMin'));
CENet=load('Channel_Estimation_Net_10_10dB.mat');
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
opts.snr_param = [0]; % SNR in dB.
opts.Nreal = 30; % number of realizations.
opts.Nchannels = 5; % number of channel matrices for the input data, make it 100 for better sampling the data space.
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

for kkt = 1:train_size



    
    fprintf(2,['Train ChannelNet_DC{' num2str(kkt) '} \n'])
    [CEnet{1,kkt}.net_dc] = train_ChannelNet(CENet.CENet{1,kkt}.X_dc,CENet.CENet{1,kkt}.Y_dc,0.0005);
    fprintf(2,['Train ChannelNet_CC{' num2str(kkt) '} \n'])
    [CEnet{1,kkt}.net_cc] = train_ChannelNet(CENet.CENet{1,kkt}.X_cc,CENet.CENet{1,kkt}.Y_cc,0.0002); % 0000011

    
    
end

