%% Generate 8 signals with different frequencies
%% Generate the first signal
Fs = 144000; % Sampling rate of 44.1 kHz
T = 0:1/Fs:0.001;
F1 = 1900;  % Frequency of signal 1 is 1.9 kHz
Signal1 = sin(2 * pi * T * F1);

%% Generate the second signal
F2 = 2600;  % Frequency of signal 1 is 2.6 kHz
Signal2 = sin(2 * pi * T * F2);

%% Generate the third signal
F3 = 4800;  % Frequency of signal 1 is 4.8 kHz
Signal3 = sin(2 * pi * T * F3);

%% Generate the fourth signal
F4 = 10000;  % Frequency of signal 1 is 10 kHz
Signal4 = sin(2 * pi * T * F4);

%% Generate the fifth signal
F5 = 21000;  % Frequency of signal 1 is 21 kHz
Signal5 = sin(2 * pi * T * F5);

%% Generate the sixth signal
F6 = 27500;  % Frequency of signal 1 is 27.5 kHz
Signal6 = sin(2 * pi * T * F6);

%% Generate the seventh signal
F7 = 32000;  % Frequency of signal 1 is 32 kHz
Signal7 = sin(2 * pi * T * F7);

%% Generate the eighth signal
F8 = 44100;  % Frequency of signal 1 is 44.1 kHz
Signal8 = sin(2 * pi * T * F8);

figure(1)
hold on;
plot(T, Signal1)
plot(T, Signal2)
plot(T, Signal3)
plot(T, Signal4)
plot(T, Signal5)
plot(T, Signal6)
plot(T, Signal7)
plot(T, Signal8)
hold off;

figure(2)
plot(T, Signal1 + Signal2 + Signal3 + Signal4 + Signal5 + Signal6 + Signal7 + Signal8)