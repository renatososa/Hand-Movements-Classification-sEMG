#define FS 300.0 //Frecuencia de muestreo (Hz)
#define T_delay (1000000/FS)-2 //us

const float W_ms = 500;
const float I_ms = 25;

const int W = FS*W_ms/1000;
const int I = FS*I_ms/1000;
const int nChannels = 8;
const int analogPins[nChannels] = {25, 26, 32, 33, 34, 35, 36, 39};


float rmsScale[8] = {0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1};
float sscScale[8] = {1,1,1,1,1,1,1,1};
float zcScale[8] = {1,1,1,1,1,1,1,1};
float wlScale[8] = {1,1,1,1,1,1,1,1};

const float sscUmbral[8] = {0.2, 0.05, 0.5, 0.01, 0.2, 0.2, 0.01, 0.2};
const float zcUmbral[8] = {0.3, 0.3, 0.3, 0.5, 0.3, 0.3, 0.3, 0.3};
