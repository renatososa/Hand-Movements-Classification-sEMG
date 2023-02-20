#define FS 400.0 //Frecuencia de muestreo (Hz)
#define T_delay (1000000/FS)-2 //us

const float W_ms = 1000;
const float I_ms = 50;
const String movs[9] = {"Reposo", "Menique", "Anular", "Medio", "Indice", "Pulgar", "Pronacion", "Supinacion", "Cerrada"};

const int W = FS*W_ms/1000;
const int I = FS*I_ms/1000;
const int nChannels = 8;
const int analogPins[nChannels] = {25, 26, 32, 33, 34, 35, 36, 39};


float rmsScale[8] = {0.1681,0.0543,0.0637,0.1285,0.0500,0.1120,0.0806,0.0500};
float sscScale[8] = {50.00,43.00,4.00,73.00,1.00,51.00,61.00,20.00};
float zcScale[8] = {79.00,45.00,39.00,41.00,1.00,71.00,32.00,39.00};
float wlScale[8] = {70.11,40.61,37.02,56.82,1.00,70.64,34.85,41.23};

const float sscUmbral[8] = {0.2, 0.05, 0.5, 0.01, 0.2, 0.2, 0.01, 0.2};
const float zcUmbral[8] = {0.3, 0.3, 0.3, 0.5, 0.3, 0.3, 0.3, 0.3};
