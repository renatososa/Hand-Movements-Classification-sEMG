#include "constants.h"

bool calibFlag = false;
bool calibFeaturesFlag = false;
float channelScale[nChannels] = {1,1,1,1,1,1,1,1};
int lastMovsDetect[3] = {0};
int t = 0;
int count = 0;
unsigned long b, a = 0;
int bits = 12;
float channels[nChannels][W];
float features[8][4] = {0};

int label = 0; 
int channels_raw[nChannels];
int calibButtonPress = 0;
int sample = 20*FS;
