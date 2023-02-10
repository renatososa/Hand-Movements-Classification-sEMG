#include "constants.h"

bool calibFlag = false;
bool calibFeaturesFlag = false;
int channelScale[nChannels] = {2561,1598,1582,2397,2361,2277,1993,1825};
int t = 0;
int count = 0;
unsigned long b, a = 0;
int bits = 12;
float channels[nChannels][W];
float features[8][4] = {0};

int label = 0; 
int channels_raw[nChannels];
bool calBootonPress = false;
int calibTime = 0;
int calibFeaturesTime = 0;
int calibCount = 0;
