#include "filters.h"
#include "globalVariables.h"

const float low_cutoff_freq   = 200.0;  //Cutoff frequency in Hz
const float high_cutoff_freq   = 200.0;  //Cutoff frequency in Hz
const float sampling_time = 1/FS; //Sampling time in seconds.
IIR::ORDER  order  = IIR::ORDER::OD3; // Order (OD1 to OD4)

Filter f(low_cutoff_freq, sampling_time, order, IIR::TYPE::HIGHPASS);
Filter f1(low_cutoff_freq, sampling_time, order, IIR::TYPE::HIGHPASS);
Filter f2(low_cutoff_freq, sampling_time, order, IIR::TYPE::HIGHPASS);
Filter f3(low_cutoff_freq, sampling_time, order, IIR::TYPE::HIGHPASS);
Filter f4(low_cutoff_freq, sampling_time, order, IIR::TYPE::HIGHPASS);
Filter f5(low_cutoff_freq, sampling_time, order, IIR::TYPE::HIGHPASS);
Filter f6(low_cutoff_freq, sampling_time, order, IIR::TYPE::HIGHPASS);
Filter f7(low_cutoff_freq, sampling_time, order, IIR::TYPE::HIGHPASS);
Filter fHP(high_cutoff_freq, sampling_time, order, IIR::TYPE::LOWPASS);
Filter fHP1(high_cutoff_freq, sampling_time, order, IIR::TYPE::LOWPASS);
Filter fHP2(high_cutoff_freq, sampling_time, order, IIR::TYPE::LOWPASS);
Filter fHP3(high_cutoff_freq, sampling_time, order, IIR::TYPE::LOWPASS);
Filter fHP4(high_cutoff_freq, sampling_time, order, IIR::TYPE::LOWPASS);
Filter fHP5(high_cutoff_freq, sampling_time, order, IIR::TYPE::LOWPASS);
Filter fHP6(high_cutoff_freq, sampling_time, order, IIR::TYPE::LOWPASS);
Filter fHP7(high_cutoff_freq, sampling_time, order, IIR::TYPE::LOWPASS);

float ssc(float datos[], int L, float umbral);
float zc(float datos[], int L, float umbral);
float wl(float datos[], int L);
float rms(float datos[], int L);
void barrelUpdate();
void featuresCalculate(bool scale);
void readAndFilter(bool scale);
void calibrationRoutine(int sample);
void dataToTrain();
void normalRoutine();

void readAndFilter(bool scale){
  barrelUpdate();
  for(int i = 0; i<nChannels; i++)
    channels_raw[i] = analogRead(analogPins[i]);
  channels[0][W-1] = fHP.filterIn(f.filterIn(channels_raw[0]));
  channels[1][W-1] = fHP1.filterIn(f1.filterIn(channels_raw[1]));
  channels[2][W-1] = fHP2.filterIn(f2.filterIn(channels_raw[2]));
  channels[3][W-1] = fHP3.filterIn(f3.filterIn(channels_raw[3]));
  channels[4][W-1] = fHP4.filterIn(f4.filterIn(channels_raw[4]))*0;
  channels[5][W-1] = fHP5.filterIn(f5.filterIn(channels_raw[5]));
  channels[6][W-1] = fHP6.filterIn(f6.filterIn(channels_raw[6]));
  channels[7][W-1] = fHP7.filterIn(f7.filterIn(channels_raw[7]));
  if(scale)
    for(int i = 0; i<nChannels; i++)
      channels[i][W-1] = channels[i][W-1]/channelScale[i];
}

void barrelUpdate(){
  for(int i = 0; i<W-1;i++)
    for(int j = 0; j<nChannels; j++)
      channels[j][i] = channels[j][i+1]; 
}
  
void featuresCalculate(bool scale){
  for(int i=0; i<nChannels; i++){
    features[i][0] = wl(channels[i], W);
    features[i][1] = zc(channels[i], W, zcUmbral[i]);
    features[i][2] = ssc(channels[i], W, sscUmbral[i]);
    features[i][3] = rms(channels[i], W);    
  }
  if(scale){
    for(int i=0; i<nChannels; i++){
      features[i][0] = features[i][0]/wlScale[i];
      features[i][1] = features[i][1]/zcScale[i];
      features[i][2] = features[i][2]/sscScale[i];
      features[i][3] = features[i][3]/rmsScale[i];    
    }
  } 
}

float ssc(float datos[], int L, float umbral){
  float out = 0;
  for(int i = 0; i<L-2; i++){
    if( (datos[i+1]-datos[i])*(datos[i+1] - datos[i+2])>= umbral)
      out++;
  }
  return out;
}

float zc(float datos[], int L, float umbral){
  float out = 0;
  for(int i = 0; i<L-1; i++){
    if( (datos[i] * datos[i+1] < 0) && (abs(datos[i] - datos[i+1])>umbral))
      out++;
  }
  return out;
}

float wl(float datos[], int L){
  float out = 0;
  for(int i = 0; i<L-1; i++){
    out += abs(datos[i+1]-datos[i]);
  }
  return out;
}

float rms(float datos[], int L){
  float out = 0;
  for(int i = 0; i<L-1; i++){
    out += pow(datos[i],2);
  }
  return (out/L);
}


void calibrationRoutine(int sample){
  if(sample<10*FS){
    readAndFilter(false);
    for(int i = 0; i<nChannels; i++)
      if(abs(channels[i][W-1])>abs(channelScale[i]))
        channelScale[i] = abs(channels[i][W-1]);   
  }
  else if(sample==10*FS){
    digitalWrite(2, 0);
    delay(200);
    for(int j = 0; j<W; j++)
      for(int i = 0; i<nChannels; i++)
        channels[i][j] = 0;
  }
  else{
    readAndFilter(true);
    if(count==I){
      featuresCalculate(false);
      for(int i = 0; i<nChannels; i++){
        if(features[i][0]>wlScale[i])
          wlScale[i] = features[i][0];
        if(features[i][1]>zcScale[i])
          zcScale[i] = features[i][1];
        if(features[i][2]>sscScale[i])
          sscScale[i] = features[i][2];
        if(features[i][3]>rmsScale[i])
          rmsScale[i] = features[i][3];
      }
      count = 0;
    }
    count++;
  }
  
}

void dataToTrain(){
  Serial.print(label);
  for(int i = 0; i<8; i++){
    for(int j = 0; j<4; j++){
      Serial.print(',');
      Serial.print(features[i][j]*100);
    }
  }
  Serial.println("");
}

void normalRoutine(){
  float input[32];
  for(int i = 0; i<8; i++){
    for(int j = 0; j<4; j++)
      input[i*4+j] = features[i][j];
  }
  int y_pred = mlp.predictClass(input);
  Serial.println(movs[y_pred]);
}