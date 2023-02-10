// replace with your actual file
#include "model.h"
#include "functions.h"
#define CALIB 1 // 0 to data adquisition, 1 to enable the calibration.


void setup() {
  Serial.begin(1000000);
  analogReadResolution(bits);
  pinMode(15, INPUT_PULLUP);
  pinMode(2, OUTPUT);
  while (!mlp.begin()) {
      Serial.print("Error in NN initialization: ");
      Serial.println(mlp.getErrorMessage());
  }
  Serial.println("1\t\t2\t\t3\t\t4\t\t5\t\t6\t\t7\t\t8");
  
}

void loop() {
  a = micros();
  t = t + abs(a - b);
  //Serial.println(abs(a - b));
  b = a; 
  label = !digitalRead(15);

  if(calibCount<10*FS+1){
    calibCount++;  
  }
  if(calibCount==10*FS)
    calibFlag=true;
  
  if(calibTime==10*FS){
        calibFlag = false;
        calibTime = 0;
        calibFeaturesFlag = true;
        digitalWrite(2, 0);
        delay(200);
        for(int j = 0; j<W; j++)
          for(int i = 0; i<nChannels; i++)
        channels[i][j] = 0;
  }
  if(calibFeaturesTime==10*FS){
      calibFeaturesFlag = false;
      calibFeaturesTime = 0;
      calibFlag = false;
  }
    
  if(calibFlag){
    readAndFilter(false);
    for(int i = 0; i<nChannels; i++){
      if(abs(channels[i][W-1])>abs(channelScale[i]))
        channelScale[i] = abs(channels[i][W-1]);   
    }
    calibTime++;
  }
  else if(calibFeaturesFlag){
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
    calibFeaturesTime++;      
  }
  else{
    readAndFilter(true);
    if(count==I){
      featuresCalculate(true);
      float input[32];
      for(int i = 0; i<8; i++){
        input[i*4] = features[i][0];
        input[i*4+1] = features[i][1];
        input[i*4+2] = features[i][2];
        input[i*4+3] = features[i][3];
      }
      if(1){
      Serial.print(label);
      Serial.print(',');
      Serial.print(features[0][0]*100);
      Serial.print(',');
      Serial.print(features[0][1]*100);
      Serial.print(',');
      Serial.print(features[0][2]*100);
      Serial.print(',');
      Serial.print(features[0][3]*100);
      for(int i = 1; i<8; i++){
        for(int j = 0; j<4; j++){
          Serial.print(',');
          Serial.print(features[i][j]*100);
        }
      }
      Serial.println("");
      }
      else{
        for(int i=0; i<7; i++){
          Serial.print(wlScale[i]);
          Serial.print(',');
          Serial.print(zcScale[i]);
          Serial.print(',');
          Serial.print(sscScale[i]);
          Serial.print(',');
          Serial.print(rmsScale[i]);
          Serial.print(',');
        }
        Serial.print(wlScale[7]);
        Serial.print(','); 
        Serial.print(zcScale[7]);
        Serial.print(',');
        Serial.print(sscScale[7]);
        Serial.print(',');
        Serial.println(rmsScale[7]);
      }
      //Serial.println(features[2][2]);
      //float y_pred = mlp.predictClass(input);
      //Serial.println(y_pred);
      count =0;
    }
    count++;
  }
  digitalWrite(2, calibFlag||calibFeaturesFlag);
  
  while ( micros() < (a + T_delay) ) {
  }
}




//float features[][], float channels[], int nChannels
