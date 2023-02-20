#include "src/model.h"
#include "src/functions.h"
#define TRAINING 0 // 1 training, 0 predict.

void setup(){
  Serial.begin(1000000);
  analogReadResolution(bits);
  pinMode(15, INPUT_PULLUP);
  pinMode(2, OUTPUT);
  while (!mlp.begin()) {
      Serial.print("Error in NN initialization: ");
      Serial.println(mlp.getErrorMessage());
  }  
}

void loop(){
  a = micros();
  t = t + abs(a - b);
  b = a; 
  
  if(label&&!digitalRead(15)){
    calibButtonPress++;
  }
  if(calibButtonPress > 2*FS){
    calibButtonPress = 0;
    sample = 0;
  }
  label = !digitalRead(15);
  digitalWrite(2, calibFlag || calibFeaturesFlag);
  if(sample < 20*FS){
    calibrationRoutine(sample);
    sample++;
  }
  else{
    readAndFilter(true);
    if(count == I){
      featuresCalculate(true);
      if(TRAINING){
        dataToTrain();
      }
      else{
        normalRoutine();
      }
      count = 0;
    }
    count++;
  }
  
  while ( micros() < (a + T_delay) ) {
  }
}
