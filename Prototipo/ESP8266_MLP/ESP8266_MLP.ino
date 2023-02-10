// replace with your actual file
#include <Arduino.h>
#include "mlp.h"
#define FS 40.0 //Frecuencia de muestreo (Hz)
#define T_delay (1000000/FS) //us
int a = 0, b = 0, t = 0;

void setup() {
  Serial.begin(1000000);
  while (!mlp.begin()) {
      Serial.print("Error in NN initialization: ");
      Serial.println(mlp.getErrorMessage());
  }
}

void loop() {
  a = micros();
  t = t + abs(a - b);
  b = a; 
  int c = micros();
  Serial.println(micros()-c);
  float input[32] = {0.3021206,0.41242938,0.07272727,0.00120874,0.25953459,0.26973684,0.0,0.00142821,0.39566474,0.40425532,0.2231405,0.00175069,0.55897554,0.53146853,0.43820225,0.00314321,0.44653045,0.49142857,0.20253165,0.00214088,0.35937415,0.14606742,0.05263158,0.00399403,0.40643957,0.44871795,0.30851064,0.00234997,0.21391629,0.08510638,0.0,0.00195765};
  float y_pred = mlp.predictClass(input);
  
  while ( micros() < (a + T_delay) ) {
  }
}
