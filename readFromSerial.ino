void setup() {
  // initialize serial communication:
  Serial.begin(9600); 
   // initialize the LED pins:
      for (int thisPin = 2; thisPin < 6; thisPin++) {
        pinMode(thisPin, OUTPUT);
      } 
      
      for (int thisPin = 2; thisPin < 6; thisPin++) {
        digitalWrite(thisPin, HIGH);
      }
      
}

void loop() {
  // read the sensor:
  if (Serial.available() > 0) {
    int inByte = Serial.read();
    // do something different depending on the character received.  
    // The switch statement expects single number values for each case;
    // in this exmaple, though, you're using single quotes to tell
    // the controller to get the ASCII value for the character.  For 
    // example 'a' = 97, 'b' = 98, and so forth:

   // car controls 
   // 2 = forward
   // 3 = backward
   // 4 = left
   // 5 = right
   
    switch (inByte) {
      
    case 'a':    //forward
      // turn all the LEDs off:
//      for (int thisPin = 2; thisPin < 6; thisPin++) {
//        digitalWrite(thisPin, HIGH);
//      }
      for (int thisPin = 4; thisPin < 6; thisPin++) {
        digitalWrite(thisPin, HIGH);
      }
      digitalWrite(2, LOW);
      break;
      
    case 'b':    //backward
//      for (int thisPin = 2; thisPin < 6; thisPin++) {
//        digitalWrite(thisPin, HIGH);
//      }
      for (int thisPin = 4; thisPin < 6; thisPin++) {
        digitalWrite(thisPin, HIGH);
      }
      digitalWrite(3, LOW);
      break;
      
    case 'c':    //forward left
//      for (int thisPin = 2; thisPin < 6; thisPin++) {
//        digitalWrite(thisPin, HIGH);
//      }
      digitalWrite(5, HIGH);
//      delay(50);
      digitalWrite(4, LOW);
      delay(100);
      digitalWrite(2, LOW);
      break;
      
    case 'd':    //forward right
//      for (int thisPin = 2; thisPin < 6; thisPin++) {
//        digitalWrite(thisPin, HIGH);
//      }
      digitalWrite(4, HIGH);
//      delay(50);
      digitalWrite(5, LOW);
      delay(100);
      digitalWrite(2, LOW);
      break;
    case 'e':    //backward left
//      for (int thisPin = 2; thisPin < 6; thisPin++) {
//        digitalWrite(thisPin, HIGH);
//      }
      digitalWrite(5, HIGH);
      digitalWrite(4, LOW);
      delay(100);
      digitalWrite(3, LOW);
      break;
      
    case 'f':    //backward right
//      for (int thisPin = 2; thisPin < 6; thisPin++) {
//        digitalWrite(thisPin, HIGH);
//      }
      digitalWrite(4, HIGH);
      digitalWrite(5, LOW);
      delay(100);
      digitalWrite(3, LOW);
      break;
    default:
      // turn all the LEDs off:
      for (int thisPin = 2; thisPin < 6; thisPin++) {
        digitalWrite(thisPin, HIGH);
      }
     }
     delay(400);
     for (int thisPin = 2; thisPin < 4; thisPin++) {
       digitalWrite(thisPin, HIGH);
//     for (int thisPin = 2; thisPin < 6; thisPin++) {
//       digitalWrite(thisPin, HIGH);
     //delay(100);
    } 
  }
}

