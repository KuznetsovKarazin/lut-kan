#include <Arduino.h>

#if CONFIG_IDF_TARGET_ESP32C3
  #include "hal/usb_serial_jtag_ll.h"
  #define USBSerial Serial
#endif

void setup() {
  USBSerial.begin(115200);
  while (!USBSerial) delay(10);
  delay(2000);
  USBSerial.println("HELLO_ESP32C3_WORKS");
}

void loop() {
  USBSerial.println("TICK");
  delay(1000);
}