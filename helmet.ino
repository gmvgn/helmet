/*****************************************************************
Garrett Glissmann
*****************************************************************/

// The SFE_LSM9DS1 library requires both Wire and SPI be
// included BEFORE including the 9DS1 library.
#include <Wire.h>
#include <SPI.h>
#include <SparkFunLSM9DS1.h>
#include <Adafruit_NeoPixel.h>

//////////////////////////
// LSM9DS1 Library Init //
//////////////////////////
// Use the LSM9DS1 class to create an object.
LSM9DS1 imu;

///////////////////////
// Example I2C Setup //
///////////////////////
// SDO_XM and SDO_G are both pulled high, so our addresses are:
#define LSM9DS1_M  0x1E // Would be 0x1C if SDO_M is LOW
#define LSM9DS1_AG  0x6B // Would be 0x6A if SDO_AG is LOW

////////////////////////////
// Sketch Output Settings //
////////////////////////////
#define PRINT_SPEED 100 // 100 ms between reads
static unsigned long lastPrint = 0; // Keep track of print time

// Earth's magnetic field varies by location. Add or subtract 
// a declination to get a more accurate heading. Calculate 
// your's here:
// http://www.ngdc.noaa.gov/geomag-web/#declination
#define DECLINATION -8.58 // Declination (degrees) in Boulder, CO.

///////////////////////
// NeoPixel Settings //
///////////////////////
#define NEO_PIN 6
// Parameter 1 = number of pixels in strip
// Parameter 2 = Arduino pin number (most are valid)
// Parameter 3 = pixel type flags, add together as needed
Adafruit_NeoPixel strip = Adafruit_NeoPixel(12, NEO_PIN, NEO_GRB + NEO_KHZ800);

////////////////////
// State Settings //
////////////////////

// Signal states
bool left_signal = false;
bool right_signal = false;
bool brake_signal = false;

// TODO: fill in
int left_i = {};
float left_weights = {};
float left_intercept = 0;

int right_i = {};
float right_weights = {};
float right_intercept = 0;

int brake_i = {};
float brake_weights = {};
float brake_intercept = 0;

void setup() 
{  
  Serial.begin(115200);
  
  // Before initializing the IMU, there are a few settings
  // we may need to adjust. Use the settings struct to set
  // the device's communication mode and addresses:
  imu.settings.device.commInterface = IMU_MODE_I2C;
  imu.settings.device.mAddress = LSM9DS1_M;
  imu.settings.device.agAddress = LSM9DS1_AG;
  // The above lines will only take effect AFTER calling
  // imu.begin(), which verifies communication with the IMU
  // and turns it on.
  if (!imu.begin())
  {
    Serial.println("Failed to communicate with LSM9DS1.");
    Serial.println("Double-check wiring.");
    Serial.println("Default settings in this sketch will " \
                  "work for an out of the box LSM9DS1 " \
                  "Breakout, but may need to be modified " \
                  "if the board jumpers are.");
    while (1)
      ;
  }
  // Start NeoPixel strip
  strip.begin();
  strip.show(); // Initialize all pixels to 'off'  
}

void loop()
{
  // Update the sensor values whenever new data is available
  if ( imu.gyroAvailable() )
  {
    // To read from the gyroscope,  first call the
    // readGyro() function. When it exits, it'll update the
    // gx, gy, and gz variables with the most current data.
    imu.readGyro();
  }
  if ( imu.accelAvailable() )
  {
    // To read from the accelerometer, first call the
    // readAccel() function. When it exits, it'll update the
    // ax, ay, and az variables with the most current data.
    imu.readAccel();
  }
  if ( imu.magAvailable() )
  {
    // To read from the magnetometer, first call the
    // readMag() function. When it exits, it'll update the
    // mx, my, and mz variables with the most current data.
    imu.readMag();
  }

  if ((lastPrint + PRINT_SPEED) < millis())
  {
    // Sensor readings
    float data[10];
    getSensorReadings(data);

    left_signal = lin_decision(data, &left_weights, &left_i, left_intercept, 7);
    right_signal = lin_decision(data, &right_weights, &right_i, right_intercept, 7);
    brake_signal = lin_decision(data, &brake_weights, &brake_i, brake_intercept, 7);
    
    lastPrint = millis(); // Update lastPrint time
  }

  // TODO: Handle NeoPixels
  uint32_t c = strip.Color(255, 0, 0);
  for (uint16_t i = 0; i < strip.numPixels(); i++) {
    strip.setPixelColor(i, c);
  }
  strip.show();
}

bool lin_decision(float readings[], float weights[], int indexes[], float intercept, int len)
{
  float s = intercept;
  for (int i = 0; i < len; i++) {
    // Sensor value * weight
    s += readings[indexes[i]] * weights[i];
  }
  bool b = false;
  if (s >= 0) {
    b = true;
  }
  return b;
}

void getSensorReadings(float data[])
{
  // Acceleration
  data[0] = imu.calcAccel(imu.ax);
  data[1] = imu.calcAccel(imu.ay);
  data[2] = imu.calcAccel(imu.az);
  // Gyro
  data[3] = imu.calcGyro(imu.gx);
  data[4] = imu.calcGyro(imu.gy);
  data[5] = imu.calcGyro(imu.gz);
  // Attitude
  setAttitudeReadings(
    data,
    imu.ax, imu.ay, imu.az,
    -imu.my, -imu.mx, imu.mz
  );
  // Additional acceleration value
  data[9] = sqrt(data[0] * data[0] + data[1] * data[1] + data[2] * data[2]);
}

// Calculate pitch, roll, and heading.
// Pitch/roll calculations take from this app note:
// http://cache.freescale.com/files/sensors/doc/app_note/AN3461.pdf?fpsp=1
// Heading calculations taken from this app note:
// http://www51.honeywell.com/aero/common/documents/myaerospacecatalog-documents/Defense_Brochures-documents/Magnetic__Literature_Application_notes-documents/AN203_Compass_Heading_Using_Magnetometers.pdf
void setAttitudeReadings(float data[], float ax, float ay, float az, float mx, float my, float mz)
{
  float roll = atan2(ay, az);
  float pitch = atan2(-ax, sqrt(ay * ay + az * az));
  
  float heading;
  if (my == 0)
    heading = (mx < 0) ? PI : 0;
  else
    heading = atan2(mx, my);
    
  heading -= DECLINATION * PI / 180;
  
  if (heading > PI) heading -= (2 * PI);
  else if (heading < -PI) heading += (2 * PI);
  else if (heading < 0) heading += 2 * PI;
  
  // Convert everything from radians to degrees:
  data[6] = heading * (180.0 / PI);
  data[7] = pitch * (180.0 / PI);
  data[8] = roll * (180.0 / PI);
}
