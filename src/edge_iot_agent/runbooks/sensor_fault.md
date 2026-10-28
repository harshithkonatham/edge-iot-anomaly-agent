# Runbook: sensor fault

**Signature:** a sensor reports exactly the same value for many readings in a row (flatline),
jumps to an impossible value, or disagrees with related sensors on the same motor. The motor
itself may be healthy.

**Actions**
1. Treat that sensor's readings as untrusted until fixed; rely on the other sensors.
2. Check the sensor's power, cable and connector; look for a stuck gateway buffer.
3. Compare the other sensors on the motor with the fleet to confirm the motor is healthy.
4. Recalibrate or replace the sensor.
5. Raise the priority if the stuck sensor is the one protecting the motor (e.g. temperature).
