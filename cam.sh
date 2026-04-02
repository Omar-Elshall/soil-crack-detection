gst-launch-1.0 nvarguscamerasrc sensor-id=0 sensor-mode=2 wbmode=5 ! 'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=60/1' ! nvvidconv ! nveglglessink
