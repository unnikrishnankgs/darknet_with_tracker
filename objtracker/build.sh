g++ multitracker.cpp `pkg-config --cflags --libs opencv` --shared -o libsjtracker.so -fPIC -I../darknet_track/include/
