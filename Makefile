w.exe: w.o
	g++ w.o -o w.exe

w.o:
	g++ -c -g w.cpp -o w.o -std=c++11

clean:
	rm *.o
	rm *.exe
