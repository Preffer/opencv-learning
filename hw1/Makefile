CC			= g++
CFLAGS		= -std=c++11 -Wall -march=native `pkg-config --cflags opencv freetype2`
LINKFLAGS	= `pkg-config --libs opencv freetype2`
SRCS		= $(wildcard *.cpp)
OBJS		= $(SRCS:.cpp=.o)
PROG		= main

all: $(SRCS) $(PROG)

$(PROG): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $@ $(INCFLAGS) $(LINKFLAGS)

.cpp.o:
	$(CC) $(CFLAGS) $< -c -o $@ $(INCFLAGS)

clean:
	rm $(OBJS) $(PROG)