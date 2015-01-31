CC			= g++
CFLAGS		= -std=c++14 -Wall -march=native `pkg-config --cflags opencv`
LINKFLAGS	= -lboost_filesystem -lboost_system `pkg-config --libs opencv`
SRCS		= main.cpp
OBJS		= $(SRCS:.cpp=.o)
PROG		= main

all: $(SRCS) $(PROG)

$(PROG): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $@ $(INCFLAGS) $(LINKFLAGS)

.cpp.o:
	$(CC) $(CFLAGS) $< -c -o $@ $(INCFLAGS)

clean:
	rm $(OBJS) $(PROG)