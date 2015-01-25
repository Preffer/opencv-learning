CC			= g++
CFLAGS		= -std=c++14 -D_REENTRANT -Wall -march=native -O2 `pkg-config --cflags opencv`
LINKFLAGS	= -lboost_thread -lboost_program_options -lboost_filesystem -lboost_system `pkg-config --libs opencv`
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