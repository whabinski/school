CC = gcc
CFLAGS = -Wall
BUILD_DIR = build
TEST_DIR = test
SRC_DIR = src
TEST_DATA = test_data


all_targets = multiply square

all: $(all_targets)
	mv library.o $(BUILD_DIR)

#all: multiply square
#	mv library.o build

$(all_targets): library.o
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -o ./$(BUILD_DIR)/$(@) $(SRC_DIR)/$(@).c library.o
	
#multiply: library.o
#	mkdir -p build
#	gcc -o multiply multiply.c library.o

#square: library.o
#	mkdir -p build
#	gcc -o square square.c library.o

library.o: ./$(SRC_DIR)/library.c
	$(CC) -c ./$(SRC_DIR)/library.c

#library.o: library.c
#	gcc -c library.c

test: 
	mkdir -p $(TEST_DIR)

	./$(BUILD_DIR)/square inches < ./$(TEST_DATA)/square_input.txt > ./$(TEST_DIR)/square_output.txt
	diff ./$(TEST_DATA)/square_expected.txt ./$(TEST_DIR)/square_output.txt

	./$(BUILD_DIR)/multiply 2 < ./$(TEST_DATA)/multiply_input.txt > ./$(TEST_DIR)/multiply_output.txt
	diff ./$(TEST_DATA)/multiply_expected.txt ./$(TEST_DIR)/multiply_output.txt

	rm -f -r $(TEST_DIR)

clean:
	rm -f -r $(BUILD_DIR) $(TEST_DIR)
	



