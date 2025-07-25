# Project Settings

TARGET := bin/myapp
SRC_DIR := src
BUILD_DIR := build

# Compiler Settings

CXX := g++
CXXFLAGS := -Wall -Wextra -std=c++11


# Source Files
SRCS := $(wildcard $(SRC_DIR)/*.cpp)
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRCS))

# Targets

.PHONY: all clean dirs

all: dirs $(TARGET)

# Final Binary
$(TARGET): $(OBJS)
	@echo "Linking..."
	@$(CXX) $(CXXFLAGS) -o $@ $^

# Compile each .cpp into .o
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo "Compiling $<"
	@$(CXX) $(CXXFLAGS) -c $< -o $@

# Create output directories
dirs:
	@mkdir -p $(BUILD_DIR) bin

# Clean build artifacts
clean:
	@echo "Cleaning project..."
	@rm -rf $(BUILD_DIR) bin
