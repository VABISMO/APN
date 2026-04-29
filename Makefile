CC      = gcc
CFLAGS  = -O3 -march=native -ffast-math -funroll-loops -fopenmp -mfma
LDFLAGS = -lm
TARGET  = apn

# Auto-detect AVX-512
AVX512 := $(shell echo | $(CC) -mavx512f -x c -E - 2>/dev/null && echo 1 || echo 0)
ifeq ($(AVX512),1)
  CFLAGS += -mavx512f -mavx512dq -mavx512bw
  $(info [APN] AVX-512 detected - using SIMD acceleration)
else
  $(info [APN] No AVX-512 - using AVX2/FMA fallback)
endif

SRCS = apn_main.c
HDRS = src/tensor.h src/optimizer.h src/apn_layer.h src/attention.h \
       src/transformer.h src/tokenizer.h src/generate.h src/train.h

.PHONY: all clean bench demo info test install-py

all: $(TARGET)
	@echo ""
	@echo "  ╔══════════════════════════════════════════════════════╗"
	@echo "  ║  APN v9 built successfully!                           ║"
	@echo "  ╠══════════════════════════════════════════════════════╣"
	@echo "  ║  ./apn bench              APN vs SwiGLU bench         ║"
	@echo "  ║  ./apn train --data f.txt  Train from scratch        ║"
	@echo "  ║  ./apn generate --model m.apn --prompt 'Hi'          ║"
	@echo "  ║  ./apn chat    --model m.apn  Interactive chat      ║"
	@echo "  ║  ./apn info    --model m.apn  Show model info        ║"
	@echo "  ╠══════════════════════════════════════════════════════╣"
	@echo "  ║  HuggingFace conversion (needs: pip install torch    ║"
	@echo "  ║                          transformers safetensors):  ║"
	@echo "  ║  python3 apn_complete.py convert                      ║"
	@echo "  ║    --model google/gemma-3-2b-it --out gemma3_apn     ║"
	@echo "  ╚══════════════════════════════════════════════════════╝"
	@echo ""

$(TARGET): $(SRCS) $(HDRS)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRCS) $(LDFLAGS)
	@echo "  Built: $(TARGET)"

bench: $(TARGET)
	./$(TARGET) bench

demo: $(TARGET)
	@python3 -c "\
import random, math; random.seed(42); lines=[]\n\
for i in range(1,40):\n\
  for j in range(1,40): lines+=[f'{i}+{j}={i+j}',f'{i}*{j}={i*j}']\n\
words='the cat dog ran big small fast slow over under and but'.split()\n\
for _ in range(3000): lines.append(' '.join(random.choices(words,k=random.randint(4,9))))\n\
open('examples/demo_corpus.txt','w').write('\n'.join(lines*6))\n\
print('Generated examples/demo_corpus.txt')\n\
"
	./$(TARGET) train \
		--data examples/demo_corpus.txt \
		--out  demo.apn \
		--d_model 64 --n_layers 2 --n_heads 4 --ffn_hidden 256
	./$(TARGET) generate --model demo.apn --prompt "the cat" --max_tokens 60

info:
	@echo "System info:"
	@echo "  GCC: $$(gcc --version | head -1)"
	@echo "  CPU: $$(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2)"
	@echo "  Cores: $$(nproc)"
	@echo "  AVX-512: $(AVX512)"

test: $(TARGET)
	./$(TARGET) bench
	@echo "All tests passed"

install-py:
	pip install torch transformers accelerate safetensors sentencepiece

debug: CFLAGS = -g -O0 -fsanitize=address,undefined -fopenmp
debug: $(TARGET)

clean:
	rm -f $(TARGET) *.apn demo_corpus.txt
	rm -rf __pycache__ *_apn *_apn_trained