CC      = gcc
CFLAGS  = -O3 -march=native -ffast-math -funroll-loops -fopenmp -mfma
LDFLAGS = -lm
TARGET  = probnet

# Auto-detect AVX-512
AVX512 := $(shell echo | $(CC) -mavx512f -x c -E - 2>/dev/null && echo 1 || echo 0)
ifeq ($(AVX512),1)
  CFLAGS += -mavx512f -mavx512dq -mavx512bw
  $(info [ProbNet] AVX-512 detected - using SIMD acceleration)
else
  $(info [ProbNet] No AVX-512 - using AVX2/FMA fallback)
endif

SRCS = probnet_main.c
HDRS = src/tensor.h src/optimizer.h src/apn_layer.h src/attention.h \
       src/transformer.h src/tokenizer.h src/generate.h src/train.h

.PHONY: all clean bench demo info test install-py

all: $(TARGET)
	@echo ""
	@echo "  ╔══════════════════════════════════════════════════════╗"
	@echo "  ║  ProbNet v9 built successfully!                      ║"
	@echo "  ╠══════════════════════════════════════════════════════╣"
	@echo "  ║  ./probnet bench              APN vs SwiGLU bench    ║"
	@echo "  ║  ./probnet train --data f.txt  Train from scratch    ║"
	@echo "  ║  ./probnet generate --model m.pnet --prompt 'Hi'     ║"
	@echo "  ║  ./probnet chat    --model m.pnet  Interactive chat  ║"
	@echo "  ║  ./probnet info    --model m.pnet  Show model info   ║"
	@echo "  ╠══════════════════════════════════════════════════════╣"
	@echo "  ║  HuggingFace conversion (needs: pip install torch    ║"
	@echo "  ║                          transformers safetensors):  ║"
	@echo "  ║  python3 probnet_complete.py convert                 ║"
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
		--out  demo.pnet \
		--d_model 64 --n_layers 2 --n_heads 4 --ffn_hidden 256
	./$(TARGET) generate --model demo.pnet --prompt "the cat" --max_tokens 60

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
	rm -f $(TARGET) *.pnet demo_corpus.txt
	rm -rf __pycache__ *_apn *_apn_trained
