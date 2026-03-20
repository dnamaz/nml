# ═══════════════════════════════════════════
# NML — Neural Machine Language
# Makefile for v0.6.4
# ═══════════════════════════════════════════

CC      = gcc
CFLAGS  = -O2 -Wall -std=c99
LDFLAGS = -lm

# ═══════════════════════════════════════════
# Build
# ═══════════════════════════════════════════

nml: runtime/nml.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)
	@echo "  Built: nml (v0.7.0, 82 instructions, 32 registers — portable)"

nmld: runtime/nmld.c runtime/nml.c
	$(CC) $(CFLAGS) -o $@ runtime/nmld.c $(LDFLAGS)
	@echo "  Built: nmld (NML daemon — generic execution server)"

nml-fast: runtime/nml.c
ifeq ($(shell uname),Darwin)
	$(CC) -O3 -march=native -std=c99 -DNML_USE_ACCELERATE -DACCELERATE_NEW_LAPACK -o $@ $< -lm -framework Accelerate
	@echo "  Built: nml-fast (v0.7.0, BLAS via Apple Accelerate)"
else
	$(CC) -O3 -march=native -std=c99 -DNML_USE_OPENBLAS -o $@ $< -lm -lopenblas
	@echo "  Built: nml-fast (v0.7.0, BLAS via OpenBLAS)"
endif

nml-metal: runtime/nml.c
ifeq ($(shell uname),Darwin)
	clang -O3 -march=native -x objective-c -D_DARWIN_C_SOURCE \
	    -DNML_USE_METAL -DNML_USE_ACCELERATE -DACCELERATE_NEW_LAPACK \
	    -framework Metal -framework MetalPerformanceShaders \
	    -framework Accelerate -framework Foundation \
	    -o $@ $< -lm -Wno-deprecated-declarations
	@echo "  Built: nml-metal (v0.8.0, Metal GPU + BLAS via Accelerate)"
else
	@echo "  Error: Metal requires macOS. Use nml-fast for CPU acceleration."
	@exit 1
endif

nml-fmt: runtime/nml_fmt.c runtime/nml_fmt.h
	$(CC) $(CFLAGS) -o $@ runtime/nml_fmt.c
	@echo "  Built: nml-fmt (syntax converter: classic ↔ symbolic ↔ verbose)"

nml-crypto: runtime/nml.c runtime/nml_crypto.h runtime/tweetnacl.c runtime/tweetnacl.h
	$(CC) $(CFLAGS) -DNML_CRYPTO -o $@ runtime/nml.c runtime/tweetnacl.c $(LDFLAGS)
	@echo "  Built: nml-crypto (v0.7.0, Ed25519 + HMAC-SHA256 signing)"

nml-wasm: runtime/nml.c
	emcc -O2 -std=c99 -o terminal/nml.js runtime/nml.c -lm \
	    -s MODULARIZE=1 -s EXPORT_NAME='NMLModule' \
	    -s EXPORTED_RUNTIME_METHODS='["callMain","FS"]' \
	    -s ALLOW_MEMORY_GROWTH=1
	@echo "  Built: nml-wasm (terminal/nml.js + terminal/nml.wasm)"

release: nml
	strip nml
	@echo "  nml: $$(wc -c < nml | tr -d ' ') bytes"

# ═══════════════════════════════════════════
# Core tests
# ═══════════════════════════════════════════

test-anomaly: nml
	./nml programs/anomaly_detector.nml programs/anomaly_weights.nml.data

test-extensions: nml
	./nml programs/extension_demo.nml programs/extension_demo.nml.data

test-symbolic: nml
	@echo "--- Symbolic anomaly detector ---"
	./nml tests/test_symbolic.nml programs/anomaly_weights.nml.data 2>&1 | grep -E "(HALTED|anomaly_score)"
	@echo "--- Symbolic features ---"
	./nml tests/test_symbolic_features.nml 2>&1 | grep -E "(HALTED|sdiv|cmpi|cmp_|sum_|call_)"

test-verbose: nml
	./nml tests/test_verbose.nml 2>&1 | grep -E "(HALTED|tax_amount)"

test-features: nml
	./nml tests/test_features.nml 2>&1 | grep -E "(HALTED|sdiv|cmpi|cmp_|sum_|call_)"

test-hello: nml
	@echo "--- Hello World (NML-G) ---"
	./nml programs/hello_world.nml 2>&1 | grep -E "(Hello|HALTED)"

test-fibonacci: nml
	@echo "--- Fibonacci (NML-G) ---"
	./nml programs/fibonacci.nml 2>&1 | grep -E "^[0-9]" | head -5
	@echo "  ..."

test-fizzbuzz: nml
	@echo "--- FizzBuzz (NML-G) ---"
	./nml programs/fizzbuzz.nml 2>&1 | grep -E "^-?[0-9]" | head -5
	@echo "  ..."

test-primes: nml
	@echo "--- Primes (NML-G) ---"
	./nml programs/primes.nml 2>&1 | grep -E "^[0-9]"

test-gp: test-hello test-fibonacci test-fizzbuzz test-primes
	@echo ""
	@echo "  NML-G tests passed."

test: test-anomaly test-extensions test-symbolic test-verbose test-features test-gp
	@echo ""
	@echo "  All core tests passed."

# ═══════════════════════════════════════════
# Domain targets (require domain/ populated)
# ═══════════════════════════════════════════

domain-test-tax: nml
	@echo "--- Junior Developer ---"
	@echo "@employee_data shape=1,8 data=65000.0,0.0,0.0,0.03,0.0,26.0,0.06,2.0" > /tmp/nml_test.data
	./nml domain/programs/tax_calculator.nml /tmp/nml_test.data 2>&1 | grep -E "(HALTED|net_pay)"

domain-transpile-scan:
	cd domain/transpilers && python3 domain_transpiler.py scan

domain-transpile-library: nml
	cd domain/transpilers && python3 domain_build_library.py --validate

domain-transpile-library-symbolic: nml
	cd domain/transpilers && python3 domain_build_library.py --syntax symbolic --no-comments

domain-train:
	cd domain/transpilers && python3 tax_pipeline.py

domain-benchmark:
	cd domain/transpilers && python3 benchmark.py

domain-prepare-training:
	cd domain/transpilers && python3 finetune_pipeline.py \
		--inputs ../output/training/nml_code_pairs.jsonl \
		         ../output/training/all_gaps_combined.jsonl \
		         ../output/training/rag_gaps.jsonl \
		         ../output/training/constants_pairs.jsonl \
		         ../output/training/nml_syntax.jsonl \
		--output-dir ../output/training/mlx-combined \
		--prepare-only

domain-finetune:
	cd domain/transpilers && python3 finetune_pipeline.py \
		--inputs ../output/training/nml_code_pairs.jsonl \
		         ../output/training/all_gaps_combined.jsonl \
		         ../output/training/rag_gaps.jsonl \
		         ../output/training/constants_pairs.jsonl \
		         ../output/training/nml_syntax.jsonl \
		--base-model ../output/model/Mistral-7B-Instruct-v0.3-4bit \
		--output-dir ../output/training/mlx-combined \
		--adapter-dir ../output/model/nml-combined-adapters \
		--train

domain-finetune-merge:
	cd domain/transpilers && python3 finetune_pipeline.py \
		--base-model ../output/model/Mistral-7B-Instruct-v0.3-4bit \
		--adapter-dir ../output/model/nml-combined-adapters \
		--merge-to ../output/model/nml-combined-merged \
		--merge-only

domain-rag-server:
	cd domain/transpilers && python3 domain_rag_server.py --domains tax

# ═══════════════════════════════════════════
# Agent services
# ═══════════════════════════════════════════

agent-start: nml
	bash serve/start_agents.sh

agent-start-headless: nml
	bash serve/start_agents.sh --no-ui

agent-gateway: nml
	cd domain/transpilers && python3 domain_rag_server.py --domains tax

agent-status:
	@curl -s localhost:8082/health 2>/dev/null | python3 -m json.tool || echo "  NML Server: OFFLINE"
	@curl -s localhost:8083/health 2>/dev/null | python3 -m json.tool || echo "  Gateway: OFFLINE"

# ═══════════════════════════════════════════
# Clean
# ═══════════════════════════════════════════

clean:
	rm -f nml
	rm -f /tmp/nml_test.data

# ═══════════════════════════════════════════
# Help
# ═══════════════════════════════════════════

help:
	@echo ""
	@echo "  NML v0.6.4 — Neural Machine Language"
	@echo "  ═════════════════════════════════════"
	@echo ""
	@echo "  Build:"
	@echo "    make nml              Build the NML runtime (82 instructions, all extensions)"
	@echo "    make release          Build + strip"
	@echo ""
	@echo "  Test (core):"
	@echo "    make test             Run all core tests"
	@echo "    make test-anomaly     Anomaly detection (neural net)"
	@echo "    make test-extensions  Extension demo"
	@echo "    make test-symbolic    Symbolic syntax tests"
	@echo "    make test-verbose     Verbose syntax test"
	@echo "    make test-features    Core features (SDIV, CMP, CALL/RET, backward jumps)"
	@echo "    make test-gp          General-purpose (hello world, fibonacci, fizzbuzz, primes)"
	@echo ""
	@echo "  Domain (requires domain/ populated):"
	@echo "    make domain-test-tax                  Tax calculator test"
	@echo "    make domain-transpile-scan             Scan tax-data/ and classify"
	@echo "    make domain-transpile-library          Build + validate full NML tax library"
	@echo "    make domain-transpile-library-symbolic Build library in symbolic syntax"
	@echo "    make domain-prepare-training           Combine JSONL → train/valid splits"
	@echo "    make domain-finetune                   Prepare + LoRA fine-tune (Mistral 7B)"
	@echo "    make domain-finetune-merge             Merge LoRA adapters into base model"
	@echo "    make domain-rag-server                 Start multi-domain RAG server"
	@echo ""
    @echo "  Agent services:"
	@echo "    make agent-start      Start NML server + gateway + chat UI"
	@echo "    make agent-start-headless  Start NML server + gateway (no UI)"
	@echo "    make agent-gateway    Start domain RAG gateway only"
	@echo "    make agent-status     Check health of running services"
	@echo ""
	@echo "  Other:"
	@echo "    make clean            Remove built binary"
	@echo "    make help             Show this message"
	@echo ""

.PHONY: nml release test test-anomaly test-extensions test-symbolic test-verbose test-features test-hello test-fibonacci test-fizzbuzz test-primes test-gp domain-test-tax domain-transpile-scan domain-transpile-library domain-transpile-library-symbolic domain-train domain-benchmark domain-prepare-training domain-finetune domain-finetune-merge domain-rag-server agent-start agent-start-headless agent-gateway agent-status clean help
