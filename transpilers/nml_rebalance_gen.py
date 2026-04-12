#!/usr/bin/env python3
"""
NML Training Data Rebalance Generator

Generates training examples to rebalance the dataset:
1. v0.9 opcodes (TRAIN, INFER, WDECAY, TLOG) - currently 0 examples
2. Weak backward ops (SOFTBK, TANHBK, etc.) - boost to 2,500+ each
3. Other low-count opcodes - boost to reasonable levels

Based on real programs in nml-programs/sports/basketball and domain patterns.
"""

import json
import random
import argparse
from typing import List, Dict, Tuple

# Target counts for rebalancing
REBALANCE_TARGETS = {
    # v0.9 opcodes (currently 0)
    "TRAIN": 1500,
    "INFER": 1200,
    "WDECAY": 600,
    "TLOG": 400,

    # Critical backward ops (currently <500)
    "SOFTBK": 2000,

    # Low backward ops (currently 1000-2000)
    "TANHBK": 1500,
    "ATTNBK": 1500,
    "NORMBK": 1500,
    "CONVBK": 1500,
    "POOLBK": 1500,
    "GELUBK": 1000,
    "RELUBK": 1000,
    "SIGMBK": 800,

    # Extension opcodes that need boost
    "SOFT": 2000,
    "ITOF": 1000,
    "FTOI": 1000,
    "BNOT": 1000,
    "SYNC": 1000,
    "TRAP": 1000,
    "TRNS": 1500,
    "RSHP": 1500,
    "CMPR": 1200,
    "WHER": 1200,
    "EMBD": 1500,
    "UPSC": 1000,
    "PADZ": 1000,
    "SDOT": 1500,
    "SPLT": 1500,
    "CLMP": 1200,
}


class NMLRebalanceGenerator:
    def __init__(self, seed=42):
        random.seed(seed)
        self.examples = []

    def generate_all(self) -> List[Dict]:
        """Generate all rebalancing examples."""
        print("Generating rebalanced training data...")

        # v0.9 opcodes
        self.examples.extend(self.generate_train_examples())
        self.examples.extend(self.generate_infer_examples())
        self.examples.extend(self.generate_wdecay_examples())
        self.examples.extend(self.generate_tlog_examples())

        # Backward opcodes
        self.examples.extend(self.generate_softbk_examples())
        self.examples.extend(self.generate_tanhbk_examples())
        self.examples.extend(self.generate_attnbk_examples())
        self.examples.extend(self.generate_normbk_examples())
        self.examples.extend(self.generate_convbk_examples())
        self.examples.extend(self.generate_poolbk_examples())
        self.examples.extend(self.generate_gelubk_examples())
        self.examples.extend(self.generate_relubk_examples())
        self.examples.extend(self.generate_sigmbk_examples())

        # Extension opcodes
        self.examples.extend(self.generate_soft_examples())
        self.examples.extend(self.generate_type_conversion_examples())
        self.examples.extend(self.generate_data_flow_examples())
        self.examples.extend(self.generate_reduction_examples())
        self.examples.extend(self.generate_vision_examples())
        self.examples.extend(self.generate_transformer_examples())
        self.examples.extend(self.generate_misc_examples())

        print(f"Generated {len(self.examples)} total examples")
        return self.examples

    # ========================================================================
    # v0.9 Opcodes - TRAIN, INFER, WDECAY, TLOG
    # ========================================================================

    def generate_train_examples(self) -> List[Dict]:
        """Generate TRAIN opcode examples (config-driven training)."""
        examples = []
        count = REBALANCE_TARGETS["TRAIN"]

        # Pattern 1: Basic TRAIN with config tensor
        for i in range(count // 4):
            epochs = random.choice([100, 500, 1000, 2000, 5000])
            lr = random.choice([0.001, 0.005, 0.01, 0.02, 0.05])
            optimizer = random.choice([0, 1])  # SGD or Adam
            opt_name = "SGD" if optimizer == 0 else "Adam"

            examples.append({
                "messages": [
                    {"role": "user", "content": f"Write NML to train a network using config: {epochs} epochs, lr={lr}, {opt_name} optimizer"},
                    {"role": "assistant", "content": f"LD    RC @train_config\nTRAIN RC @data @labels\nST    R8 @final_loss\nHALT"}
                ]
            })

        # Pattern 2: TRAIN with early stopping
        for i in range(count // 4):
            epochs = random.choice([1000, 2000, 5000])
            lr = random.choice([0.001, 0.01])
            patience = random.choice([50, 100, 200])
            min_delta = random.choice([0.0001, 0.001, 0.01])

            examples.append({
                "messages": [
                    {"role": "user", "content": f"Train network with early stopping: patience={patience}, min_delta={min_delta}"},
                    {"role": "assistant", "content": f"LEAF  RC #[{epochs},{lr},1,0,{patience},{min_delta}]\nTRAIN RC @train_x @train_y\nST    R8 @loss\nHALT"}
                ]
            })

        # Pattern 3: TRAIN in loop (iterative training)
        for i in range(count // 4):
            steps = random.choice([100, 500, 1000, 5000])
            lr = random.choice([0.001, 0.01, 0.1])

            examples.append({
                "messages": [
                    {"role": "user", "content": f"Train with SGD in {steps} steps, applying weight decay after each step"},
                    {"role": "assistant", "content": f"LD    RC @sgd_config\nLOOP  #{steps}\nTRAIN RC\nWDECAY R1 #0.00001\nWDECAY R3 #0.00001\nENDP\nST    R8 @loss\nHALT"}
                ]
            })

        # Pattern 4: Symbolic syntax variants
        for i in range(count // 4):
            epochs = random.choice([500, 1000])
            lr = random.choice([0.001, 0.01])

            examples.append({
                "messages": [
                    {"role": "user", "content": f"Use TRAIN with symbolic NML syntax (epochs={epochs}, lr={lr}, Adam)"},
                    {"role": "assistant", "content": f"↓  γ  @config\n⟴  γ  @features  @targets\n↑  ρ  @loss\n◼"}
                ]
            })

        return examples[:count]

    def generate_infer_examples(self) -> List[Dict]:
        """Generate INFER opcode examples (forward-only inference)."""
        examples = []
        count = REBALANCE_TARGETS["INFER"]

        # Pattern 1: Single inference
        for i in range(count // 3):
            examples.append({
                "messages": [
                    {"role": "user", "content": "Do forward pass only (no training) and save output"},
                    {"role": "assistant", "content": "LD    R0 @input\nINFER RA\nST    RA @output\nHALT"}
                ]
            })

        # Pattern 2: Batch inference in loop
        for i in range(count // 3):
            batch_size = random.choice([10, 32, 64, 100])

            examples.append({
                "messages": [
                    {"role": "user", "content": f"Train once, then run inference on {batch_size} test samples"},
                    {"role": "assistant", "content": f"LD    RC @config\nTRAIN RC @train_x @train_y\nLD    R0 @test_data\nLOOP  #{batch_size}\nINFER RA R0\nST    RA @predictions\nENDP\nHALT"}
                ]
            })

        # Pattern 3: Symbolic + custom input register
        for i in range(count // 3):
            examples.append({
                "messages": [
                    {"role": "user", "content": "Forward inference with symbolic syntax, custom input register"},
                    {"role": "assistant", "content": "↓  ι  @test_input\n⟶  α  ι\n↑  α  @result\n◼"}
                ]
            })

        return examples[:count]

    def generate_wdecay_examples(self) -> List[Dict]:
        """Generate WDECAY opcode examples (weight decay / L2 reg)."""
        examples = []
        count = REBALANCE_TARGETS["WDECAY"]

        # Pattern 1: WDECAY after weight update
        for i in range(count // 3):
            lambda_val = random.choice([0.00001, 0.0001, 0.0005, 0.001])

            examples.append({
                "messages": [
                    {"role": "user", "content": f"Apply L2 regularization (lambda={lambda_val}) to weights after update"},
                    {"role": "assistant", "content": f"WUPD  R1 RG #0.01\nWDECAY R1 #{lambda_val}\nWUPD  R3 RH #0.01\nWDECAY R3 #{lambda_val}\nHALT"}
                ]
            })

        # Pattern 2: WDECAY in training loop
        for i in range(count // 3):
            lambda_val = random.choice([0.00001, 0.0001, 0.0005])

            examples.append({
                "messages": [
                    {"role": "user", "content": "Manual training loop with weight decay on all layers"},
                    {"role": "assistant", "content": f"LOOP  #1000\nINFER RA R0\nLOSS  RG RA R9\nWUPD  R1 RH #0.01\nWDECAY R1 #{lambda_val}\nWUPD  R3 RI #0.01\nWDECAY R3 #{lambda_val}\nENDP\nHALT"}
                ]
            })

        # Pattern 3: Symbolic syntax
        for i in range(count // 3):
            lambda_val = random.choice([0.00001, 0.0001])

            examples.append({
                "messages": [
                    {"role": "user", "content": "Weight decay on multiple layers (symbolic NML)"},
                    {"role": "assistant", "content": f"ω  κ  #{lambda_val}\nω  μ  #{lambda_val}\nω  ξ  #{lambda_val}\n◼"}
                ]
            })

        return examples[:count]

    def generate_tlog_examples(self) -> List[Dict]:
        """Generate TLOG opcode examples (training log interval)."""
        examples = []
        count = REBALANCE_TARGETS["TLOG"]

        # Pattern 1: TLOG before TRAIN
        for i in range(count // 2):
            interval = random.choice([10, 50, 100, 500])

            examples.append({
                "messages": [
                    {"role": "user", "content": f"Print training progress every {interval} epochs"},
                    {"role": "assistant", "content": f"TLOG  #{interval}\nLD    RC @config\nTRAIN RC @data @labels\nHALT"}
                ]
            })

        # Pattern 2: TLOG with TRAIN
        for i in range(count // 2):
            interval = random.choice([10, 50, 100])

            examples.append({
                "messages": [
                    {"role": "user", "content": "Verbose training with logging, use TRAIN"},
                    {"role": "assistant", "content": f"TLOG  #{interval}\nALLC  RU [6] 1000,0.01,1,0,0,0\nTRAIN RU @data @labels\nHALT"}
                ]
            })

        return examples[:count]

    # ========================================================================
    # Backward Opcodes - Critical boosts
    # ========================================================================

    def generate_softbk_examples(self) -> List[Dict]:
        """Generate SOFTBK examples (softmax backward)."""
        examples = []
        count = REBALANCE_TARGETS["SOFTBK"]

        for i in range(count):
            variant = i % 4

            if variant == 0:
                examples.append({
                    "messages": [
                        {"role": "user", "content": "Compute gradient through softmax layer"},
                        {"role": "assistant", "content": "LD    R0 @d_loss\nLD    R1 @softmax_output\nSOFTBK R2 R0 R1\nST    R2 @d_input\nHALT"}
                    ]
                })
            elif variant == 1:
                examples.append({
                    "messages": [
                        {"role": "user", "content": "Backprop through softmax in classification network"},
                        {"role": "assistant", "content": "LOSS  R4 RA R9 #1\nSOFTBK RG R4 RA\nMMULBK RH RI RG R7 R3\nWUPD  R3 RI #0.01\nHALT"}
                    ]
                })
            elif variant == 2:
                examples.append({
                    "messages": [
                        {"role": "user", "content": "Use SOFTBK in multi-class classifier backward pass (symbolic)"},
                        {"role": "assistant", "content": "Σˈ  RG  R4  RA\n×ˈ  RH  RI  RG  R7  R3\n⟳  μ  RI  #0.01\n◼"}
                    ]
                })
            else:
                examples.append({
                    "messages": [
                        {"role": "user", "content": "Full training step with softmax output layer"},
                        {"role": "assistant", "content": "MMUL  RA R7 R3\nMADD  RA RA R4\nSOFT  RA RA\nLOSS  R8 RA R9 #1\nSOFTBK RG R8 RA\nMMULBK RH RI RG R7 R3\nWUPD  R3 RI #0.01\nWUPD  R4 RG #0.01\nHALT"}
                    ]
                })

        return examples[:count]

    def generate_tanhbk_examples(self) -> List[Dict]:
        """Generate TANHBK examples (tanh backward)."""
        examples = []
        count = REBALANCE_TARGETS["TANHBK"]

        for i in range(count):
            if i % 3 == 0:
                examples.append({
                    "messages": [
                        {"role": "user", "content": "Propagate gradient through tanh activation"},
                        {"role": "assistant", "content": "LD    R0 @d_loss\nLD    R1 @tanh_input\nTANHBK R2 R0 R1\nST    R2 @d_input\nHALT"}
                    ]
                })
            elif i % 3 == 1:
                examples.append({
                    "messages": [
                        {"role": "user", "content": "Use TANH_BK in backpropagation"},
                        {"role": "assistant", "content": "TANH_BK RG R4 R2\nMMULBK RH RI RG R0 R1\nWUPD  R1 RI #0.005\nHALT"}
                    ]
                })
            else:
                examples.append({
                    "messages": [
                        {"role": "user", "content": "Backward pass through tanh (symbolic NML)"},
                        {"role": "assistant", "content": "τˈ  RG  R4  R2\n×ˈ  RH  RI  RG  ι  κ\n⟳  κ  RI  #0.005\n◼"}
                    ]
                })

        return examples[:count]

    def generate_attnbk_examples(self) -> List[Dict]:
        """Generate ATTNBK examples (attention backward)."""
        examples = []
        count = REBALANCE_TARGETS["ATTNBK"]

        for i in range(count):
            examples.append({
                "messages": [
                    {"role": "user", "content": "Compute Q, K, V gradients through attention layer"},
                    {"role": "assistant", "content": "LD    R0 @d_output\nLD    R1 @q_fwd\nLD    R2 @k_fwd\nLD    R3 @v_fwd\nATTNBK R5 R0 R1 R2 R3\nST    R5 @d_q\nST    R6 @d_k\nST    R7 @d_v\nHALT"}
                ]
            })

        return examples[:count]

    def generate_normbk_examples(self) -> List[Dict]:
        """Generate NORMBK examples (layer norm backward)."""
        examples = []
        count = REBALANCE_TARGETS["NORMBK"]

        for i in range(count):
            if i % 2 == 0:
                examples.append({
                    "messages": [
                        {"role": "user", "content": "Gradient through layer normalization"},
                        {"role": "assistant", "content": "LD    R0 @d_output\nLD    R1 @norm_input\nNORMBK R2 R0 R1\nST    R2 @d_input\nHALT"}
                    ]
                })
            else:
                examples.append({
                    "messages": [
                        {"role": "user", "content": "Backprop through NORM in transformer (symbolic)"},
                        {"role": "assistant", "content": "‖ˈ  RG  R4  R2\n×ˈ  RH  RI  RG  ι  κ\n◼"}
                    ]
                })

        return examples[:count]

    def generate_convbk_examples(self) -> List[Dict]:
        """Generate CONVBK examples (convolution backward)."""
        examples = []
        count = REBALANCE_TARGETS["CONVBK"]

        for i in range(count):
            examples.append({
                "messages": [
                    {"role": "user", "content": "Compute input and kernel gradients through convolution"},
                    {"role": "assistant", "content": "LD    R0 @d_output\nLD    R1 @conv_input\nLD    R2 @kernel\nCONVBK R3 R4 R0 R1 R2\nST    R3 @d_input\nST    R4 @d_kernel\nHALT"}
                ]
            })

        return examples[:count]

    def generate_poolbk_examples(self) -> List[Dict]:
        """Generate POOLBK examples (max pool backward)."""
        examples = []
        count = REBALANCE_TARGETS["POOLBK"]

        for i in range(count):
            pool_size = random.choice([2, 3])
            stride = random.choice([2, 3])

            examples.append({
                "messages": [
                    {"role": "user", "content": f"Route gradient through max pool (size={pool_size}, stride={stride})"},
                    {"role": "assistant", "content": f"LD    R0 @d_output\nLD    R1 @pool_input\nPOOLBK R2 R0 R1 #{pool_size} #{stride}\nST    R2 @d_input\nHALT"}
                ]
            })

        return examples[:count]

    def generate_gelubk_examples(self) -> List[Dict]:
        """Generate GELUBK examples (GELU backward)."""
        examples = []
        count = REBALANCE_TARGETS["GELUBK"]

        for i in range(count):
            examples.append({
                "messages": [
                    {"role": "user", "content": "Propagate gradient through GELU activation"},
                    {"role": "assistant", "content": "LD    R0 @d_loss\nLD    R1 @gelu_input\nGELUBK R2 R0 R1\nST    R2 @d_input\nHALT"}
                ]
            })

        return examples[:count]

    def generate_relubk_examples(self) -> List[Dict]:
        """Generate RELUBK examples (ReLU backward)."""
        examples = []
        count = REBALANCE_TARGETS["RELUBK"]

        for i in range(count):
            if i % 2 == 0:
                examples.append({
                    "messages": [
                        {"role": "user", "content": "Compute gradient through ReLU"},
                        {"role": "assistant", "content": "LD    R0 @d_loss\nLD    R1 @relu_input\nRELUBK R2 R0 R1\nST    R2 @d_input\nHALT"}
                    ]
                })
            else:
                examples.append({
                    "messages": [
                        {"role": "user", "content": "Use RELU_BK in backward pass (symbolic)"},
                        {"role": "assistant", "content": "⌐ˈ  RG  R4  R2\n×ˈ  RH  RI  RG  ι  κ\n◼"}
                    ]
                })

        return examples[:count]

    def generate_sigmbk_examples(self) -> List[Dict]:
        """Generate SIGMBK examples (sigmoid backward)."""
        examples = []
        count = REBALANCE_TARGETS["SIGMBK"]

        for i in range(count):
            examples.append({
                "messages": [
                    {"role": "user", "content": "Gradient through sigmoid activation"},
                    {"role": "assistant", "content": "LD    R0 @d_loss\nLD    R1 @sigmoid_input\nSIGMBK R2 R0 R1\nST    R2 @d_input\nHALT"}
                ]
            })

        return examples[:count]

    # ========================================================================
    # Extension Opcodes - General boosts
    # ========================================================================

    def generate_soft_examples(self) -> List[Dict]:
        """Generate SOFT (softmax) examples."""
        examples = []
        count = REBALANCE_TARGETS["SOFT"]

        for i in range(count):
            if i % 3 == 0:
                examples.append({
                    "messages": [
                        {"role": "user", "content": "Apply softmax to output layer for multi-class classification"},
                        {"role": "assistant", "content": "LD    R0 @logits\nSOFT  R1 R0\nST    R1 @probabilities\nHALT"}
                    ]
                })
            elif i % 3 == 1:
                examples.append({
                    "messages": [
                        {"role": "user", "content": "Softmax in classification network"},
                        {"role": "assistant", "content": "MMUL  R3 R0 R1\nMADD  R3 R3 R2\nSOFT  RA R3\nST    RA @class_probs\nHALT"}
                    ]
                })
            else:
                examples.append({
                    "messages": [
                        {"role": "user", "content": "Use softmax (symbolic syntax)"},
                        {"role": "assistant", "content": "Σ  α  R3\n↑  α  @output\n◼"}
                    ]
                })

        return examples[:count]

    def generate_type_conversion_examples(self) -> List[Dict]:
        """Generate ITOF, FTOI, BNOT examples."""
        examples = []

        # ITOF
        for i in range(REBALANCE_TARGETS["ITOF"]):
            examples.append({
                "messages": [
                    {"role": "user", "content": "Convert integer tensor to float"},
                    {"role": "assistant", "content": f"LD    R0 @int_data\nITOF  R1 R0\nST    R1 @float_data\nHALT"}
                ]
            })

        # FTOI
        for i in range(REBALANCE_TARGETS["FTOI"]):
            examples.append({
                "messages": [
                    {"role": "user", "content": "Convert float tensor to integer (truncation)"},
                    {"role": "assistant", "content": f"LD    R0 @float_data\nFTOI  R1 R0\nST    R1 @int_data\nHALT"}
                ]
            })

        # BNOT
        for i in range(REBALANCE_TARGETS["BNOT"]):
            examples.append({
                "messages": [
                    {"role": "user", "content": "Bitwise NOT on integer tensor"},
                    {"role": "assistant", "content": f"LD    R0 @mask\nBNOT  R1 R0\nST    R1 @inverted_mask\nHALT"}
                ]
            })

        return examples

    def generate_data_flow_examples(self) -> List[Dict]:
        """Generate TRNS, RSHP, SPLT examples."""
        examples = []

        # TRNS
        for i in range(REBALANCE_TARGETS["TRNS"]):
            examples.append({
                "messages": [
                    {"role": "user", "content": "Transpose weight matrix"},
                    {"role": "assistant", "content": "LD    R0 @weights\nTRNS  R1 R0\nST    R1 @weights_t\nHALT"}
                ]
            })

        # RSHP
        for i in range(REBALANCE_TARGETS["RSHP"]):
            shape = random.choice(["#[4,8]", "#[16,2]", "#[1,32]"])
            examples.append({
                "messages": [
                    {"role": "user", "content": f"Reshape tensor to {shape}"},
                    {"role": "assistant", "content": f"LD    R0 @data\nRSHP  R1 R0 {shape}\nST    R1 @reshaped\nHALT"}
                ]
            })

        # SPLT
        for i in range(REBALANCE_TARGETS["SPLT"]):
            dim = random.choice([0, 1])
            examples.append({
                "messages": [
                    {"role": "user", "content": f"Split tensor along dimension {dim}"},
                    {"role": "assistant", "content": f"LD    R0 @data\nSPLT  R1 R2 R0 #{dim}\nST    R1 @part1\nST    R2 @part2\nHALT"}
                ]
            })

        return examples

    def generate_reduction_examples(self) -> List[Dict]:
        """Generate CMPR, WHER, CLMP examples."""
        examples = []

        # CMPR
        for i in range(REBALANCE_TARGETS["CMPR"]):
            op = random.choice([0, 1, 2, 3])  # <, <=, >, >=
            thresh = random.choice([0.0, 0.5, 1.0])
            examples.append({
                "messages": [
                    {"role": "user", "content": f"Create mask where values > {thresh}"},
                    {"role": "assistant", "content": f"LD    R0 @data\nCMPR  R1 R0 #{op} #{thresh}\nST    R1 @mask\nHALT"}
                ]
            })

        # WHER
        for i in range(REBALANCE_TARGETS["WHER"]):
            examples.append({
                "messages": [
                    {"role": "user", "content": "Conditional select based on mask"},
                    {"role": "assistant", "content": "LD    R0 @condition\nLD    R1 @true_vals\nLD    R2 @false_vals\nWHER  R3 R0 R1 R2\nST    R3 @result\nHALT"}
                ]
            })

        # CLMP
        for i in range(REBALANCE_TARGETS["CLMP"]):
            min_val = random.choice([0.0, -1.0])
            max_val = random.choice([1.0, 10.0])
            examples.append({
                "messages": [
                    {"role": "user", "content": f"Clamp values between {min_val} and {max_val}"},
                    {"role": "assistant", "content": f"LD    R0 @data\nCLMP  R1 R0 #{min_val} #{max_val}\nST    R1 @clamped\nHALT"}
                ]
            })

        return examples

    def generate_vision_examples(self) -> List[Dict]:
        """Generate UPSC, PADZ examples."""
        examples = []

        # UPSC
        for i in range(REBALANCE_TARGETS["UPSC"]):
            factor = random.choice([2, 4])
            examples.append({
                "messages": [
                    {"role": "user", "content": f"Upscale image by factor {factor}"},
                    {"role": "assistant", "content": f"LD    R0 @image\nUPSC  R1 R0 #{factor}\nST    R1 @upscaled\nHALT"}
                ]
            })

        # PADZ
        for i in range(REBALANCE_TARGETS["PADZ"]):
            pad = random.choice([1, 2])
            examples.append({
                "messages": [
                    {"role": "user", "content": f"Zero-pad image with {pad} pixel border"},
                    {"role": "assistant", "content": f"LD    R0 @image\nPADZ  R1 R0 #{pad}\nST    R1 @padded\nHALT"}
                ]
            })

        return examples

    def generate_transformer_examples(self) -> List[Dict]:
        """Generate EMBD examples."""
        examples = []

        for i in range(REBALANCE_TARGETS["EMBD"]):
            examples.append({
                "messages": [
                    {"role": "user", "content": "Lookup token embeddings from table"},
                    {"role": "assistant", "content": "LD    R0 @embedding_table\nLD    R1 @token_ids\nEMBD  R2 R0 R1\nST    R2 @embeddings\nHALT"}
                ]
            })

        return examples

    def generate_misc_examples(self) -> List[Dict]:
        """Generate SYNC, TRAP, SDOT examples."""
        examples = []

        # SYNC
        for i in range(REBALANCE_TARGETS["SYNC"]):
            examples.append({
                "messages": [
                    {"role": "user", "content": "Synchronize agents in multi-agent system"},
                    {"role": "assistant", "content": "LD    R0 @local_result\nSYNC\nST    R0 @synced_result\nHALT"}
                ]
            })

        # TRAP
        for i in range(REBALANCE_TARGETS["TRAP"]):
            code = random.choice([1, 2, 3])
            examples.append({
                "messages": [
                    {"role": "user", "content": f"Trigger runtime fault with error code {code}"},
                    {"role": "assistant", "content": f"CMPI  RE R0 #0.0\nJMPF  #1\nTRAP  #{code}\nHALT"}
                ]
            })

        # SDOT
        for i in range(REBALANCE_TARGETS["SDOT"]):
            examples.append({
                "messages": [
                    {"role": "user", "content": "Compute dot product of two vectors"},
                    {"role": "assistant", "content": "LD    R0 @vec1\nLD    R1 @vec2\nSDOT  R2 R0 R1\nST    R2 @dot_product\nHALT"}
                ]
            })

        return examples


def main():
    parser = argparse.ArgumentParser(description="Generate rebalanced NML training data")
    parser.add_argument("--output", "-o", default="nml_rebalanced_pairs.jsonl",
                        help="Output JSONL file")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    generator = NMLRebalanceGenerator(seed=args.seed)
    examples = generator.generate_all()

    # Write to JSONL
    with open(args.output, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\n✅ Wrote {len(examples)} examples to {args.output}")

    # Print distribution
    print("\n📊 DISTRIBUTION BY OPCODE:")
    opcode_counts = {}
    for ex in examples:
        content = ""
        for msg in ex["messages"]:
            content += msg["content"] + " "

        for opcode in REBALANCE_TARGETS.keys():
            if opcode in content:
                opcode_counts[opcode] = opcode_counts.get(opcode, 0) + 1

    for opcode, count in sorted(opcode_counts.items(), key=lambda x: -x[1]):
        target = REBALANCE_TARGETS.get(opcode, 0)
        print(f"  {opcode:10s} {count:5d} examples (target: {target})")


if __name__ == "__main__":
    main()
