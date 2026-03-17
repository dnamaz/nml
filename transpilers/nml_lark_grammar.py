"""
NML Lark Grammar — for use with Outlines CFG constrained decoding.
Guarantees 100% syntactic correctness by construction.
"""

NML_GRAMMAR = r"""
?start: program

program: (line "\n")* line?

line: comment | instruction trailing_comment? | meta_line trailing_comment? | data_line trailing_comment?

comment: /;[^\n]+/
trailing_comment: /\s*;[^\n]*/

meta_line: "META" WS /@[a-z_][a-z0-9_]*/ WS /\"[^\"]*\"/
         | "§" WS /@[a-z_][a-z0-9_]*/ WS /\"[^\"]*\"/

data_line: /@[a-z_][a-z0-9_]*/ WS /shape=[0-9,x]+/ (WS /dtype=[a-z0-9]+/)? WS /data=-?[0-9][0-9.e,\-]*/

instruction: arith_instr | activ_instr | memory_instr | data_flow_instr
           | compare_instr | branch_instr | loop_instr | call_instr
           | tree_instr | system_instr | general_instr
           | vision_instr | transformer_instr | reduction_instr | signal_instr
           | m2m_instr | training_instr

arith_instr: ARITH_OP WS register WS register WS register
           | ARITH_OP WS register WS register WS immediate
ARITH_OP: "MMUL" | "MADD" | "MSUB" | "EMUL" | "EDIV" | "SDOT" | "DOT" | "SCLR" | "SDIV"
        | "×" | "⊕" | "⊖" | "⊗" | "⊘" | "·" | "∗" | "÷"

activ_instr: ACTIV_OP WS register WS register
ACTIV_OP: "RELU" | "SIGM" | "TANH" | "SOFT" | "GELU"
        | "⌐" | "σ" | "τ" | "Σ" | "ℊ"

memory_instr: LOAD_OP WS register WS operand
            | "ST" WS register WS memory_ref
            | "↑" WS register WS memory_ref
            | "STORE" WS register WS memory_ref
            | "ALLC" WS register WS shape_literal
            | "□" WS register WS shape_literal
            | "ALLOCATE" WS register WS shape_literal
LOAD_OP: "LD" | "LEAF" | "MOV" | "LOAD" | "SET_VALUE" | "COPY"
       | "↓" | "∎" | "←"

data_flow_instr: DFLOW_OP WS register WS register (WS operand)?
DFLOW_OP: "RSHP" | "TRNS" | "SPLT" | "MERG" | "RESHAPE" | "TRANSPOSE" | "SPLIT" | "MERGE"
        | "⊞" | "⊤" | "⊢" | "⊣"

compare_instr: "CMP" WS register WS register
             | "≶" WS register WS register
             | CMPI_OP WS register WS register WS immediate
             | CMPI_OP WS register WS immediate
             | CMPF_OP WS register WS register WS immediate WS immediate
             | CMPF_OP WS register WS register WS register
             | CMPF_OP WS register WS register WS immediate
CMPI_OP: "CMPI" | "COMPARE_VALUE" | "≺" | "ϟ"
CMPF_OP: "CMPF" | "COMPARE_FEATURE" | "⋈"

branch_instr: BRANCH_OP WS immediate
BRANCH_OP: "JMPT" | "JMPF" | "JUMP" | "JMP"
         | "↗" | "↘" | "→"
         | "BRANCH_TRUE" | "BRANCH_FALSE" | "BRANCH"

loop_instr: LOOP_OP WS operand
          | ENDP_OP
LOOP_OP: "LOOP" | "REPEAT" | "↻"
ENDP_OP: "ENDP" | "END_REPEAT" | "↺"

call_instr: "CALL" WS immediate
          | "⇒" WS immediate
          | "RET" | "⇐" | "RETURN"

tree_instr: TACC_OP WS register WS register WS register
          | TACC_OP WS register WS register
TACC_OP: "TACC" | "∑" | "ACCUMULATE"

system_instr: HALT_OP
            | "SYNC" | "⏸" | "BARRIER"
            | "TRAP" WS immediate
            | "⚠" WS immediate
            | "TRAP" | "⚠"
HALT_OP: "HALT" | "◼" | "STOP"

general_instr: "SYS" WS register WS immediate
             | "⚙" WS register WS immediate
             | "MOD" WS register WS register WS register
             | "ITOF" WS register WS register
             | "FTOI" WS register WS register
             | "BNOT" WS register WS register
             | "⊶" WS register WS register
             | "⊷" WS register WS register
             | "¬" WS register WS register

vision_instr: CONV_OP WS register WS register WS register (WS immediate)*
            | POOL_OP WS register WS register (WS immediate)*
            | UPSC_OP WS register WS register (WS immediate)?
            | PADZ_OP WS register WS register (WS immediate)?
CONV_OP: "CONV" | "⊛" | "CONVOLVE"
POOL_OP: "POOL" | "⊓" | "MAX_POOL"
UPSC_OP: "UPSC" | "⊔" | "UPSCALE"
PADZ_OP: "PADZ" | "⊡" | "ZERO_PAD"

transformer_instr: ATTN_OP WS register WS register WS register (WS register)?
                 | NORM_OP WS register WS register (WS register)*
                 | EMBD_OP WS register WS register WS register
ATTN_OP: "ATTN" | "⊙" | "ATTENTION"
NORM_OP: "NORM" | "‖" | "LAYER_NORM"
EMBD_OP: "EMBD" | "⊏" | "EMBED"

reduction_instr: RDUC_OP WS register WS register (WS immediate)*
              | WHER_OP WS register WS register WS register (WS register)?
              | CLMP_OP WS register WS register WS immediate WS immediate
              | CMPR_OP WS register WS register WS immediate WS immediate
RDUC_OP: "RDUC" | "⊥" | "ϛ" | "REDUCE"
WHER_OP: "WHER" | "⊻" | "WHERE"
CLMP_OP: "CLMP" | "⊧" | "CLAMP"
CMPR_OP: "CMPR" | "⊜" | "MASK_COMPARE"

signal_instr: FFT_OP WS register WS register WS register
            | FILT_OP WS register WS register WS register
FFT_OP: "FFT" | "∿" | "FOURIER"
FILT_OP: "FILT" | "⋐" | "FILTER"

m2m_instr: "FRAG" WS /[a-z_][a-z0-9_]*/
         | "◆" WS /[a-z_][a-z0-9_]*/
         | "ENDF" | "◇"
         | "LINK" WS memory_ref
         | VOTE_OP WS register WS register WS immediate (WS immediate)?
         | PROJ_OP WS register WS register WS register
         | DIST_OP WS register WS register WS register (WS immediate)?
         | GATH_OP WS register WS register WS register
         | SCAT_OP WS register WS register WS register
         | SIGN_LINE
         | VRFY_OP WS memory_ref WS memory_ref
VOTE_OP: "VOTE" | "⚖" | "CONSENSUS"
PROJ_OP: "PROJ" | "⟐" | "PROJECT"
DIST_OP: "DIST" | "⟂" | "DISTANCE"
GATH_OP: "GATH" | "⊃" | "GATHER"
SCAT_OP: "SCAT" | "SCTR" | "⊂" | "SCATTER"
VRFY_OP: "VRFY" | "✓" | "VERIFY_SIGNATURE"
SIGN_LINE: /[✦][ \t]+agent=[^\n]+/
         | /SIGN[ \t]+agent=[^\n]+/

training_instr: BKWD_OP WS register WS register WS register (WS register)?
              | WUPD_OP WS register WS register WS operand
              | WUPD_OP WS register WS register WS register WS operand
              | LOSS_OP WS register WS register WS register (WS immediate)?
              | TNET_OP (WS operand)+
              | ACT_BK_OP WS register WS register WS register
              | MMULBK_OP WS register WS register WS register WS register WS register
              | CONVBK_OP WS register WS register WS register WS register WS register
              | POOLBK_OP WS register WS register WS register (WS immediate (WS immediate)?)?
              | NORMBK_OP WS register WS register WS register
              | ATTNBK_OP WS register WS register WS register WS register (WS register)?
              | TNDEEP_OP (WS operand)+
BKWD_OP: "BKWD" | "∇" | "BACKWARD"
WUPD_OP: "WUPD" | "⟳" | "WEIGHT_UPDATE"
LOSS_OP: "LOSS" | "△" | "COMPUTE_LOSS"
TNET_OP: "TNET" | "⥁" | "TRAIN_NETWORK"
ACT_BK_OP: "RELUBK" | "⌐ˈ" | "RELU_BACKWARD" | "RELU_BK"
          | "SIGMBK" | "σˈ" | "SIGMOID_BACKWARD" | "SIGM_BK"
          | "TANHBK" | "τˈ" | "TANH_BACKWARD" | "TANH_BK"
          | "GELUBK" | "ℊˈ" | "GELU_BACKWARD" | "GELU_BK"
          | "SOFTBK" | "Σˈ" | "SOFTMAX_BACKWARD" | "SOFT_BK"
MMULBK_OP: "MMULBK" | "×ˈ" | "MATMUL_BACKWARD" | "MMUL_BK"
CONVBK_OP: "CONVBK" | "⊛ˈ" | "CONV_BACKWARD" | "CONV_BK"
POOLBK_OP: "POOLBK" | "⊓ˈ" | "POOL_BACKWARD" | "POOL_BK"
NORMBK_OP: "NORMBK" | "‖ˈ" | "NORM_BACKWARD" | "NORM_BK"
ATTNBK_OP: "ATTNBK" | "⊙ˈ" | "ATTN_BACKWARD" | "ATTN_BK"
TNDEEP_OP: "TNDEEP" | "⥁ˈ" | "TRAIN_DEEP"

operand: register | immediate | memory_ref | shape_literal

register: REGISTER
REGISTER: /R[0-9]/i | /R[A-V]/i
        | "ι" | "κ" | "λ" | "μ" | "ν" | "ξ" | "ο" | "π" | "ρ" | "ς"
        | "α" | "β" | "γ" | "δ" | "φ" | "ψ"
        | "η" | "θ" | "ζ" | "ω" | "χ" | "υ" | "ε"
        | "ACCUMULATOR" | "GENERAL" | "SCRATCH" | "COUNTER" | "FLAG" | "STACK"
        | "INPUT" | "KERNEL" | "LAYER" | "MOMENTUM"

immediate: IMMEDIATE
IMMEDIATE: /#-?[0-9]+(\.[0-9]+)?/
         | /-?[0-9]+(\.[0-9]+)?/

memory_ref: MEMORY_REF
MEMORY_REF: /@[a-z_][a-z0-9_]*/

shape_literal: SHAPE
SHAPE: /#?\[[0-9]+(,[0-9]+)*\]/

WS: /[ \t]+/
"""

if __name__ == "__main__":
    from lark import Lark
    parser = Lark(NML_GRAMMAR, parser="earley")

    tests = [
        "LEAF R0 #42.0\nHALT",
        "LD R0 @input\nMMUL R2 R0 R1\nST R2 @result\nHALT",
        "∎ ι #10.0\n∑ α ι κ\n↑ α @result\n◼",
        "CMPI R0 #100\nJMPT #2\nLEAF RA #0.0\nHALT",
        "TACC R5 R0\nTACC R5 R1\nHALT",
        "TNET #1000 #0.01 #0\nHALT",
        "TNET R0 R1 R2 R3 R4 R5 #1000 #0.01\nHALT",
        "LEAF R4 R2\nST R4 @result\nHALT",
        "META @name \"test\"\nLD R0 @input\nHALT",
        "WUPD R0 R0 R7 R4\nHALT",
        "RELUBK R2 R1 R0\nHALT",
        "MMULBK R3 R4 R0 R1 R2\nHALT",
        "POOLBK R2 R0 R1 #2 #2\nHALT",
        "ATTNBK R5 R0 R1 R2 R3\nHALT",
        "TNDEEP #500 #0.01 #0\nHALT",
    ]

    passed = 0
    for t in tests:
        try:
            parser.parse(t)
            print(f"  PASS: {t[:50]}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {t[:50]}")
            print(f"        {str(e)[:100]}")

    print(f"\n  {passed}/{len(tests)} passed")
